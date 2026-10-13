"""Immutable, bounded-memory training indexes over committed paired tar caches."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset, Sampler

from h3dmd.mapper_pretraining import load_pair
from h3dmd.stride_cache import sha_file, atomic_json



def _manifest_records(event):
    manifest = Path(event['manifest'])
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != event['manifest_sha256']:
        raise ValueError(f'Manifest checksum mismatch: {manifest}')
    ids, records, offset = set(), [], 0
    for line in raw.splitlines(keepends=True):
        row = json.loads(line)
        if row['split'] not in {'train', 'val'} or row['seconds'] not in {5, 10, 15}:
            raise ValueError('Unexpected split or duration')
        if row['id'] in ids:
            raise ValueError('Duplicate pair within a committed manifest')
        ids.add(row['id'])
        if event['kind'] == 'chunk' and row['cache_path'] != event['archive']:
            raise ValueError('Committed archive binding mismatch')
        records.append((row['id'], row['group'], row['split'], row['seconds'],
                        str(manifest), offset, len(line)))
        offset += len(line)
    failures = {r['id'] for r in event['failures']}
    if len(ids) != event['count'] or ids & failures or ids | failures != set(event['processed_ids']):
        raise ValueError('Committed outcomes mismatch')
    return records


def snapshot_cache(cache_root, output):
    """Freeze committed journal prefixes. Never scan attempts or unfinished tar files."""
    root, output = Path(cache_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Immutable index already exists: {output}')
    dataset = json.loads((root / 'dataset.json').read_text())
    if dataset['cache_protocol'] != 'h3_static_single_writer_v3':
        raise ValueError('Expected the static v3 committed-cache protocol')
    if (root / 'INTEGRITY_ALERT.json').exists():
        raise ValueError('Cache integrity alert must be resolved first')
    # Capture every size before reading manifests: later commits cannot enter this snapshot.
    durable = {}
    for p in sorted((root / 'progress').glob('*.json')):
        for task, offset in json.loads(p.read_text()).get('offsets', {}).items():
            durable[task] = max(durable.get(task, 0), offset)
    boundaries = [(p, durable[p.stem]) for p in sorted((root / 'journals').glob('*.jsonl'))
                  if durable.get(p.stem, 0)]
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + f'.tmp.{os.getpid()}.{uuid.uuid4().hex}')
    fd, local_name = tempfile.mkstemp(prefix='h3-upsampler-index-', suffix='.sqlite')
    os.close(fd)
    local = Path(local_name)
    db = sqlite3.connect(local)
    db.execute('PRAGMA cache_size=-65536')
    db.executescript('''
        CREATE TABLE pairs (ordinal INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
          source_group TEXT NOT NULL, split TEXT NOT NULL, seconds INTEGER NOT NULL,
          manifest TEXT NOT NULL, offset INTEGER NOT NULL, length INTEGER NOT NULL);
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    count, chunks = 0, 0
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        for journal, boundary in boundaries:
            with journal.open('rb') as handle:
                raw = handle.read(boundary)
            if len(raw) != boundary or not raw.endswith(b'\n'):
                raise ValueError('Durable journal boundary changed')
            lines = raw.splitlines()
            events = [json.loads(line) for line in lines]
            events = [e for e in events if e['kind'] in {'chunk', 'seed'}]
            for event, records in zip(events, pool.map(_manifest_records, events)):
                task = journal.stem + '.jsonl'
                if event['task_sha256'] != dataset['task_sha256'][task]:
                    raise ValueError('Journal input binding changed')
                db.executemany('INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?)',
                               ((count+i+1, *record) for i, record in enumerate(records)))
                count += len(records)
                chunks += 1
                if chunks % 1000 == 0:
                    db.commit()
                    print(json.dumps(dict(event='snapshot', chunks=chunks, pairs=count)), flush=True)
        if db.execute('SELECT 1 FROM pairs GROUP BY source_group HAVING COUNT(DISTINCT split)>1 LIMIT 1').fetchone():
            raise ValueError('Train/validation source groups overlap')
        if (root / 'INTEGRITY_ALERT.json').exists():
            raise ValueError('Cache integrity alert appeared during snapshot')
        counts = {f'{s}_{t}s': n for s, t, n in db.execute(
            'SELECT split,seconds,COUNT(*) FROM pairs GROUP BY split,seconds')}
        if not counts.get('train_5s') or not counts.get('val_5s'):
            raise ValueError('Snapshot needs nonempty training and validation splits')
        metadata = dict(schema='h3_upsampler_index_v1', cache_root=str(root), pairs=count,
                        counts=counts, manifests=chunks, created=time.time(),
                        dataset_sha256=sha_file(root / 'dataset.json'),
                        journal_boundaries={p.name: size for p, size in boundaries})
        for key, value in metadata.items():
            db.execute('INSERT INTO metadata VALUES (?,?)', (key, json.dumps(value)))
        db.execute('CREATE INDEX split_duration ON pairs(split,seconds)')
        db.commit()
        db.close()
        pool.shutdown(wait=True)
        shutil.copyfile(local, tmp)
        local.unlink()
        with tmp.open('rb') as handle:
            os.fsync(handle.fileno())
        # Exclusive publication; a competing snapshot builder cannot overwrite this index.
        os.link(tmp, output)
        tmp.unlink()
        atomic_json(output.with_suffix('.json'), dict(metadata, sha256=sha_file(output)))
        return metadata
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        db.close()
        tmp.unlink(missing_ok=True)
        local.unlink(missing_ok=True)
        raise


def open_index(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro&immutable=1', uri=True)


class IndexedPairs(Dataset):
    """Only compact row numbers live in RAM; metadata and tensors are loaded lazily."""
    def __init__(self, path, config, split, limit=0):
        self.path, self.config = str(Path(path).resolve()), config
        self.connection, self.pid = None, None
        with open_index(self.path) as db:
            if json.loads(db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()[0]) != 'h3_upsampler_index_v1':
                raise ValueError('Unsupported paired training index')
            self.ordinals = torch.tensor([r[0] for r in db.execute(
                'SELECT ordinal FROM pairs WHERE split=? ORDER BY ordinal', (split,))], dtype=torch.int64)
            if limit:
                # Balanced 5/10/15 heldout, deterministic within each bucket.
                selected = []
                for seconds in (5, 10, 15):
                    values = [r[0] for r in db.execute('SELECT ordinal FROM pairs WHERE split=? AND seconds=? ORDER BY id', (split, seconds))]
                    values.sort(key=lambda i: hashlib.sha256(f'{config.training.seed}:{i}'.encode()).digest())
                    selected.extend(values[:limit // 3 + (seconds == 5) * (limit % 3)])
                self.ordinals = torch.tensor(selected, dtype=torch.int64)
                if len(self.ordinals) != limit:
                    raise ValueError('Insufficient samples for balanced validation')
        if not len(self.ordinals):
            raise ValueError(f'Empty {split} dataset')

    def __len__(self):
        return len(self.ordinals)

    def duration_buckets(self):
        with open_index(self.path) as db:
            buckets = {}
            for seconds in (5, 10, 15):
                ordinals = torch.tensor([r[0] for r in db.execute(
                    'SELECT ordinal FROM pairs WHERE split=? AND seconds=? ORDER BY ordinal',
                    ('train', seconds))], dtype=torch.int64)
                positions = torch.searchsorted(self.ordinals, ordinals)
                if (positions >= len(self.ordinals)).any() or not torch.equal(self.ordinals[positions], ordinals):
                    raise ValueError('Duration buckets require the full ordered training split')
                buckets[seconds] = positions
        return buckets

    def __getitem__(self, key):
        crop = None
        if isinstance(key, tuple):
            index, seed = key[:2]
            if len(key) == 3:
                crop = key[2]
        else:
            index, seed = key, None
        if self.pid != os.getpid():
            if self.connection is not None:
                self.connection.close()
            self.connection, self.pid = open_index(self.path), os.getpid()
        row_id, group, split, seconds, manifest, offset, length = self.connection.execute(
            'SELECT id,source_group,split,seconds,manifest,offset,length FROM pairs WHERE ordinal=?',
            (int(self.ordinals[index]),)).fetchone()
        with open(manifest, 'rb') as handle:
            handle.seek(offset)
            row = json.loads(handle.read(length))
        if (row['id'], row['group'], row['split'], row['seconds']) != (row_id, group, split, seconds):
            raise ValueError('Frozen index metadata changed')
        low, high = load_pair(row, self.config)
        if seed is not None:
            rng = torch.Generator().manual_seed(seed)
            length = (low.shape[1] if crop == 0 else
                      self.config.crossres.crop_latent_frames if crop is None else crop)
            if low.shape[1] < length:
                raise ValueError('Training crop exceeds cached duration')
            start = int(torch.randint(low.shape[1] - length + 1, (), generator=rng))
            low, high = low[:, start:start+length], high[:, start:start+length]
            if bool(torch.randint(2, (), generator=rng)):
                low, high = low.flip(-1), high.flip(-1)
        return low.contiguous(), high.contiguous()


class ResumeBatches(Sampler):
    """Global shuffled stream, rank-disjoint; exact continuation from optimizer step."""
    def __init__(self, size, batch_size, start, stop, seed, rank=0, world=1):
        self.size, self.batch_size, self.start, self.stop = size, batch_size, start, stop
        self.seed, self.rank, self.world = seed, rank, world
        if size < batch_size * world:
            raise ValueError('Dataset smaller than a global batch')

    def __len__(self):
        return max(0, self.stop-self.start)

    def __iter__(self):
        epoch, order = -1, None
        for step in range(self.start, self.stop):
            batch = []
            for local in range(self.batch_size):
                position = (step*self.world+self.rank)*self.batch_size+local
                e, i = divmod(position, self.size)
                if e != epoch:
                    order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed+e))
                    epoch = e
                batch.append((int(order[i]), self.seed + position*100003))
            yield batch


class MixedDurationBatches(Sampler):
    """Same full-clip duration across ranks and accumulation microbatches.

    A shuffled weighted cycle fixes duration frequencies. Per-bucket sample
    streams are rank-disjoint and recoverable solely from optimizer step.
    """
    def __init__(self, buckets, batch_size, accumulation, start, stop, seed,
                 rank=0, world=1, weights=None):
        self.buckets = {int(k): torch.as_tensor(v, dtype=torch.int64) for k,v in buckets.items()}
        self.weights = {int(k):v for k,v in (weights or {5:6,10:3,15:1}).items()}
        if (set(self.buckets)!=set(self.weights) or not self.weights or
                any(type(v) is not int or v<=0 for v in self.weights.values())):
            raise ValueError('Mixed duration weights must be positive integer bucket counts')
        self.cycle = [k for k in sorted(self.weights) for _ in range(self.weights[k])]
        self.batch_size,self.accumulation,self.start,self.stop = batch_size,accumulation,start,stop
        self.seed,self.rank,self.world = seed,rank,world
        if min(batch_size,accumulation,world)<1 or not 0<=rank<world:
            raise ValueError('Invalid mixed duration batch geometry')
        if any(len(v)<batch_size*accumulation*world for v in self.buckets.values()):
            raise ValueError('Duration bucket smaller than an effective global batch')

    def __len__(self):
        return max(0,self.stop-self.start)*self.accumulation

    def __iter__(self):
        cache={}
        for step in range(self.start,self.stop):
            cycle,slot=divmod(step,len(self.cycle))
            permutation=torch.randperm(len(self.cycle),generator=torch.Generator().manual_seed(self.seed+cycle)).tolist()
            schedule=[self.cycle[i] for i in permutation]
            seconds=schedule[slot]
            occurrence=cycle*self.weights[seconds]+schedule[:slot].count(seconds)
            bucket=self.buckets[seconds]
            for micro in range(self.accumulation):
                result=[]
                for local in range(self.batch_size):
                    position=((occurrence*self.world+self.rank)*self.accumulation+micro)*self.batch_size+local
                    epoch,index=divmod(position,len(bucket))
                    if seconds not in cache or cache[seconds][0]!=epoch:
                        order=torch.randperm(len(bucket),generator=torch.Generator().manual_seed(self.seed+seconds*1009+epoch))
                        cache[seconds]=(epoch,order)
                    sample=int(bucket[cache[seconds][1][index]])
                    # crop=0 preserves the complete cache clip; seed still controls flipping.
                    result.append((sample,self.seed+seconds*9973+position*100003,0))
                yield result
