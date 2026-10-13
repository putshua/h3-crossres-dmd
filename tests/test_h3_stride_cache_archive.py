import json
from pathlib import Path

import pytest
import torch

from h3dmd.stride_cache import PairArchive, read_member, sha_file, verify_commit


def test_real_tar_offsets_roundtrip_and_corruption(tmp_path):
    sink = PairArchive(tmp_path / 'attempt')
    low, high = torch.randn(24, 2, 2, 2), torch.randn(24, 2, 3, 3)
    for i in range(3):
        sink.add({'id': f'{i:064d}'}, low, high, {'sampling': 'test'})
    task = tmp_path / 'task.jsonl'
    task.write_text('immutable input\n')
    commit = sink.close([], sha_file(task))
    verify_commit(commit, task, deep=True)
    rows = [json.loads(line) for line in Path(commit['manifest']).read_text().splitlines()]
    assert [r['cache_member']['offset'] for r in rows] == sorted(r['cache_member']['offset'] for r in rows)
    assert torch.equal(read_member(commit['archive'], rows[-1]['cache_member'])['high'], high)
    with open(commit['archive'], 'r+b') as f:
        f.seek(rows[1]['cache_member']['offset'] + 10)
        f.write(b'corrupted')
    with pytest.raises(ValueError, match='SHA256'):
        read_member(commit['archive'], rows[1]['cache_member'])
    with pytest.raises(ValueError, match='corrupt'):
        verify_commit(commit, task)


def test_input_mutation_rejected_on_resume(tmp_path):
    sink = PairArchive(tmp_path / 'attempt')
    task = tmp_path / 'task.jsonl'
    task.write_text('first')
    commit = sink.close([], sha_file(task))
    task.write_text('changed')
    with pytest.raises(ValueError, match='task changed'):
        verify_commit(commit, task)


def test_stride_archive_mapper_reader_and_lazy_loading(tmp_path):
    from h3dmd.config import H3Config
    from h3dmd.mapper_pretraining import load_pair, LazyPairs
    from h3dmd.stride_pairs import SAMPLING
    config = H3Config(method='h3_crossres_dmd')
    config.data.width, config.data.height = 512, 320
    config.crossres.score_width, config.crossres.score_height = 768, 480
    pairing = dict(sampling=SAMPLING, fps=24, same_source_frames=True,
                   normalization='h3_standardized_video_v1')
    low, high = torch.randn(24, 2, 20, 32), torch.randn(24, 2, 30, 48)
    sink = PairArchive(tmp_path / 'attempt')
    sink.add({'id': 'a'*64, 'num_frames': 5}, low, high, pairing)
    task = tmp_path / 'task.jsonl'; task.write_text('input')
    commit = sink.close([], sha_file(task))
    row = json.loads(Path(commit['manifest']).read_text())
    actual = load_pair(row, config)
    assert torch.equal(actual[0], low) and torch.equal(actual[1], high)
    lazy = LazyPairs([row], config)
    assert len(lazy) == 1 and torch.equal(lazy[0][1], high)
    # Manifest identity changes must not silently relabel an archived tensor.
    row['id'] = 'b'*64
    with pytest.raises(ValueError, match='identity'):
        load_pair(row, config)
