from collections import Counter
import json
import torch
from h3dmd.upsampler_data import MixedDurationBatches,IndexedPairs,snapshot_cache
from h3dmd.upsampler_training import load_settings,train
from test_h3_upsampler_training import fixture_cache


def test_mixed_batches_align_ranks_accumulation_and_exact_resume():
    buckets={5:torch.arange(400),10:torch.arange(400,800),15:torch.arange(800,1200)}
    schedules=[list(MixedDurationBatches(buckets,2,2,0,10,42,rank,2)) for rank in range(2)]
    durations=[];seen=[]
    for step in range(10):
        keys=[key for rank in range(2) for batch in schedules[rank][step*2:step*2+2] for key in batch]
        seconds=[5 if key[0]<400 else 10 if key[0]<800 else 15 for key in keys]
        assert len(set(seconds))==1 and len(keys)==8
        assert all(key[2]==0 for key in keys)
        seen.extend(key[0] for key in keys);durations.append(seconds[0])
    assert Counter(durations)=={5:6,10:3,15:1}
    assert len(seen)==len(set(seen))
    assert schedules[0][14:]==list(MixedDurationBatches(buckets,2,2,7,10,42,0,2))


def test_full_clip_mixed_training_and_accumulation(tmp_path):
    torch.set_num_threads(2)
    index=tmp_path/'index.sqlite';snapshot_cache(fixture_cache(tmp_path/'cache'),index)
    settings,cfg=load_settings('configs/h3_upsampler_train_320to480.json',[
        'mapper_width=8','max_steps=10','batch_size=1','num_workers=0',
        'validation_samples=3','validation_every=10','save_every=10',
        'crop_latent_frames=37','tensorboard=false'])
    settings.update(gradient_accumulation_steps=2,temporal_bucket_weights={'5':6,'10':3,'15':1})
    data=IndexedPairs(index,cfg,'train')
    assert {k:len(v) for k,v in data.duration_buckets().items()}=={5:2,10:2,15:2}
    # Fixture clips are shorter than configured crop37; full-clip keys must bypass cropping.
    assert data[(0,42,0)][0].shape[1]==2
    output=tmp_path/'mixed';train(settings,cfg,index,output,torch.device('cpu'))
    records=[json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()]
    steps=[r for r in records if r['event']=='train']
    assert len(steps)==10
    assert all(r['global_batch']==2 and r['gradient_accumulation_steps']==2 for r in steps)
    saved=torch.load(output/'latest.pt',weights_only=True)
    assert saved['step']==10
    train(settings,cfg,index,output,torch.device('cpu'))
