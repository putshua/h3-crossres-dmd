import json
from pathlib import Path
import sqlite3

import pytest
import torch
from torch.nn import functional as F

from h3dmd.stride_cache import PairArchive, sha_file, atomic_json
from h3dmd.stride_pairs import SAMPLING
from h3dmd.upsampler_data import snapshot_cache, IndexedPairs, ResumeBatches, open_index
from h3dmd.upsampler_training import load_settings, train
from h3dmd.upsampling import frames, video


def fixture_cache(root, leak=False, duplicate=False):
    root.mkdir()
    for d in ('journals','progress'):(root/d).mkdir()
    task_sha='a'*64
    atomic_json(root/'dataset.json',dict(cache_protocol='h3_static_single_writer_v3',task_sha256={'000000.jsonl':task_sha}))
    archive=PairArchive(root/'archive')
    torch.manual_seed(3)
    for i in range(9):
        split='train' if i<6 else 'val'
        low=torch.randn(24,2,20,32)
        high=video(F.interpolate(frames(low[None]),size=(30,48),mode='bicubic',align_corners=False),1,2)[0]+.2
        row=dict(id=str(i),group='0' if leak else str(i),split=split,seconds=(5,10,15)[i%3],num_frames=5)
        pairing=dict(sampling=SAMPLING,fps=24,same_source_frames=True,normalization='h3_standardized_video_v1')
        archive.add(row,low,high,pairing)
    e=archive.close([],task_sha)
    e.update(kind='chunk',processed_ids=[str(i) for i in range(9)])
    journal=root/'journals'/'000000.jsonl'
    raw=(json.dumps(e)+'\n').encode();journal.write_bytes(raw+(raw if duplicate else b'{"uncommitted":'))
    atomic_json(root/'progress'/'p0000-g0.json',dict(offsets={'000000':len(raw)*(2 if duplicate else 1)}))
    return root


def test_committed_snapshot_and_readonly_pair_roundtrip(tmp_path):
    cache=fixture_cache(tmp_path/'cache');index=tmp_path/'index.sqlite'
    result=snapshot_cache(cache,index)
    assert result['pairs']==9
    settings,c=load_settings('configs/h3_upsampler_train_320to480.json',['crop_latent_frames=2'])
    pairs=IndexedPairs(index,c,'train')
    a,b=pairs[(0,123)]
    assert a.shape==(24,2,20,32) and b.shape==(24,2,30,48)
    torch.testing.assert_close(a,pairs[(0,123)][0],rtol=0,atol=0)
    with pytest.raises(FileExistsError):snapshot_cache(cache,index)
    # Reader rejects metadata changes after the fixed snapshot.
    manifest=cache/'archive'/'manifest.jsonl'
    manifest.write_bytes(manifest.read_bytes().replace(b'"group": "0"',b'"group": "X"',1))
    with pytest.raises(ValueError,match='metadata changed'):pairs[0]


@pytest.mark.parametrize('leak,duplicate,error',[(True,False,ValueError),(False,True,sqlite3.IntegrityError)])
def test_bad_cache_cannot_publish_training_index(tmp_path,leak,duplicate,error):
    root=fixture_cache(tmp_path/'cache',leak,duplicate);index=tmp_path/'bad.sqlite'
    with pytest.raises(error):snapshot_cache(root,index)
    assert not index.exists()


def test_resume_sampler_is_exact_and_rank_disjoint():
    all_batches=list(ResumeBatches(97,3,0,10,42,0,2))
    assert all_batches[6:]==list(ResumeBatches(97,3,6,10,42,0,2))
    other=list(ResumeBatches(97,3,0,10,42,1,2))
    ids=[i for batch in all_batches+other for i,seed in batch]
    assert len(ids)==len(set(ids))


def test_training_checkpoint_resume_and_artifact(tmp_path):
    torch.set_num_threads(2)
    index=tmp_path/'index.sqlite';snapshot_cache(fixture_cache(tmp_path/'cache'),index)
    settings,c=load_settings('configs/h3_upsampler_train_320to480.json',[
        'mapper_width=8','max_steps=3','batch_size=2','num_workers=0',
        'validation_samples=3','validation_every=1','save_every=1',
        'crop_latent_frames=2','tensorboard=false'])
    output=tmp_path/'training'
    result=train(settings,c,index,output,torch.device('cpu'))
    saved=torch.load(output/'latest.pt',weights_only=True)
    assert saved['step']==3
    assert any(v.abs().sum()>0 for k,v in saved['model'].items() if k=='decoder.weight')
    train(settings,c,index,output,torch.device('cpu'))
    if result['accepted']:
        frozen=torch.load(output/'frozen.pt',weights_only=True)
        assert frozen['accepted'] and frozen['pretrain_steps_completed']==3
    changed=dict(settings,batch_size=1)
    with pytest.raises(ValueError,match='Resume'):train(changed,c,index,output,torch.device('cpu'))
    extended = dict(settings, max_steps=5)
    train(extended,c,index,output,torch.device('cpu'))
    resumed = torch.load(output/'latest.pt',weights_only=True)
    assert resumed['step'] == 5
    assert (output/'completion_at_000000003.json').exists()
    clean = tmp_path/'extended_clean'
    train(extended,c,index,clean,torch.device('cpu'))
    uninterrupted = torch.load(clean/'latest.pt',weights_only=True)
    for key in resumed['model']:
        torch.testing.assert_close(resumed['model'][key], uninterrupted['model'][key], rtol=0, atol=0)
    with pytest.raises(ValueError,match='Resume'):
        train(settings,c,index,output,torch.device('cpu'))


def test_interrupted_training_restarts_from_saved_optimizer_and_data_position(tmp_path,monkeypatch):
    from h3dmd import upsampler_training as trainer
    torch.set_num_threads(2)
    index=tmp_path/'index.sqlite';snapshot_cache(fixture_cache(tmp_path/'cache'),index)
    settings,c=load_settings('configs/h3_upsampler_train_320to480.json',[
        'mapper_width=8','max_steps=3','batch_size=2','num_workers=0',
        'validation_samples=3','validation_every=3','save_every=1',
        'crop_latent_frames=2','tensorboard=false'])
    output=tmp_path/'interrupted'
    save=trainer.atomic_save
    def interrupted(value,path):
        save(value,path)
        if Path(path).name=='latest.pt' and value['step']==1:
            raise RuntimeError('simulated process interruption')
    monkeypatch.setattr(trainer,'atomic_save',interrupted)
    with pytest.raises(RuntimeError,match='simulated'):train(settings,c,index,output,torch.device('cpu'))
    monkeypatch.setattr(trainer,'atomic_save',save)
    train(settings,c,index,output,torch.device('cpu'))
    clean=tmp_path/'uninterrupted';train(settings,c,index,clean,torch.device('cpu'))
    a=torch.load(output/'latest.pt',weights_only=True);b=torch.load(clean/'latest.pt',weights_only=True)
    assert a['step']==b['step']==3
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)


def test_visualization_exports_comparison_and_rejects_changed_input(tmp_path,monkeypatch):
    import numpy as np
    from h3dmd.upsampler_visualization import visualization_sources,UpsamplerVisualization
    from h3dmd.upsampling import new_mapper
    settings,c=load_settings('configs/h3_upsampler_train_320to480.json')
    source=tmp_path/'source.pt'
    torch.save(dict(video_latents=torch.ones(1,24,2,20,32),metadata=dict(width=512,height=320,num_frames=5,fps=24)),source)
    manifest=tmp_path/'inputs.json';manifest.write_text(json.dumps([dict(id='fixed',path=str(source))]))
    sources=visualization_sources(manifest)
    callback=UpsamplerVisualization(settings,tmp_path/'out',torch.device('cpu'),sources)
    def decode(latent,n):
        return np.full((n,latent.shape[-2]*16,latent.shape[-1]*16,3),127,dtype=np.uint8)
    monkeypatch.setattr(callback,'decode',decode)
    report=callback.run(new_mapper(c),0)
    assert len(report['samples'])==1
    destination=tmp_path/'out'/'visualization'/'iter_000000000'
    assert (destination/'fixed_comparison.mp4').stat().st_size>0
    assert (destination/'fixed_contact.jpg').exists() and (destination/'COMPLETE.json').exists()
    source.write_bytes(source.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='source changed'):callback.run(new_mapper(c),1)
    assert not (tmp_path/'out'/'visualization'/'iter_000000001'/'COMPLETE.json').exists()


def test_reporting_changes_resume_and_keep_immutable_checkpoints(tmp_path):
    from h3dmd.upsampling import file_sha256
    torch.set_num_threads(2)
    index=tmp_path/'index.sqlite';snapshot_cache(fixture_cache(tmp_path/'cache'),index)
    settings,c=load_settings('configs/h3_upsampler_train_320to480.json',[
        'mapper_width=8','max_steps=2','batch_size=2','num_workers=0',
        'validation_samples=3','validation_every=2','save_every=1',
        'crop_latent_frames=2','tensorboard=false'])
    output=tmp_path/'training';train(settings,c,index,output,torch.device('cpu'))
    extended=dict(settings,max_steps=4,save_every=2,keep_periodic_checkpoints=True,visualization_every=2000)
    train(extended,c,index,output,torch.device('cpu'))
    snapshot=output/'checkpoints/iter_000000004/state.pt'
    assert torch.load(snapshot,weights_only=True)['step']==4
    digest=file_sha256(snapshot)
    train(dict(extended,max_steps=6),c,index,output,torch.device('cpu'))
    assert file_sha256(snapshot)==digest
    assert torch.load(output/'latest.pt',weights_only=True)['step']==6


def test_preview_cadence_includes_initial_periodic_and_final(tmp_path,monkeypatch):
    import h3dmd.upsampler_visualization as preview
    calls=[]
    monkeypatch.setattr(preview,'visualization_sources',lambda _: [{'id':'fixed','sha256':'a'*64}])
    class FakePreview:
        def __init__(self,settings,output,device,sources):self.output=output
        def run(self,model,step):
            calls.append(step)
            destination=self.output/'visualization'/f'iter_{step:09d}'
            assert (destination/'mapper.pt').exists()
            atomic_json(destination/'COMPLETE.json',dict(step=step))
            return dict(seconds=0,samples=[])
    monkeypatch.setattr(preview,'UpsamplerVisualization',FakePreview)
    torch.set_num_threads(2)
    index=tmp_path/'index.sqlite';snapshot_cache(fixture_cache(tmp_path/'cache'),index)
    settings,c=load_settings('configs/h3_upsampler_train_320to480.json',[
        'mapper_width=8','max_steps=3','batch_size=2','num_workers=0',
        'validation_samples=3','validation_every=3','save_every=2',
        'crop_latent_frames=2','tensorboard=false'])
    settings.update(visualization_manifest='fixed',visualization_every=2)
    train(settings,c,index,tmp_path/'training',torch.device('cpu'))
    assert calls==[0,2,3]
