import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from h3dmd.config import H3Config
from h3dmd.data import CachedH3Dataset
from h3dmd.staged_data import StagedH3Sampler
from h3dmd.methods.staged_crossres_dmd import H3PCMWarmupCrossResolutionDMDMethod
from test_dmd import make_method, make_batch
from test_h3_crossres_dmd import settings
from h3dmd.upsampling import new_mapper


def staged():
    c = settings()
    c.method = 'h3_pcm_warmup_crossres_dmd'
    c.pcm.enabled = c.pcm.warmup_only = True
    c.pcm.data_mode = 'data'
    c.pcm.warmup_steps = 500
    c.dmd.update_schedule = 'alternating'
    c.dmd.student_update_frequency = 6
    c.dmd.cycle_rollout_steps = True
    c.dmd.rollout_steps = 4
    _, models = make_method()
    return H3PCMWarmupCrossResolutionDMDMethod(c, models, score_mapper=new_mapper(c))


@pytest.mark.parametrize('step,role,objective', [
    (499,'student','pcm'),(500,'critic','rf'),(599,'critic','rf'),
    (600,'student','dmd'),(601,'critic','rf'),(618,'student','dmd')])
def test_phase_domains_and_active_gradients(step, role, objective):
    m=staged()
    calls=[]
    original=m._call
    def record(role, latents, *args):
        calls.append((role,tuple(latents.video.shape[-2:])))
        return original(role,latents,*args)
    with patch.object(m,'_call',side_effect=record):
        closure,=m.training_step_closures(make_batch(),step)
        assert (closure.role,closure.name)==(role,objective)
        result=closure.compute()
        result.loss.backward()
    assert torch.isfinite(result.loss)
    assert getattr(m.models,role).scale.grad is not None
    inactive='critic' if role=='student' else 'student'
    assert getattr(m.models,inactive).scale.grad is None
    if objective=='pcm':
        assert all(size==(4,4) for _,size in calls)
    else:
        assert all(size==(4,4) for who,size in calls if who=='student')
        assert all(size==(6,6) for who,size in calls if who!='student')
    assert all(p.grad is None for p in m.score_mapper.parameters())


def test_staged_sampler_switches_sources_and_resume_has_identical_tail():
    from types import SimpleNamespace as NS
    pcm=NS(entries=[{'num_frames':f} for f in (124,243,362) for _ in range(64)])
    dmd=NS(entries=[{'num_frames':f} for f in (124,243,362) for _ in range(96)])
    # Production dataset exposes lengths as well as entries.
    class D:
        def __init__(self,entries):self.entries=entries
        def __len__(self):return len(self.entries)
    dataset=NS(pcm=D(pcm.entries),dmd=D(dmd.entries))
    config=staged().config
    config.training.max_steps=650
    sampler=StagedH3Sampler(dataset,config,rank=1,world_size=32)
    full=list(sampler)
    assert len(full)==650
    assert all(i<len(dataset.pcm) for i in full[:500])
    assert all(i>=len(dataset.pcm) for i in full[500:])
    entries=dataset.pcm.entries+dataset.dmd.entries
    assert [entries[i]['num_frames'] for i in full[498:503]]==[124,243,124,243,362]
    assert [entries[i]['num_frames'] for i in full[600:618]]==[124]*6+[243]*6+[362]*6
    for resume in (0,499,500,599,600,607,649,650):
        sampler.set_epoch(resume)
        assert list(sampler)==full[resume:]
        assert len(sampler)==len(full[resume:])


def test_packed_text_cache_reads_only_selected_prompt_and_retains_av_shape(tmp_path):
    path=tmp_path/'pack.safetensors'
    save_file({'p0000.prompt_embeds':torch.ones(3,5120,dtype=torch.bfloat16),
               'p0000.text_token_tags':torch.ones(3,dtype=torch.long),
               'p0001.prompt_embeds':torch.full((4,5120),2.,dtype=torch.bfloat16),
               'p0001.text_token_tags':torch.ones(4,dtype=torch.long)},str(path))
    manifest=tmp_path/'manifest.jsonl'
    manifest.write_text(json.dumps(dict(id='second',cache_path=str(path),text_cache_key='p0001',
        num_frames=243,width=512,height=320,shape_only=True))+'\n')
    row=CachedH3Dataset(manifest,height=320,width=512)[0]
    assert row['prompt_embeds'].shape==(4,5120)
    assert row['prompt_embeds'].eq(2).all()
    assert row['video_latents'].shape==(24,72,20,32)
    assert row['audio_latents'].shape==(2,32,405)


def test_combined_config_keeps_old_pure_dmd_guard():
    c=staged().config
    c.validate()
    c.method='h3_crossres_dmd'
    with pytest.raises(ValueError,match='pure DMD'):
        c.validate()


@pytest.mark.parametrize('resume',[0,1])
def test_inherited_trainer_exits_when_sampler_ends_exactly_at_max_iter(monkeypatch,resume):
    from contextlib import nullcontext
    from types import SimpleNamespace as NS
    from unittest.mock import MagicMock
    import rcm.trainers.trainer_distillation as module
    from imaginaire.utils import misc
    trainer=module.ImaginaireTrainer_Distill.__new__(module.ImaginaireTrainer_Distill)
    trainer.config=NS(trainer=NS(memory_format=torch.preserve_format,
        grad_scaler_args={'enabled':False},distributed_parallelism='fsdp',
        run_validation=False,max_iter=1,timeout_period=100),optimizer=None,scheduler=None,
        checkpoint=NS(save_iter=500,save_latest_iter=100,save_on_exit=False))
    trainer.callbacks=MagicMock()
    trainer.checkpointer=MagicMock()
    trainer.checkpointer.load.return_value=resume
    trainer.training_timer=misc.TrainingTimer()
    trainer.training_step=MagicMock(return_value=({},torch.tensor(0.),0))
    model=MagicMock();model.to.return_value=model;model.training=True
    monkeypatch.setattr(module,'maybe_enable_profiling',lambda *a,**k:nullcontext())
    monkeypatch.setattr(module,'maybe_enable_memory_snapshot',lambda *a,**k:nullcontext())
    monkeypatch.setattr(module.distributed,'barrier',lambda:None)
    monkeypatch.setattr(module.misc,'to',lambda value,**k:value)
    monkeypatch.setattr(module.signal,'alarm',lambda n:None)
    class ExactLoader:
        sampler=NS(set_epoch=lambda n:None)
        epochs=0
        def __iter__(self):
            self.epochs+=1
            if self.epochs>1:
                raise AssertionError('Trainer restarted an already completed empty epoch')
            return iter([{}] if resume==0 else [])
    loader=ExactLoader()
    trainer.train(model,loader,None)
    trainer.callbacks.on_train_end.assert_called_once_with(model,iteration=1)
    assert trainer.training_step.call_count==1-resume


def test_pcm_accumulation_keeps_optimizer_schedule_and_resume():
    from types import SimpleNamespace as NS
    class D:
        def __init__(self):
            self.entries = [{'num_frames':f} for f in (124,243,362) for _ in range(512)]
        def __len__(self): return len(self.entries)
    dataset=NS(pcm=D(),dmd=D())
    config=staged().config
    config.pcm.warmup_steps=4
    config.staged.fake_warmup_steps=3
    config.staged.pcm_gradient_accumulation_steps=2
    config.training.max_steps=13
    sampler=StagedH3Sampler(dataset,config,rank=1,world_size=32)
    full=list(sampler)
    assert len(full)==len(sampler)==17
    entries=dataset.pcm.entries+dataset.dmd.entries
    assert [entries[i]['num_frames'] for i in full[:8]]==[124,124,243,243,362,362,124,124]
    assert full[0]!=full[1]  # Two different examples per update, not duplicated data.
    assert all(i < len(dataset.pcm) for i in full[:8])
    assert all(i >= len(dataset.pcm) for i in full[8:])
    for resume in range(14):
        sampler.set_epoch(resume)
        offset=resume*2 if resume<4 else resume+4
        assert list(sampler)==full[offset:]
        assert len(sampler)==len(full[offset:])


def test_trainer_pcm_two_microbatches_make_one_mean_gradient_update():
    from types import SimpleNamespace as NS
    from unittest.mock import MagicMock
    from imaginaire.utils.misc import TrainingTimer
    from rcm.trainers.trainer_distillation import ImaginaireTrainer_Distill
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.tensor(1.))
            self.optimizer=torch.optim.SGD(self.parameters(),lr=0.1)
            self.scheduler=MagicMock()
        def gradient_accumulation_steps(self,iteration): return 2 if iteration==0 else 1
        def training_step_closures(self,batch,iteration):
            yield 'pcm',lambda: ({},self.weight*batch['value']),True
        def on_after_backward(self): pass
        def on_before_zero_grad(self,*args,**kwargs): pass
        def get_optimizers(self,iteration): return [self.optimizer]
        def get_lr_schedulers(self,iteration): return [self.scheduler]
    model=Model()
    trainer=ImaginaireTrainer_Distill.__new__(ImaginaireTrainer_Distill)
    trainer.config=NS(trainer=NS(distributed_parallelism='fsdp',grad_accum_iter=1))
    trainer.callbacks=MagicMock();trainer.training_timer=TrainingTimer()
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    _,_,accum=trainer.training_step(model,scaler,{'value':torch.tensor(2.)},iteration=0)
    assert accum==1 and model.weight.item()==1.
    _,_,accum=trainer.training_step(model,scaler,{'value':torch.tensor(4.)},iteration=0,grad_accum_iter=accum)
    assert accum==0
    assert model.weight.item()==pytest.approx(0.7)
    assert model.scheduler.step.call_count==1
    _,_,accum=trainer.training_step(model,scaler,{'value':torch.tensor(4.)},iteration=1)
    assert accum==0 and model.weight.item()==pytest.approx(0.3)
    assert model.scheduler.step.call_count==2
