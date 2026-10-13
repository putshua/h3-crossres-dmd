"""Standalone distributed spatial-mapper training, sharing the DMD artifact format."""
import json
from contextlib import nullcontext
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset

from h3dmd.config import H3Config
from h3dmd.mapper_pretraining import atomic_save, evaluate, accepted
from h3dmd.upsampling import new_mapper, mapper_spec, file_sha256, reconstruction_losses
from h3dmd.upsampler_data import IndexedPairs, ResumeBatches, MixedDurationBatches
from h3dmd.stride_cache import atomic_json


def load_settings(path, overrides=()):
    settings = dict(path) if isinstance(path, dict) else json.loads(Path(path).read_text())
    for option in overrides:
        key, value = option.split('=', 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        if key not in settings:
            raise ValueError(f'Unknown upsampler setting: {key}')
        settings[key] = value
    config = H3Config(method='h3_crossres_dmd')
    config.pcm.enabled = False
    config.dmd.rf_loss_weight = 0
    config.dmd.data_mode = 'rollout'
    config.data.height, config.data.width = 320, 512
    config.crossres.score_height, config.crossres.score_width = 480, 768
    for key in ('mapper_kind', 'mapper_width', 'mapper_groups', 'crop_latent_frames',
                'learning_rate', 'mse_ratio_limit', 'temporal_ratio_limit'):
        setattr(config.crossres, key, settings[key])
    for key in ('mapper_blocks_per_stage', 'mapper_gradient_checkpointing'):
        if key in settings:
            setattr(config.crossres, key, settings[key])
    if settings.get('precision', 'fp32') not in {'fp32', 'bf16'}:
        raise ValueError('Upsampler precision must be fp32 or bf16')
    accumulation=settings.get('gradient_accumulation_steps',1)
    if type(accumulation) is not int or accumulation<1:
        raise ValueError('gradient_accumulation_steps must be a positive integer')
    config.training.seed = settings['seed']
    # Run the standard H3 geometry/architecture validation too.
    config.validate()
    for key in ('max_steps','batch_size','save_every','validation_every','validation_samples','log_every','visualization_every'):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if settings['num_workers'] < 0 or not math.isfinite(settings['grad_clip']) or settings['grad_clip'] <= 0:
        raise ValueError('Invalid loader or gradient clipping setting')
    return settings, config


def train(settings, config, index, output, device, rank=0, world=1):
    torch.manual_seed(config.training.seed)
    output, index = Path(output).resolve(), Path(index).resolve()
    output.mkdir(parents=True, exist_ok=True)
    from h3dmd.upsampler_visualization import visualization_sources, UpsamplerVisualization
    sources = visualization_sources(settings.get('visualization_manifest', ''))
    initial=settings.get('initial_checkpoint','')
    signature = dict(visualization_sources=sources, schema='h3_upsampler_training_v1', settings=settings,
                     spec=mapper_spec(config), index_sha256=file_sha256(index), world_size=world)
    if initial:
        signature['initial_checkpoint_sha256']=file_sha256(initial)
    training = IndexedPairs(index, config, 'train')
    validation = IndexedPairs(index, config, 'val', settings['validation_samples'])
    if len(validation) < world:
        raise ValueError('Each distributed rank needs a heldout sample')
    heldout = Subset(validation, list(range(rank, len(validation), world)))
    module = new_mapper(config).to(device)
    if device.type == 'cuda':
        from rcm.utils.torch_fused_adamw_dtensor import FusedAdamW
        optimizer = FusedAdamW(module.parameters(), lr=config.crossres.learning_rate,
                              betas=(.9,.99), weight_decay=0, master_weights=True)
    else:
        optimizer = torch.optim.AdamW(module.parameters(), lr=config.crossres.learning_rate,
                                      betas=(.9,.99), weight_decay=0)
    start, best, best_step, best_model, best_report = 0, float('inf'), 0, None, None
    latest = output / 'latest.pt'
    if initial and not latest.exists():
        initial_state=torch.load(initial,map_location='cpu',weights_only=True)
        module.load_state_dict(initial_state['model'],strict=True)
        del initial_state  # New stage starts a fresh optimizer and validation baseline.
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True)
        previous_signature = saved['signature']
        previous_settings = previous_signature['settings']
        # Saving, reporting and preview cadence may change without changing
        # the optimizer, objective or deterministic data stream.
        reporting = {'max_steps','save_every','validation_every','visualization_every',
                     'visualization_manifest','visualization_vae_path','keep_periodic_checkpoints',
                     'visualization_oss_prefix','visualization_oss_config'}
        extended_settings = dict(previous_settings)
        for key in reporting:
            if key in settings:
                extended_settings[key] = settings[key]
            else:
                extended_settings.pop(key, None)
        extended_signature = dict(previous_signature,settings=extended_settings,
                                  visualization_sources=signature['visualization_sources'])
        if (extended_signature != signature or
                settings['max_steps'] < previous_settings['max_steps'] or
                settings['max_steps'] < saved['step']):
            raise ValueError('Resume index, settings or world size changed')
        module.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start, best, best_step = saved['step'], saved['best_score'], saved['best_step']
        best_model, best_report = saved['best_model'], saved['best_report']
        if start == settings['max_steps'] and (output/'COMPLETE.json').exists():
            completed = json.loads((output/'COMPLETE.json').read_text())
            if completed['accepted'] and file_sha256(output/'frozen.pt') != completed['artifact_sha256']:
                raise ValueError('Completed frozen artifact changed')
            return dict(step=start, accepted=completed['accepted'])
        if rank == 0 and start < settings['max_steps'] and (output/'COMPLETE.json').exists():
            # A larger iteration budget resumes the same optimizer/data stream.
            # Retain the previous completion record, but stop advertising completion.
            (output/'COMPLETE.json').replace(output/f'completion_at_{start:09d}.json')
    if rank == 0:
        atomic_json(output/'resolved_config.json', dict(signature, index=str(index),
                    training_pairs=len(training), validation_pairs=len(validation),
                    global_batch=settings['batch_size']*world*settings.get('gradient_accumulation_steps',1),
                    parameter_count=sum(p.numel() for p in module.parameters()),
                    precision=settings.get('precision', 'fp32'), optimizer='fused_adamw_master_fp32' if device.type=='cuda' else 'adamw'))
    wrapped = DDP(module, device_ids=[device.index] if device.type=='cuda' else None,
                  broadcast_buffers=False) if world>1 else module
    accumulation=settings.get('gradient_accumulation_steps',1)
    if settings.get('temporal_bucket_weights'):
        batches=MixedDurationBatches(training.duration_buckets(),settings['batch_size'],accumulation,
            start,settings['max_steps'],config.training.seed,rank,world,settings['temporal_bucket_weights'])
    else:
        batches=ResumeBatches(len(training),settings['batch_size'],start*accumulation,
            settings['max_steps']*accumulation,config.training.seed,rank,world)
    loader = DataLoader(training, batch_sampler=batches, num_workers=settings['num_workers'],
                        pin_memory=device.type=='cuda', **(dict(prefetch_factor=2) if settings['num_workers'] else {}))
    visualizer = UpsamplerVisualization(settings, output, device, sources) if rank==0 and sources else None
    writer = None
    if rank==0 and settings['tensorboard']:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output/'tensorboard'), purge_step=start+1 if start else None)

    def log(record):
        if rank==0:
            with (output/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(record)+'\n')
            if writer:
                for k,v in record.items():
                    if k!='step' and isinstance(v,(int,float)):
                        writer.add_scalar(record['event']+'/'+k,v,record['step'])
                writer.flush()
            print(json.dumps(record),flush=True)

    def checkpoint(step):
        if rank==0:
            atomic_save(dict(signature=signature,model=module.state_dict(),optimizer=optimizer.state_dict(),
                             step=step,best_score=best,best_step=best_step,best_model=best_model,
                             best_report=best_report),latest)
            if (settings.get('keep_periodic_checkpoints') and step>0 and
                    (step%settings['save_every']==0 or step==settings['max_steps'])):
                destination=output/'checkpoints'/f'iter_{step:09d}'
                destination.mkdir(parents=True,exist_ok=True)
                # latest is atomically replaced on the next save. This hardlink
                # retains the exact immutable checkpoint without a second write.
                snapshot=destination/'state.pt'
                if not snapshot.exists():
                    os.link(latest,snapshot)
                atomic_json(destination/'COMPLETE.json',dict(step=step,path=str(snapshot)))
        if dist.is_initialized():dist.barrier()

    def validate(step):
        nonlocal best,best_step,best_model,best_report
        report=evaluate(module,heldout,device)
        log(dict(event='validation',step=step,**report))
        if accepted(report,config) and report['mse_ratio']<best:
            best,best_step,best_report=report['mse_ratio'],step,report
            best_model={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
        return report

    def visualize(step):
        if not sources:
            return
        error, report = None, None
        if rank==0:
            try:
                snapshot=output/'visualization'/f'iter_{step:09d}'/'mapper.pt'
                if not snapshot.exists():
                    atomic_save(dict(step=step,spec=mapper_spec(config),
                                     model={k:v.detach().cpu() for k,v in module.state_dict().items()}),snapshot)
                report=visualizer.run(module,step)
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
        status=[error]
        if dist.is_initialized():dist.broadcast_object_list(status,src=0)
        if status[0]:raise RuntimeError('Upsampler visualization failed: '+status[0])
        if rank==0:log(dict(event='visualization',step=step,seconds=report['seconds'],samples=len(report['samples'])))

    step=start
    try:
        if not start:
            validate(0)
            checkpoint(0)
            visualize(0)
        elif sources and not (output/'visualization'/f'iter_{start:09d}'/'COMPLETE.json').is_file():
            visualize(start)
        last_finished=time.monotonic()
        iterator=iter(loader)
        for step in range(start+1,settings['max_steps']+1):
            begun=time.monotonic();data_seconds=0.;wrapped.train();optimizer.zero_grad(set_to_none=True)
            metric_sum=None;temporal_frames=None
            for micro in range(accumulation):
                reading=time.monotonic();low,high=next(iterator);data_seconds+=time.monotonic()-reading
                if temporal_frames is not None and temporal_frames!=low.shape[2]:
                    raise ValueError('Accumulation microbatches must share temporal geometry')
                temporal_frames=low.shape[2]
                sync=(wrapped.no_sync() if world>1 and micro<accumulation-1 else nullcontext())
                with sync:
                    with torch.autocast(device.type,dtype=torch.bfloat16,
                                        enabled=settings.get('precision','fp32')=='bf16'):
                        prediction=wrapped(low.to(device,non_blocking=True))
                        loss,parts=reconstruction_losses(prediction,high.to(device,non_blocking=True))
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite upsampler loss')
                    (loss/accumulation).backward()
                metrics=torch.stack([loss.detach(),*[v.detach() for v in parts.values()]])/accumulation
                metric_sum=metrics if metric_sum is None else metric_sum+metrics
                del prediction,loss,low,high
            norm=torch.nn.utils.clip_grad_norm_(module.parameters(),settings['grad_clip'],error_if_nonfinite=True)
            optimizer.step()
            if step%settings['log_every']==0:
                values=torch.cat([metric_sum[:1],norm.detach()[None],metric_sum[1:]])
                if dist.is_initialized():dist.all_reduce(values);values/=world
                values=values.cpu().tolist()
                total_seconds=time.monotonic()-begun
                compute_seconds=max(0.,total_seconds-data_seconds)
                timing=torch.tensor([compute_seconds,data_seconds,total_seconds],device=device)
                if dist.is_initialized():dist.all_reduce(timing,op=dist.ReduceOp.MAX)
                compute_seconds,data_seconds,step_seconds=timing.cpu().tolist()
                log(dict(event='train',step=step,loss=values[0],grad_norm=values[1],
                         **dict(zip(parts,values[2:])),lr=config.crossres.learning_rate,
                         temporal_frames=temporal_frames,micro_batch_size=settings['batch_size'],
                         gradient_accumulation_steps=accumulation,global_batch=settings['batch_size']*world*accumulation,
                         compute_seconds=compute_seconds,data_seconds=data_seconds,step_seconds=step_seconds,
                         samples_per_second=settings['batch_size']*world*accumulation/max(step_seconds,1e-9)))
            validation_due=step%settings['validation_every']==0 or step==settings['max_steps']
            if validation_due:validate(step)
            if step%settings['save_every']==0 or validation_due:checkpoint(step)
            if sources and (step%settings['visualization_every']==0 or step==settings['max_steps']):visualize(step)
            last_finished=time.monotonic()
        if rank==0:
            if best_model is not None:
                atomic_save(dict(spec=mapper_spec(config),model=best_model,signature=signature,accepted=True,
                                 report=best_report,best_step=best_step,pretrain_steps_completed=step),output/'frozen.pt')
            atomic_json(output/'COMPLETE.json',dict(step=step,accepted=best_model is not None,
                        best_step=best_step,report=best_report,
                        artifact_sha256=file_sha256(output/'frozen.pt') if best_model is not None else None))
        if dist.is_initialized():dist.barrier()
        return dict(step=step,accepted=best_model is not None)
    finally:
        if writer:writer.close()
