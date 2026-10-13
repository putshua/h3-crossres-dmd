from pathlib import Path
import pytest
from rcm.checkpointers.rolling import checkpoint_name, publish_and_prune


def test_retention_keeps_archives_and_only_latest_recovery(tmp_path):
    for iteration in (100, 200, 500, 600, 1000):
        name = checkpoint_name(iteration, 500, 100)
        (tmp_path / name).mkdir()
        publish_and_prune(tmp_path, name)
        assert (tmp_path / 'latest_checkpoint.txt').read_text().strip() == name
        assert len(list(tmp_path.glob('latest_iter_*'))) == (iteration % 500 != 0)
        for archive in (500, 1000):
            if archive <= iteration:
                assert (tmp_path / f'iter_{archive:09}').is_dir()


def test_failed_publication_preserves_previous_checkpoint(tmp_path, monkeypatch):
    import rcm.checkpointers.rolling as rolling
    old = checkpoint_name(100, 500, 100)
    new = checkpoint_name(200, 500, 100)
    (tmp_path / old).mkdir(); publish_and_prune(tmp_path, old)
    (tmp_path / new).mkdir()
    def fail(*args):
        raise OSError('injected publication failure')
    monkeypatch.setattr(rolling.os, 'replace', fail)
    with pytest.raises(OSError):
        publish_and_prune(tmp_path, new)
    assert (tmp_path / 'latest_checkpoint.txt').read_text().strip() == old
    assert (tmp_path / old).is_dir()


def test_real_dcp_rolling_saves_and_restores_master(tmp_path, monkeypatch):
    import torch
    import torch.distributed as dist
    from test_upstream_integration import configured_model
    from rcm.checkpointers.dcp_distill import DistributedCheckpointer_Distill
    monkeypatch.setenv('IMAGINAIRE_OUTPUT_PATH_LOCAL', str(tmp_path))
    c, model = configured_model(tmp_path)
    model.init_optimizer_scheduler(c.optimizer, c.scheduler)
    for opt in model.optimizer_dict.values():
        for group in opt.param_groups:
            for p in group['params']: p.grad = torch.ones_like(p)
        opt.step(); opt.zero_grad()
    dist.init_process_group('gloo', init_method=f'file://{tmp_path}/rendezvous', rank=0, world_size=1)
    try:
        cp = DistributedCheckpointer_Distill(c.checkpoint, c.job, None)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        masters = {role: [opt.state[p]['master_param'].clone() for g in opt.param_groups for p in g['params']]
                   for role, opt in model.optimizer_dict.items()}
        for iteration in (100, 200, 500, 600):
            cp.save(model, model.optimizer_dict, model.scheduler_dict, scaler, iteration=iteration)
        root = Path(cp.save_dirname)
        assert sorted(p.name for p in root.iterdir() if p.is_dir()) == ['iter_000000500','latest_iter_000000600']
        model.init_optimizer_scheduler(c.optimizer, c.scheduler)
        with torch.no_grad(): model.net.scale.add_(1)
        assert cp.load(model, model.optimizer_dict, model.scheduler_dict, scaler) == 600
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, saved[k], rtol=0, atol=0)
        for role, opt in model.optimizer_dict.items():
            for p, master in zip([p for g in opt.param_groups for p in g['params']], masters[role]):
                torch.testing.assert_close(opt.state[p]['master_param'], master, rtol=0, atol=0)
        cp.finalize()
    finally:
        dist.destroy_process_group()
