from collections import Counter

import pytest
import torch

from h3dmd.methods.staged_dmd import H3PCMWarmupDMDMethod
from test_dmd import make_method, make_batch


def staged():
    old, models = make_method()
    cfg = old.config
    cfg.method = "h3_pcm_warmup_dmd"
    cfg.pcm.enabled = cfg.pcm.warmup_only = True
    cfg.pcm.warmup_steps = 500
    cfg.dmd.data_mode = "rollout"
    cfg.dmd.rf_loss_weight = 0
    cfg.dmd.update_schedule = "alternating"
    cfg.dmd.student_update_frequency = 6
    cfg.dmd.rollout_steps = 4
    cfg.dmd.cycle_rollout_steps = True
    return H3PCMWarmupDMDMethod(cfg, models)


def test_phase_roles_objectives_and_rollout_cycle():
    m = staged()
    batch = make_batch()
    counts = Counter()
    for step in range(2500):
        closures = list(m.training_step_closures(batch, step))
        assert len(closures) == 1
        c = closures[0]
        counts[m.phase(step), c.role, c.name] += 1
    assert counts == {
        ("pcm", "student", "pcm"): 500,
        ("fake_warmup", "critic", "rf"): 100,
        ("dmd", "student", "dmd"): 317,
        ("dmd", "critic", "rf"): 1583,
    }
    assert [m._rollout_count(i) for i in range(600, 624, 6)] == [1, 2, 3, 4]
    assert [m._rollout_count(i, critic=True) for i in (601, 602, 603, 604)] == [1, 2, 3, 4]
    assert [m._rollout_count(i, critic=True) for i in range(500, 504)] == [1, 2, 3, 4]


def test_scheduler_phase_reset_and_resume():
    m = staged()
    assert m.lr_multiplier("net", 0) == .02
    assert m.lr_multiplier("net", 49) == 1
    assert m.lr_multiplier("net", 499) == pytest.approx(.1)
    assert m.lr_multiplier("net", 500) == .05
    assert m.lr_multiplier("net", 519) == 1
    assert m.lr_multiplier("fake_score", 0) == .01
    assert m.lr_multiplier("fake_score", 99) == 1
    p = torch.nn.Parameter(torch.tensor(1.))
    opt = torch.optim.AdamW([p], lr=2e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: m.lr_multiplier("net", n))
    for _ in range(507):
        opt.step()
        scheduler.step()
    saved_opt, saved_sched = opt.state_dict(), scheduler.state_dict()
    new_opt = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.))], lr=2e-6)
    new_sched = torch.optim.lr_scheduler.LambdaLR(new_opt, lambda n: m.lr_multiplier("net", n))
    new_opt.load_state_dict(saved_opt)
    new_sched.load_state_dict(saved_sched)
    for _ in range(20):
        opt.step(); scheduler.step()
        new_opt.step(); new_sched.step()
        assert opt.param_groups[0]["lr"] == new_opt.param_groups[0]["lr"]


def test_actual_backward_only_updates_the_active_role():
    m = staged()
    m.config.schedule.inference_q = [1., .75, .5, .25, 0.]
    m.config.schedule.num_inference_steps = 5
    batch = make_batch()
    for step, role in ((499, "student"), (500, "critic"), (600, "student"), (601, "critic")):
        m.models.student.zero_grad(set_to_none=True)
        m.models.critic.zero_grad(set_to_none=True)
        closure, = m.training_step_closures(batch, step)
        result = closure.compute()
        assert torch.isfinite(result.loss)
        result.loss.backward()
        active = getattr(m.models, role).scale.grad
        inactive = getattr(m.models, "critic" if role == "student" else "student").scale.grad
        assert active is not None and torch.isfinite(active)
        assert inactive is None
