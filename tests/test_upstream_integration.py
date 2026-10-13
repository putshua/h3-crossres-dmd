"""CPU contracts for the actual imported trainer/config/checkpointer stack."""
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from imaginaire.lazy_config import instantiate
from imaginaire.utils import callback, misc
from imaginaire.utils.config_helper import override
from rcm.configs.registry_h3 import make_config
from rcm.configs.h3 import native_config
from rcm.models.h3_distill import H3DistillModel
from rcm.trainers.trainer_distillation import ImaginaireTrainer_Distill
from test_dmd import TinyH3Role, make_batch


def cpu_network(config, role, context):
    net = TinyH3Role({"student": 0.1, "teacher": 0.5, "critic": -0.25, "ema": 0.1}[role])
    return net.freeze() if role in ("teacher", "ema") else net


def configured_model(tmp_path, pcm=False, ema=False):
    c = override(make_config(), ["--",
        "model.config.net._target_=test_upstream_integration.cpu_network",
        "model.config.h3.logging.tensorboard=false",
        "trainer.callbacks.fixed_prompt.enabled=false",
        f"model.config.h3.pcm.enabled={str(pcm).lower()}",
        f"model.config.ema.enabled={str(ema).lower()}"])
    return c, instantiate(c.model)


def test_hydra_registers_upstream_components_and_numeric_overrides():
    c = override(make_config(), ["--", "optimizer.lr=4e-6", "trainer.max_iter=17"])
    assert "ImaginaireTrainer_Distill" in str(c.trainer.type)
    assert "DistributedCheckpointer_Distill" in str(c.checkpoint.type)
    h3 = native_config(c.model.config.h3)
    assert c.optimizer.lr == h3.training.learning_rate == 4e-6
    assert c.model.config.optimizer_fake_score.lr == 1e-7
    assert list(c.optimizer.betas) == [0.0, 0.999]
    assert c.optimizer.optim_type == "fusedadam"
    assert c.optimizer.backend == "torch" and c.optimizer.master_weights
    assert c.model.config.optimizer_fake_score.master_weights
    assert h3.training.max_steps == 17
    assert h3.dmd.rf_loss_weight == 0 and h3.dmd.rollout_steps == 4
    assert not h3.pcm.enabled
    assert h3.data.frame_buckets == [124, 243, 362]
    assert h3.distributed.fsdp_size == 32
    assert h3.schedule.video_shift == 5
    assert h3.schedule.train_video_shift == 5
    assert h3.schedule.audio_shift == 3
    assert native_config(override(make_config(), ["--", "experiment=h3_synthetic_audio_pcm_dmd"]).model.config.h3).pcm.enabled


def test_launch_command_resolves_one_hydra_config():
    import shlex
    from test_train_launcher import launch
    result = launch("train", LR="4e-6", CRITIC_LR="8e-7", MAX_ITER="23", PCM_ENABLED="1")
    assert result.returncode == 0
    tokens = shlex.split(result.stdout.splitlines()[-1])
    c = override(make_config(), tokens[tokens.index("--"):])
    native = native_config(c.model.config.h3)
    assert c.optimizer.lr == native.training.learning_rate == 4e-6
    assert native.training.critic_learning_rate == 8e-7
    assert c.trainer.max_iter == native.training.max_steps == 23
    assert native.pcm.enabled and native.pcm.data_mode == "rollout"
    assert native.model.lora_rank == native.model.critic_lora_rank == 0


def test_saved_upstream_yaml_can_configure_h3_inference(tmp_path):
    from imaginaire.lazy_config import LazyConfig
    from h3dmd.config import load_config
    c = override(make_config(), ["--"])
    path = tmp_path / "config.yaml"
    LazyConfig.save_yaml(c, str(path))
    restored = load_config(path)
    assert restored.pcm.enabled is False
    assert restored.training.max_steps == 3000
    assert restored.training.learning_rate == 1e-6
    assert restored.dmd.rollout_steps == 4


@pytest.mark.parametrize("pcm", [False, True])
def test_upstream_trainer_closures_optimizer_schedule_and_ema(tmp_path, monkeypatch, pcm):
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_PATH_LOCAL", str(tmp_path))
    c, model = configured_model(tmp_path, pcm=pcm, ema=True)
    model.init_optimizer_scheduler(c.optimizer, c.scheduler)
    trainer = ImaginaireTrainer_Distill.__new__(ImaginaireTrainer_Distill)
    trainer.config, trainer.training_timer = c, misc.TrainingTimer()
    trainer.callbacks = callback.CallBackGroup(c, trainer)
    trainer.callbacks.on_train_start(model, iteration=0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    hooks, outputs = [], []
    model.method.on_after_optimizer_step = lambda iteration, roles: hooks.append((iteration, roles))
    for iteration in range(6):
        previous = model.net_ema.scale.clone()
        metrics, loss, accum = trainer.training_step(model, scaler, asdict(make_batch()), iteration, 0)
        outputs.append(metrics)
        assert accum == 0 and torch.isfinite(loss)
        if iteration in (1, 2, 3, 4):
            torch.testing.assert_close(model.net_ema.scale, previous)
    assert model.optimizer_dict["net"].state[model.net.scale]["step"] == 2
    assert model.optimizer_dict["fake_score"].state[model.net_fake_score.scale]["step"] == 4
    assert [r for _, r in hooks] == [("student",), *( [("critic",)] * 4 ), ("student",)]
    assert outputs[0]["student/dmd/rollout_steps"].item() == 1
    assert outputs[5]["student/dmd/rollout_steps"].item() == 2
    assert any(k.startswith("student/pcm/") for k in outputs[0]) == pcm
    trainer.callbacks.on_train_end(model, iteration=6)


def test_upstream_dcp_restores_fresh_optimizers_and_excludes_teacher(tmp_path, monkeypatch):
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    if "flatten_optimizer_state_dict" not in StateDictOptions.__dataclass_fields__:
        pytest.skip("Upstream DCP requires project-supported Torch >=2.7")
    from rcm.checkpointers.dcp_distill import DistributedCheckpointer_Distill
    monkeypatch.setenv("IMAGINAIRE_OUTPUT_PATH_LOCAL", str(tmp_path))
    c, model = configured_model(tmp_path, ema=True)
    model.init_optimizer_scheduler(c.optimizer, c.scheduler)
    for opt in model.optimizer_dict.values():
        for group in opt.param_groups:
            for p in group["params"]:
                p.grad = torch.ones_like(p)
        opt.step()
        opt.zero_grad()
    for scheduler in model.scheduler_dict.values():
        scheduler.step()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    old = model.optimizer_dict
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        cp = DistributedCheckpointer_Distill(c.checkpoint, c.job, None)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        cp.save(model, old, model.scheduler_dict, scaler, iteration=7)
        model.init_optimizer_scheduler(c.optimizer, c.scheduler)
        with torch.no_grad():
            model.net.scale.add_(1)
            model.net_ema.scale.add_(1)
        assert cp.load(model, model.optimizer_dict, model.scheduler_dict, scaler) == 7
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, state[k])
        assert not any(k.startswith("net_teacher.") for k in state)
        for name, optimizer in model.optimizer_dict.items():
            for param, old_state in old[name].state.items():
                for key in ("step", "exp_avg", "exp_avg_sq", "master_param"):
                    torch.testing.assert_close(optimizer.state[param][key], old_state[key])
        cp.finalize()
    finally:
        dist.destroy_process_group()


def test_master_checkpoint_rejects_missing_state_before_partial_dcp_load():
    from types import SimpleNamespace
    from rcm.checkpointers.dcp import OptimizerWrapper
    from rcm.utils.torch_fused_adamw_dtensor import FusedAdamW
    model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    opt = FusedAdamW(model.parameters())
    wrapper = OptimizerWrapper(model, opt)
    state = wrapper.state_dict()
    metadata = {key: SimpleNamespace(properties=SimpleNamespace(dtype=torch.float32)) for key in state}
    reader = SimpleNamespace(read_metadata=lambda: SimpleNamespace(state_dict_metadata=metadata))
    wrapper.validate_checkpoint(reader, state)
    master = next(key for key in state if key.endswith('.master_param'))
    metadata.pop(master)
    with pytest.raises(ValueError, match='Incomplete FP32-master'):
        wrapper.validate_checkpoint(reader, state)
