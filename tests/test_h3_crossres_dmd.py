"""Cross-resolution score domain, RF targets, gradients and artifact contracts."""
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from h3dmd.config import H3Config
from h3dmd.methods import build_method
from h3dmd.methods.crossres_dmd import H3CrossResolutionDMDMethod
from h3dmd.methods.types import H3Latents
from h3dmd.upsampling import (NeuralCleanMap, frames, video, new_mapper, mapper_spec,
                              load_frozen_mapper, FrozenScoreMapper, file_sha256)
from h3dmd.mapper_pretraining import load_pair, select_pairs
from h3dmd.paired_cache import SAMPLING, NORMALIZATION
from test_dmd import make_batch, make_method


def settings():
    c = H3Config(method="h3_crossres_dmd")
    c.data.height = c.data.width = 64
    c.crossres.score_height = c.crossres.score_width = 96
    c.crossres.mapper_width = 16
    c.crossres.pretrain_steps = 1
    c.dmd.rf_loss_weight = 0
    c.dmd.data_mode = "rollout"
    c.dmd.audio_loss_weight = .2
    c.dmd.rollout_solver = "renoise"
    c.schedule.inference_q = [1, .75, .5, .25, 0]
    return c


def method():
    _, models = make_method()
    return H3CrossResolutionDMDMethod(settings(), models, score_mapper=new_mapper(settings())), models


@pytest.mark.parametrize("exit_step", [1, 2, 3, 4])
def test_teacher_fake_scores_use_new_grid_and_rollout_stays_native(exit_step):
    m, models = method()
    m.config.dmd.cycle_rollout_steps = True
    m.config.dmd.update_schedule = "alternating"
    m.config.dmd.student_update_frequency = 6
    calls = []
    original = m._call

    def record(role, latents, batch, layout, level):
        calls.append((role, latents.video.shape[-2:], latents.audio.shape, layout))
        return original(role, latents, batch, layout, level)

    with patch.object(m, "_call", side_effect=record):
        result = m.dmd_loss(make_batch(), step=(exit_step - 1) * 6)
    result.loss.backward()
    student_calls = [c for c in calls if c[0] == "student"]
    scores = [c for c in calls if c[0] != "student"]
    assert len(student_calls) == exit_step and all(tuple(c[1]) == (4, 4) for c in student_calls)
    assert [c[0] for c in scores] == ["critic", "teacher"]
    assert all(tuple(c[1]) == (6, 6) for c in scores)
    assert scores[0][3].num_target_video_rows == 2 * 3 * 3
    assert scores[0][3].num_target_audio_rows == student_calls[0][3].num_target_audio_rows
    assert scores[0][3].sequence_length > student_calls[0][3].sequence_length
    assert all(c[2] == make_batch().audio_latents.shape for c in calls)
    assert models.student.scale.grad is not None
    assert all(p.grad is None for p in models.teacher.parameters())
    assert all(p.grad is None for p in models.critic.parameters())
    assert all(p.grad is None for p in m.score_mapper.parameters())


def test_high_resolution_mean_loss_has_exact_mapper_input_jacobian():
    m, _ = method()
    batch = make_batch()
    low = torch.randn_like(batch.video_latents, requires_grad=True)
    audio = torch.randn_like(batch.audio_latents, requires_grad=True)
    clean = H3Latents(low, audio)
    level = m.training_scheduler.level(.4)
    saved_noisy = []
    original_noise = m._noise

    def noising(value, level):
        result = original_noise(value, level)
        saved_noisy.append(result[0])
        return result

    with patch.object(m, "rollout", return_value=clean), \
            patch.object(m, "_sample_level", return_value=level), \
            patch.object(m, "_noise", side_effect=noising):
        result = m.dmd_loss(batch)
    result.loss.backward()
    ref_low = low.detach().clone().requires_grad_(True)
    high = video(F.interpolate(frames(ref_low), size=(6, 6), mode="bicubic", align_corners=False), 1, 2)
    expected_loss = 0
    for mode, generated, coefficient in (("video", high, 1), ("audio", audio.detach(), .2)):
        noisy = getattr(saved_noisy[0], mode)
        sigma = getattr(level, mode + "_sigma")
        teacher = noisy + sigma * noisy * .5
        critic = noisy + sigma * noisy * -.25
        direction = (critic - teacher) / (generated.detach() - teacher).abs().mean().clamp_min(1e-5)
        expected_loss += coefficient * direction.square().mean()
        if mode == "video":
            grad = torch.autograd.grad(high, ref_low, 2 * direction / high.numel())[0]
            torch.testing.assert_close(low.grad, grad)
        else:
            torch.testing.assert_close(audio.grad, 2 * coefficient * direction / audio.numel())
    torch.testing.assert_close(result.loss, expected_loss)


def test_fake_fm_uses_mapped_generated_distribution_and_only_updates_fake():
    m, models = method()
    batch = make_batch()
    generated = H3Latents(torch.randn_like(batch.video_latents, requires_grad=True),
                          torch.randn_like(batch.audio_latents, requires_grad=True))
    with patch.object(m, "rollout", return_value=generated):
        result = m.critic_loss(batch, step=1)
    result.loss.backward()
    assert models.critic.scale.grad is not None
    assert generated.video.grad is None and generated.audio.grad is None
    assert models.student.scale.grad is None
    assert all(p.grad is None for p in m.score_mapper.parameters())
    assert result.metrics["score_height"] == 96
    assert result.metrics["source_rollout"] == 1


def artifact(tmp_path):
    c = settings()
    path = tmp_path / "frozen.pt"
    torch.save(dict(spec=mapper_spec(c), model=new_mapper(c).state_dict(), accepted=True,
                    pretrain_steps_completed=1, report={"mse_ratio": .9, "temporal_mse_ratio": .9}), path)
    c.crossres.mapper_checkpoint = str(path)
    c.crossres.mapper_sha256 = file_sha256(path)
    return c, path


def test_registration_loads_valid_artifact_and_rejects_modified_weights(tmp_path):
    c, path = artifact(tmp_path)
    _, models = make_method()
    m = build_method(c, models)
    assert isinstance(m, H3CrossResolutionDMDMethod)
    assert not any(p.requires_grad for p in m.score_mapper.parameters())
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        load_frozen_mapper(c)


def test_checkpoint_binding_rejects_other_mapper_even_with_partial_resume():
    first = FrozenScoreMapper(new_mapper(settings()), "a" * 64)
    other = FrozenScoreMapper(new_mapper(settings()), "b" * 64)
    with pytest.raises(RuntimeError, match="differs"):
        first.load_state_dict(other.state_dict(), strict=False)
    # Simulate DCP writing the buffer through an aliased state tensor before
    # calling the module's load hook. The immutable expected binding must win.
    first.artifact_sha256.copy_(other.artifact_sha256)
    with pytest.raises(RuntimeError, match="differs"):
        first.load_state_dict(other.state_dict(), strict=False)


def test_unaccepted_or_wrong_geometry_mapper_cannot_enter_dmd(tmp_path):
    c, path = artifact(tmp_path)
    c.crossres.mapper_sha256 = ""
    obj = torch.load(path, weights_only=True)
    obj["report"]["mse_ratio"] = 1.2
    torch.save(obj, path)
    with pytest.raises(ValueError, match="heldout"):
        load_frozen_mapper(c)
    c.crossres.score_width = 832
    with pytest.raises(ValueError, match="aspect ratio"):
        c.validate()


def test_paired_pretraining_rejects_timeline_metadata_mismatch(tmp_path):
    c = settings()
    provenance = dict(sampling=SAMPLING, fps=24, same_source_frames=True, normalization=NORMALIZATION)
    low, high = tmp_path / "low.pt", tmp_path / "high.pt"
    for path, side in ((low, 4), (high, 6)):
        torch.save(dict(video_latents=torch.randn(24, 2, side, side),
                        alignment_provenance={"sample_id": "x"}, pairing=provenance), path)
    row = dict(id="x", low_cache_path=str(low), high_cache_path=str(high),
               low_pixels=[64, 64], high_pixels=[96, 96], num_frames=5, pairing=provenance)
    assert load_pair(row, c)[1].shape[-2:] == (6, 6)
    altered = dict(row, pairing=dict(provenance, sampling="legacy_ffmpeg_fps24"))
    with pytest.raises(ValueError, match="aligned"):
        load_pair(altered, c)


def test_mapper_split_disallows_source_video_leakage():
    c = settings()
    c.crossres.train_samples = c.crossres.validation_samples = 1
    rows = [dict(id="a", group="shared", split="train"), dict(id="b", group="shared", split="val")]
    with pytest.raises(ValueError, match="overlap"):
        select_pairs(rows, c)


def test_registered_config_preserves_native_rf_and_f5g1():
    from imaginaire.utils.config_helper import override
    from rcm.configs.h3 import native_config
    from rcm.configs.registry_h3 import make_config
    c = override(make_config(), ["--", "experiment=h3_crossres_pure_dmd_320to480_traj4_vshift6_scorev2a2"])
    h = native_config(c.model.config.h3)
    assert h.method == "h3_crossres_dmd" and not h.pcm.enabled
    assert h.dmd.rf_loss_weight == 0 and h.dmd.data_mode == "rollout"
    assert h.dmd.student_update_frequency == 6 and h.dmd.cycle_rollout_steps
    assert h.data.frame_buckets[::6] == [124, 243, 362]
    assert h.schedule.inference_q == [1, .75, .5, .25, 0]
    assert h.schedule.video_shift == 6 and h.schedule.audio_shift == 3
    assert h.schedule.train_video_shift == h.schedule.train_audio_shift == 2
    assert h.training.learning_rate == 2e-6 and h.training.critic_learning_rate == 1e-6
    assert c.optimizer.master_weights and c.model.config.optimizer_fake_score.master_weights


def test_real_mapper_pretraining_freezes_and_reuses_accepted_artifact(tmp_path):
    import json
    from h3dmd.mapper_pretraining import pretrain
    torch.set_num_threads(2)
    torch.manual_seed(42)
    c = settings()
    c.crossres.pretrain_steps = 2
    c.crossres.max_pretrain_steps = 4
    c.crossres.validation_every = 1
    c.crossres.train_samples = 2
    c.crossres.validation_samples = 1
    c.crossres.learning_rate = 1e-3
    c.logging.tensorboard = False
    manifest = tmp_path / "pairs.jsonl"
    c.crossres.paired_manifest = str(manifest)
    provenance = dict(sampling=SAMPLING, fps=24, same_source_frames=True, normalization=NORMALIZATION)
    rows = []
    for index, value in enumerate((.8, 1.2, 1.)):
        low = torch.full((24, 2, 4, 4), value)
        high = torch.full((24, 2, 6, 6), value + .1)
        paths = {}
        for tag, tensor in (("low", low), ("high", high)):
            path = tmp_path / f"{index}_{tag}.pt"
            torch.save(dict(video_latents=tensor, pairing=provenance,
                            alignment_provenance={"sample_id": str(index)}), path)
            paths[tag + "_cache_path"] = str(path)
        rows.append(dict(id=str(index), group=str(index), split="train" if index < 2 else "val",
                         low_pixels=[64, 64], high_pixels=[96, 96], num_frames=5,
                         pairing=provenance, **paths))
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "mapper"
    pretrain(c, out, torch.device("cpu"))
    c.crossres.mapper_checkpoint = str(out / "frozen.pt")
    frozen = load_frozen_mapper(c)
    assert not any(p.requires_grad for p in frozen.parameters())
    digest = file_sha256(out / "frozen.pt")
    pretrain(c, out, torch.device("cpu"))
    assert digest == file_sha256(out / "frozen.pt")


def test_upstream_model_checkpoint_keeps_frozen_mapper_and_excludes_it_from_optimizers(tmp_path):
    from imaginaire.utils.config_helper import override
    from imaginaire.lazy_config import instantiate
    from rcm.configs.registry_h3 import make_config
    c, path = artifact(tmp_path)
    config = override(make_config(), ["--",
        "experiment=h3_crossres_pure_dmd_320to480_traj4_vshift6_scorev2a2",
        "model.config.net._target_=test_upstream_integration.cpu_network",
        "model.config.h3.data.height=64", "model.config.h3.data.width=64",
        "model.config.h3.crossres.score_height=96", "model.config.h3.crossres.score_width=96",
        "model.config.h3.crossres.mapper_width=16", "model.config.h3.crossres.pretrain_steps=1",
        f"model.config.h3.crossres.mapper_checkpoint={path}",
        f"model.config.h3.crossres.mapper_sha256={c.crossres.mapper_sha256}",
        "trainer.callbacks.fixed_prompt.enabled=false"])
    model = instantiate(config.model)
    model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    from rcm.checkpointers.dcp import ModelWrapper
    wrapper = ModelWrapper(model)
    state = wrapper.state_dict()
    assert "score_mapper.artifact_sha256" in state
    params = {id(p) for optimizer in model.optimizer_dict.values()
              for group in optimizer.param_groups for p in group["params"]}
    assert not any(id(p) in params for p in model.score_mapper.parameters())
    model.train()
    assert not model.score_mapper.training
    wrapper.load_state_dict(state)
