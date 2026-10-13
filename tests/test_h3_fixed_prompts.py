import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from h3dmd.config import H3Config
from h3dmd.fixed_prompts import load_fixed_prompts, prompt_hash, rank_prompt_slots, read_prompts
from rcm.callbacks.h3_fixed_prompt import H3FixedPromptCallback, evaluation_state
from test_inference import ZeroH3


def make_cache(root):
    rows = []
    for index in range(32):
        prompt = f"H3 test prompt {index}"
        row = dict(id=f"fixed_{index + 1:02d}", prompt=prompt, prompt_sha256=prompt_hash(prompt),
                   cache_path=f"{index}.safetensors")
        save_file({"prompt_embeds": torch.ones(index + 1, 5120, dtype=torch.bfloat16),
                   "text_token_tags": torch.ones(index + 1, dtype=torch.long)},
                  str(root / row["cache_path"]),
                  metadata={"prompt_sha256": row["prompt_sha256"], "text_encoder_layer": "50"})
        rows.append(row)
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def test_rewrite_keeps_high_motion_h3_visuals_and_source_order():
    root = Path("test_prompts/h3_high_motion_32")
    rows = read_prompts(root / "prompts.jsonl")
    originals = json.loads(Path("test_prompts/h3_high_motion_proportional_32_extended.json").read_text())["prompts"]
    from scripts.prepare_h3_fixed_prompts import parse_h3_prompt
    assert len(rows) == 32
    for i, row in enumerate(rows):
        assert row["id"] == f"high_motion_{i + 1:02d}"
        assert row["source_prompt"] == originals[i]["source_prompt"]
        assert row["source_index"] == originals[i]["index"]
        assert row["source_h3_prompt"] == originals[i]["h3_prompt"]
        visual = parse_h3_prompt(originals[i]["h3_prompt"])["integrated_multimodal_description"]
        assert row["h3_fields"]["integrated_multimodal_description"].startswith(visual + "\nAudio: ")
        assert "\nAudio: " in row["prompt"]
        assert row["prompt"] == "\n\n".join(f"{k}: {v}" for k, v in row["h3_fields"].items())
        if i in (5, 10, 29):
            assert row["h3_fields"]["non_diegetic_music"] == "N/A"
    assert "rope sounds stop" in rows[19]["prompt"].lower()
    assert "underwater listening perspective" in rows[18]["prompt"]


def test_h3_pe_parser_rejects_missing_or_duplicate_fields():
    from scripts.prepare_h3_fixed_prompts import parse_h3_prompt
    for text in ("", "overall_soundscape: rain", "integrated_multimodal_description: cat\noverall_soundscape: rain\noverall_soundscape: rain\nnon_diegetic_music: N/A"):
        with pytest.raises(ValueError, match="three ordered H3 PE fields"):
            parse_h3_prompt(text)


@pytest.mark.parametrize("world", [1, 4, 8, 32, 33, 64])
@pytest.mark.parametrize("count", [1, 7, 32])
def test_fsdp_every_rank_participates_and_every_prompt_saved_once(world, count):
    slots = [list(rank_prompt_slots(count, world, rank)) for rank in range(world)]
    assert len({len(s) for s in slots}) == 1
    assert sorted(i for rank in slots for i, keep in rank if keep) == list(range(count))
    assert all(0 <= i < count for rank in slots for i, keep in rank)


def test_cache_rejects_stale_prompt_or_wrong_encoder(tmp_path):
    manifest = make_cache(tmp_path)
    assert len(load_fixed_prompts(manifest)) == 32
    with pytest.raises(ValueError, match="wrong H3 text encoder layer"):
        load_fixed_prompts(manifest, expected_layer=49)
    rows = read_prompts(manifest)
    rows[0]["prompt"] = "changed"
    rows[0]["prompt_sha256"] = prompt_hash("changed")
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="stale prompt embedding"):
        load_fixed_prompts(manifest)


def test_cache_detects_pe_edit_after_encoding(tmp_path):
    manifest = make_cache(tmp_path)
    rows = read_prompts(manifest)
    source = tmp_path / "prompts.jsonl"
    source.write_text(manifest.read_text())
    for row in rows:
        row["prompt_source"] = "prompts.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(load_fixed_prompts(manifest)) == 32
    changed = read_prompts(source)
    changed[0]["prompt"] = "A newly edited PE"
    changed[0]["prompt_sha256"] = prompt_hash(changed[0]["prompt"])
    source.write_text("".join(json.dumps(r) + "\n" for r in changed))
    with pytest.raises(ValueError, match="PE source changed"):
        load_fixed_prompts(manifest)


def test_legacy_vae_adapter_uses_temporal_decode_and_outer_normalization():
    from h3dmd.decoders import LegacyVAEDecoder
    class RawVAE(torch.nn.Module):
        def decode_base(self, z):
            return z + 10
        def decode(self, z):
            return z + 1
    video = LegacyVAEDecoder(RawVAE(), {"latents_mean": [2], "latents_std": [3]}, video=True)
    audio = LegacyVAEDecoder(RawVAE(), {"sample_rate": 32000}, video=False)
    assert video.decode(torch.tensor(0.))[0] == 10
    assert audio.decode(torch.tensor(0.))[0] == 1
    assert video.config.latents_std == [3] and audio.config.sample_rate == 32000


def test_vae_loader_passes_string_paths_to_diffusers(tmp_path, monkeypatch):
    import sys
    from h3dmd.decoders import load_h3_vaes
    paths = []
    def from_pretrained(path, **kwargs):
        assert isinstance(path, str)
        paths.append(path)
        return torch.nn.Linear(1, 1)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=from_pretrained)))
    for name in ("video_vae", "audio_vae"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "config.json").write_text("{}")
    vaes = load_h3_vaes(tmp_path, "cpu")
    assert paths == [str(tmp_path / n) for n in ("video_vae", "audio_vae")]
    assert isinstance(vaes.video_vae, torch.nn.Linear)
    assert all(not p.requires_grad for p in vaes.audio_vae.parameters())


def test_eval_restores_rng_and_modes_even_on_failure():
    net = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Dropout())
    net[1].eval()
    before = torch.random.get_rng_state()
    py_before, np_before = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError):
        with evaluation_state(net):
            assert not net.training
            torch.rand(10)
            random.random()
            np.random.rand()
            raise RuntimeError("failed decoder")
    assert net.training and not net[1].training
    assert torch.equal(before, torch.random.get_rng_state())
    assert random.getstate() == py_before
    np.testing.assert_equal(np.random.get_state(), np_before)


def test_callback_samples_all_fixed_prompts_four_steps_and_repeats_seed(tmp_path):
    manifest = make_cache(tmp_path)
    callback = H3FixedPromptCallback(manifest=manifest, every_n=500, decode=False, run_at_start=False)
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    callback.trainer = SimpleNamespace()
    config = H3Config()
    config.data.height = config.data.width = 32
    config.schedule.inference_q = [1, .75, .5, .25, 0]
    config.dmd.rollout_solver = "renoise"
    model = SimpleNamespace(net=ZeroH3(), h3=config)
    callback.on_train_start(model)
    before = torch.random.get_rng_state()
    callback.on_training_step_end(model, {}, {}, torch.tensor(0.), 499)
    assert model.net.calls == 0
    callback.on_training_step_end(model, {}, {}, torch.tensor(0.), 500)
    assert model.net.calls == 32 * 4 and model.net.training
    assert torch.equal(before, torch.random.get_rng_state())
    output = tmp_path / "fixed_prompt" / "iter_000000500"
    assert len(list(output.glob("*.pt"))) == 32
    first = torch.load(output / "fixed_01_243f_seed1.pt", weights_only=True)
    assert first["metadata"]["num_forwards"] == 4
    assert first["metadata"]["video_sigmas"] == pytest.approx([1, 36/37, 12/13, .8, 0])
    assert first["metadata"]["audio_sigmas"] == pytest.approx([1, .9, .75, .5, 0])
    assert first["metadata"]["video_timesteps"] == pytest.approx([0, 1/37, 1/13, .2])
    assert first["video_latents"].shape == (1, 24, 72, 2, 2)
    assert first["audio_latents"].shape == (1, 2, 32, 405)
    callback.on_training_step_end(model, {}, {}, torch.tensor(0.), 1000)
    second = torch.load(tmp_path / "fixed_prompt/iter_000001000/fixed_01_243f_seed1.pt", weights_only=True)
    torch.testing.assert_close(first["video_latents"], second["video_latents"], rtol=0, atol=0)
    torch.testing.assert_close(first["audio_latents"], second["audio_latents"], rtol=0, atol=0)


def test_launcher_exposes_fixed_prompt_config_with_hydra_precedence():
    import shlex
    from test_train_launcher import launch
    from imaginaire.utils.config_helper import override
    from rcm.configs.registry_h3 import make_config
    result = launch("train", FIXED_PROMPT_EVERY="250", FIXED_PROMPT_DECODE="0",
                    FIXED_PROMPT_FRAMES="[124,243,362]", FIXED_PROMPT_COUNT="8")
    assert result.returncode == 0
    tokens = shlex.split(result.stdout.splitlines()[-1])
    c = override(make_config(), tokens[tokens.index("--"):])
    fixed = c.trainer.callbacks.fixed_prompt
    assert fixed.enabled and fixed.every_n == 250 and not fixed.decode
    assert fixed.run_at_start
    assert fixed.manifest == "test_prompts/h3_high_motion_32/encoded/manifest.jsonl"
    assert fixed.num_prompts == 8 and list(fixed.num_frames) == [124, 243, 362]


@pytest.mark.parametrize('iteration,solver,forwards', [
    (100, 'euler', 2), (500, 'euler', 2), (550, 'euler', 2),
    (600, 'euler', 2), (601, 'renoise', 4),
])
def test_staged_callback_uses_pcm_solver_through_fake_bootstrap(
        tmp_path, iteration, solver, forwards):
    manifest = make_cache(tmp_path)
    callback = H3FixedPromptCallback(manifest=manifest, num_prompts=1,
                                     decode=False, run_at_start=False)
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    config = H3Config()
    config.data.height = config.data.width = 32
    config.pcm.enabled = True
    config.pcm.warmup_steps = 500
    config.pcm.phase_boundaries_q = [1., .4, 0.]
    config.dmd.rollout_solver = 'renoise'
    config.schedule.inference_q = [1., .75, .5, .25, 0.]
    model = SimpleNamespace(net=ZeroH3(), h3=config,
                            method=SimpleNamespace(dmd_start=600))
    callback.on_train_start(model)
    callback.every_n_impl(None, model, {}, {}, None, iteration)
    saved = torch.load(tmp_path / f'fixed_prompt/iter_{iteration:09d}/fixed_01_243f_seed1.pt',
                       weights_only=True)
    assert saved['metadata']['solver'] == solver
    assert saved['metadata']['num_forwards'] == forwards
    assert model.net.calls == forwards
    assert saved['metadata']['q'] == pytest.approx(
        config.pcm.phase_boundaries_q if solver == 'euler' else config.schedule.inference_q)


def test_startup_callback_runs_at_zero_and_resume_step_without_extra_step_one(tmp_path, monkeypatch):
    manifest = make_cache(tmp_path)
    callback = H3FixedPromptCallback(manifest=manifest)
    callback.trainer = SimpleNamespace()
    model = SimpleNamespace(h3=H3Config())
    called = []
    monkeypatch.setattr(callback, "every_n_impl", lambda *args: called.append(args[-1]))
    callback.on_train_start(model, iteration=0)
    assert called == [0]
    callback.on_training_step_end(model, {}, {}, None, iteration=1)
    assert called == [0]
    callback.on_training_step_end(model, {}, {}, None, iteration=500)
    assert called == [0, 500]
    callback.on_train_start(model, iteration=750)
    assert called == [0, 500, 750]
    callback.run_at_start = False
    callback.on_train_start(model, iteration=0)
    assert called == [0, 500, 750]


def test_startup_decode_failure_is_raised_before_training(tmp_path, monkeypatch):
    manifest = make_cache(tmp_path)
    callback = H3FixedPromptCallback(manifest=manifest, num_prompts=1)
    callback.config = SimpleNamespace(job=SimpleNamespace(path_local=str(tmp_path)))
    callback.trainer = SimpleNamespace()
    config = H3Config()
    config.data.height = config.data.width = 32
    model = SimpleNamespace(net=ZeroH3(), h3=config)
    def fail(*args):
        raise ImportError("simulated VAE import failure")
    monkeypatch.setattr("rcm.callbacks.h3_fixed_prompt.load_vaes", fail)
    before = torch.random.get_rng_state()
    with pytest.raises(ImportError, match="simulated VAE"):
        callback.on_train_start(model)
    assert model.net.training and torch.equal(before, torch.random.get_rng_state())
    assert len(list((tmp_path / "fixed_prompt/iter_000000000").glob("*.pt"))) == 1
