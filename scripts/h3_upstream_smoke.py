"""Exercise the imported trainer, native H3 FSDP and upstream DCP resume.

torchrun --standalone --nproc_per_node=4 -m scripts.h3_upstream_smoke --output-dir /tmp/h3-smoke
Uses a small random H3 architecture, never the full pretrained checkpoint.
"""
import argparse
import os
from pathlib import Path


def build_small_h3(config, role, context):
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from h3dmd.h3.adapter import H3ModelAdapter, FP32_MODULE_PATTERNS
    from h3dmd.distributed import configure_parallel_model

    model = MiniMaxH3Transformer3DModel(num_attention_heads=4, attention_head_dim=16,
        hidden_size=64, num_layers=2, num_refiner_layers=1, ffn_dim=128,
        text_dim=32, freq_dim=16, time_embed_hidden_dim=64, time_embed_dim=32, rope_freq_dim=2)
    for name, p in model.named_parameters():
        p.data = p.data.to(torch.float32 if role == "ema" or any(n in name for n in FP32_MODULE_PATTERNS)
                          else torch.bfloat16)
    adapter = H3ModelAdapter(model)
    if role in ("teacher", "ema"):
        adapter.freeze()
    else:
        adapter.enable_gradient_checkpointing()
    return configure_parallel_model(adapter, role, config.distributed, context)


def main():
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from imaginaire.utils import distributed
    from imaginaire.utils.config_helper import override
    from imaginaire.lazy_config import instantiate
    from rcm.configs.registry_h3 import make_config
    from rcm.callbacks.h3 import H3TrainingCallback

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--pcm", action="store_true")
    parser.add_argument("--ema", action="store_true")
    args = parser.parse_args()
    if args.num_frames < 5 or (args.num_frames - 5) % 17:
        parser.error("num-frames must equal 17*n+5")
    os.environ["IMAGINAIRE_OUTPUT_PATH_LOCAL"] = args.output_dir
    distributed.init()
    config = override(make_config(), ["--", "trainer.max_iter=6", "checkpoint.save_iter=6",
        "model.config.net._target_=scripts.h3_upstream_smoke.build_small_h3",
        "model.config.h3.logging.tensorboard=false",
        # This random model has text_dim=32; production fixed prompts have
        # width 5120 and are exercised by the separate callback smoke test.
        "trainer.callbacks.fixed_prompt.enabled=false",
        f"model.config.h3.pcm.enabled={str(args.pcm).lower()}",
        f"model.config.ema.enabled={str(args.ema).lower()}"])
    config.validate()
    config.freeze()
    trainer = instantiate(config.trainer.type, config=config)
    model = instantiate(config.model)

    class AssertGradients(H3TrainingCallback):
        def on_before_optimizer_step(self, model, optimizer, scheduler, grad_scaler, iteration=0):
            params = [p for group in optimizer.param_groups for p in group["params"]]
            assert all(p.grad is not None for p in params), "Trainable H3 parameter missing gradient"
            return super().on_before_optimizer_step(model, optimizer, scheduler, grad_scaler, iteration)
    # Keep the actual callback lifecycle, adding a smoke assertion before clip.
    trainer.callbacks._callbacks[0].__class__ = AssertGradients
    batch = dict(prompt_embeds=torch.randn(1, 4, 32), text_token_tags=torch.ones(4, dtype=torch.long),
        video_latents=torch.zeros(1, 24, (args.num_frames - 5) // 17 * 5 + 2, 18, 32),
        audio_latents=torch.zeros(1, 2, 32, round(args.num_frames / 24 * 40)), sample_ids=["smoke"])
    trainer.train(model, DataLoader([batch], batch_size=None), None)
    assert model.optimizer_dict["net"].state[next(model.net.parameters())]["step"].item() == 2
    assert model.optimizer_dict["fake_score"].state[next(model.net_fake_score.parameters())]["step"].item() == 4
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    previous = model.optimizer_dict
    previous_schedulers = {k: s.state_dict() for k, s in model.scheduler_dict.items()}
    with torch.no_grad():
        next(model.net.parameters()).add_(1)
    model.init_optimizer_scheduler(config.optimizer, config.scheduler)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    # Same output directory: exercise upstream latest_checkpoint.txt auto-resume.
    restored = trainer.checkpointer.load(model, model.optimizer_dict, model.scheduler_dict, scaler)
    assert restored == 6
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v.to_local(), expected[k].to_local())
    for role, optimizer in model.optimizer_dict.items():
        for parameter, state in previous[role].state.items():
            for key in ("step", "exp_avg", "exp_avg_sq", "master_param"):
                a, b = state[key], optimizer.state[parameter][key]
                torch.testing.assert_close(a.to_local() if hasattr(a, "to_local") else a,
                                           b.to_local() if hasattr(b, "to_local") else b, rtol=0, atol=0)
        assert model.scheduler_dict[role].state_dict() == previous_schedulers[role]
    assert not any(k.startswith("net_teacher.") for k in expected)
    if distributed.is_rank0():
        text = f"PASS: {dist.get_world_size()} GPUs, small full-parameter H3, {args.num_frames} frames, PCM={args.pcm}, EMA={args.ema}; upstream trainer 6 iterations + fresh optimizer/scheduler DCP restore\n"
        Path(args.output_dir, "PASS.txt").write_text(text)
        print(text, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    print("Loading upstream H3 smoke dependencies...", flush=True)
    main()
