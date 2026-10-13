"""Native H3 settings within the upstream Hydra configuration tree."""
from dataclasses import asdict

from omegaconf import OmegaConf

from h3dmd.config import H3Config, _coerce_dataclass


def native_config(node):
    if OmegaConf.is_config(node):
        node = OmegaConf.to_container(node, resolve=True)
    config = _coerce_dataclass(H3Config, node)
    config.validate()
    return config


def native_defaults():
    # Legacy YAMLs remain readable, but the primary experiment is registered
    # in Hydra; runtime fields refer to the upstream trainer/optimizer config.
    value = asdict(H3Config())
    value["model"].update(pretrained_path="/mnt/data/butong/mmh3/MiniMax-H3/FL2VA",
                           loader="legacy_fl2va", lora_rank=0, critic_lora_rank=0)
    value["schedule"].update(video_shift=5.0, train_video_shift=5.0, inference_q=[1.0, 0.75, 0.5, 0.25, 0.0], num_inference_steps=5)
    value["dmd"].update(data_mode="rollout", rf_loss_weight=0.0,
        update_schedule="alternating", student_update_frequency=5,
        rollout_solver="renoise", cycle_rollout_steps=True, loss_reduction="mean")
    value["pcm"].update(enabled=False, data_mode="rollout")
    value["distributed"].update(strategy="fsdp2", fsdp_size=32)
    value["data"].update(
        manifest="/mnt/data/butong/datasets/longvideo_clean_pipeline/output/h3_promptset_synthetic_audio_v1/manifest.jsonl",
        batch_size=1, num_workers=2, height=288, width=512,
        frame_buckets=[124, 243, 362], drop_last=False)
    value["logging"].update(log_every="${trainer.logging_iter}", tensorboard=True, jsonl=True)
    value["training"].update(
        seed="${trainer.seed}", max_steps="${trainer.max_iter}",
        gradient_accumulation_steps="${trainer.grad_accum_iter}",
        learning_rate="${optimizer.lr}",
        critic_learning_rate="${model.config.optimizer_fake_score.lr}",
        weight_decay="${optimizer.weight_decay}",
    )
    value["checkpoint"].update(
        save_every="${checkpoint.save_iter}", resume="${checkpoint.load_path}",
        save_on_exit="${checkpoint.save_on_exit}",
    )
    return value
