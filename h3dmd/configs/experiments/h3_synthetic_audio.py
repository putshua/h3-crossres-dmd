from h3dmd.registry import EXPERIMENTS


@EXPERIMENTS.register("h3_synthetic_audio_dmd")
def h3_synthetic_audio_dmd():
    return {
        "defaults": ["model/h3_full", "method/pure_dmd", "distributed/fsdp32"],
        "schedule": {
            "inference_q": [1.0, 0.75, 0.5, 0.25, 0.0],
            "num_inference_steps": 5,
            "train_q_min": 0.02,
            "train_q_max": 0.98,
        },
        "data": {
            "manifest": "/mnt/data/butong/datasets/longvideo_clean_pipeline/output/h3_promptset_synthetic_audio_v1/manifest.jsonl",
            "cache_root": "",
            "batch_size": 1,
            "num_workers": 2,
            "height": 288,
            "width": 512,
            "frame_buckets": [124, 243, 362],
            "shuffle": True,
            "drop_last": False,
        },
        "training": {
            "seed": 42,
            "max_steps": 3000,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-06,
            "critic_learning_rate": 1e-07,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        },
        "checkpoint": {
            "output_dir": "outputs/h3_dmd_synthetic_audio_full",
            "save_every": 500,
            "keep_last": 2,
        },
        "logging": {"log_every": 1, "tensorboard": True, "jsonl": True},
        "pcm": {"data_mode": "rollout"},
    }


@EXPERIMENTS.register("h3_synthetic_audio_pcm_dmd")
def h3_synthetic_audio_pcm_dmd():
    config = h3_synthetic_audio_dmd()
    config["defaults"][1] = "method/pcm_dmd"
    config["checkpoint"]["output_dir"] = "outputs/h3_pcm_dmd_synthetic_audio_full"
    return config


@EXPERIMENTS.register("h3_synthetic_audio_dmd_mixed_3_3_1")
def h3_synthetic_audio_dmd_mixed_3_3_1():
    config = h3_synthetic_audio_dmd()
    config["data"].update(
        manifest="/mnt/data/butong/codebase_h3/data/h3_promptset_v3_visual_only_mixed_3_3_1/manifest.jsonl",
        height=None,
        width=None,
        resolution_schedule=["512x320"] * 15 + ["512x384"] * 15 + ["832x480"] * 5,
    )
    config["checkpoint"]["output_dir"] = "outputs/h3_dmd_synthetic_audio_mixed_3_3_1"
    return config
