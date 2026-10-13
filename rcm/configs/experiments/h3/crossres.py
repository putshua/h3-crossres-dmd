"""Independent pure-DMD, native-320p rollout / neural-480p AV supervision."""
from hydra.core.config_store import ConfigStore
from copy import deepcopy


EXPERIMENT_NAME = "h3_crossres_pure_dmd_320to480_traj4_vshift6_scorev2a2"
STAGED_EXPERIMENT_NAME = "h3_pcm500_fake100_crossres_dmd2000_320to480_traj4_vshift6_scorev2a2"
MULTIRES_EXPERIMENT_NAME = "h3_multires_pure_dmd_320_320_320_480_rcan_traj4_vshift12_scorev6a3"
BICUBIC_EXPERIMENT_NAME = "h3_multires_pure_dmd_320_320_320_480_bicubic_traj4_vshift12_scorev6a3"


def experiment_node():
    return {
        "job": {"name": EXPERIMENT_NAME},
        "optimizer": {"lr": 2e-6, "betas": [0.0, 0.999], "weight_decay": 0.01},
        "model": {"config": {
            "optimizer_fake_score": {"lr": 1e-6, "betas": [0.0, 0.999], "weight_decay": 0.01},
            "h3": {
                "method": "h3_crossres_dmd",
                "data": {
                    "manifest": "/mnt/data/butong/codebase_h3/data/h3_av50k_512x320_non_stride_v1/manifest.jsonl",
                    "width": 512, "height": 320, "num_frames": None,
                    # Each fake:G block consumes one duration. G itself rotates
                    # 5/10/15s rather than repeatedly landing on the same bucket.
                    "frame_buckets": [f for f in (124, 243, 362) for _ in range(6)],
                    "size_schedule": [], "batch_size": 1,
                },
                "crossres": {"score_width": 768, "score_height": 480, "mapper_kind": "dysample"},
                "schedule": {
                    "inference_q": [1.0, 0.75, 0.5, 0.25, 0.0], "num_inference_steps": 5,
                    "video_shift": 6.0, "audio_shift": 3.0,
                    "train_video_shift": 2.0, "train_audio_shift": 2.0,
                    "train_q_min": 0.0, "train_q_max": 1.0,
                },
                "dmd": {
                    "enabled": True, "data_mode": "rollout", "rf_loss_weight": 0.0,
                    "dmd_loss_weight": 1.0, "critic_loss_weight": 1.0,
                    "rollout_steps": 4, "rollout_grad_steps": 1, "rollout_solver": "renoise",
                    "cycle_rollout_steps": True, "update_schedule": "alternating",
                    "student_update_frequency": 6, "critic_update_frequency": 1,
                    "loss_reduction": "mean", "video_loss_weight": 1.0, "audio_loss_weight": 0.2,
                },
                "pcm": {"enabled": False, "loss_weight": 0.0, "warmup_steps": 0},
                "training": {"grad_clip_enabled": True, "max_grad_norm": 1.0},
            },
        }},
        "checkpoint": {"save_iter": 500, "save_latest_iter": 100, "load_training_state": True},
        "scheduler": {"warm_up_steps": [0], "f_start": [1.0]},
        "trainer": {"max_iter": 2000, "logging_iter": 1, "grad_accum_iter": 1,
                    "callbacks": {"fixed_prompt": {"every_n": 100, "run_at_start": True}}},
    }


def register_experiments():
    ConfigStore.instance().store(group="experiment", name=EXPERIMENT_NAME,
                                 package="_global_", node=experiment_node())
    multires = experiment_node()
    multires["job"]["name"] = MULTIRES_EXPERIMENT_NAME
    h = multires["model"]["config"]["h3"]
    h["method"] = "h3_multires_dmd"
    h["data"]["manifest"] = (
        "/mnt/data/butong/codebase_h3/data/h3_idle450k_fast_v3_text50_20261011/manifest.jsonl")
    h["crossres"].update(mapper_kind="rcan", mapper_width=128,
                          rollout_high_resolution_from_step=4)
    h["schedule"].update(video_shift=12.0, audio_shift=3.0,
                         train_video_shift=6.0, train_audio_shift=3.0)
    ConfigStore.instance().store(group="experiment", name=MULTIRES_EXPERIMENT_NAME,
                                 package="_global_", node=multires)
    bicubic = deepcopy(multires)
    bicubic["job"]["name"] = BICUBIC_EXPERIMENT_NAME
    bicubic["model"]["config"]["h3"]["crossres"].update(mapper_kind="bicubic")
    ConfigStore.instance().store(group="experiment", name=BICUBIC_EXPERIMENT_NAME,
                                 package="_global_", node=bicubic)
    node = experiment_node()
    node["job"]["name"] = STAGED_EXPERIMENT_NAME
    node["trainer"]["max_iter"] = 2500
    native = node["model"]["config"]["h3"]
    native["method"] = "h3_pcm_warmup_crossres_dmd"
    native["pcm"].update(enabled=True, data_mode="data", loss_weight=50.0,
                          warmup_steps=500, warmup_only=True,
                          grid_interpolation="sigma", steps_per_phase=24,
                          skipping_interval_steps=1,
                          phase_boundaries_q=[1.0, 0.75, 0.5, 0.25, 0.0])
    native["staged"] = {"fake_warmup_steps": 100, "pcm_lr_warmup_updates": 50,
                        "dmd_lr_warmup_updates": 20, "pcm_lr_min_ratio": 0.1}
    native["data"]["dmd_manifest"] = (
        "/mnt/data/butong/codebase_h3/data/h3_idle450k_fast_v3_text50_20261011/manifest.jsonl")
    ConfigStore.instance().store(group="experiment", name=STAGED_EXPERIMENT_NAME,
                                 package="_global_", node=node)
