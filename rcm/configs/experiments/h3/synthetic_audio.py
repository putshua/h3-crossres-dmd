from copy import deepcopy
from hydra.core.config_store import ConfigStore


def register_experiments():
    # Keep Bidirectional pure-DMD LR/betas/decay, with FP32-master AdamW.
    # Native H3 RF schedule,
    # conditional teacher and modality loss reduction are explicit in h3.
    pure = {
        "job": {"name": "h3_synthetic_audio_dmd"},
        "trainer": {"max_iter": 3000, "seed": 42, "logging_iter": 10,
                    "grad_accum_iter": 1},
        "optimizer": {"lr": 1e-6, "weight_decay": 0.01, "betas": [0.0, 0.999]},
        "model": {"config": {"optimizer_fake_score": {
            "lr": 1e-7, "weight_decay": 0.01, "betas": [0.0, 0.999]},
        }},
        "checkpoint": {"save_iter": 500, "save_latest_iter": 100, "load_training_state": True},
        # Bidirectional scheduler, constant from the first update (no warmup).
        "scheduler": {"warm_up_steps": [0], "f_start": [1.0]},
    }
    pcm = deepcopy(pure)
    pcm["job"]["name"] = "h3_synthetic_audio_pcm_dmd"
    pcm["model"]["config"]["h3"] = {"pcm": {"enabled": True}}
    high = deepcopy(pure)
    high["job"]["name"] = "h3_synthetic_audio_dmd_512x320"
    high["model"]["config"]["h3"] = {"data": {
        "height": 320,
        "width": 512,
        "manifest": "/mnt/data/butong/codebase_h3/data/h3_promptset_v3_visual_only_512x320/manifest.jsonl",
    }}
    mixed = deepcopy(pure)
    mixed["job"]["name"] = "h3_synthetic_audio_dmd_mixed_3_3_1"
    # student_update_frequency=5, so each generator update consumes five
    # consecutive batches.  3:3:1 in generator updates becomes 15:15:5
    # training iterations and therefore stays aligned with role alternation.
    mixed_schedule = ["512x320"] * 15 + ["512x384"] * 15 + ["832x480"] * 5
    mixed["model"]["config"]["h3"] = {"data": {
        "manifest": "/mnt/data/butong/codebase_h3/data/h3_promptset_v3_visual_only_mixed_3_3_1/manifest.jsonl",
        "height": None,
        "width": None,
        "resolution_schedule": mixed_schedule,
    }, "dmd": {"student_update_frequency": 5}}
    pcm_av_size = deepcopy(pure)
    pcm_av_size["job"]["name"] = "h3_av50k_pcm10_dmd_size_3_3_1"
    # Keep every AV cache record available. Five consecutive iterations per
    # exact (resolution, duration) bucket align with the 5:1 critic/G cycle.
    size_schedule = [
        f"{resolution}x{frames}"
        for frames in (124, 243, 362)
        for resolution, weight in (("512x320", 3), ("512x384", 3), ("832x480", 1))
        for _ in range(5 * weight)
    ]
    pcm_av_size["model"]["config"]["h3"] = {
        "data": {
            "manifest": "/mnt/data/butong/codebase_h3/data/av50k_pcm_full_size_20260929/manifest.jsonl",
            "height": None,
            "width": None,
            "size_schedule": size_schedule,
        },
        "dmd": {"student_update_frequency": 5},
        "pcm": {"enabled": True, "data_mode": "data", "loss_weight": 10.0, "warmup_steps": 1},
    }
    pcm_av_320_traj8 = deepcopy(pure)
    pcm_av_320_traj8["job"]["name"] = "h3_av50k_pcm10_dmd_320p_traj8_vshift12"
    pcm_av_320_traj8["model"]["config"]["h3"] = {
        "data": {
            "manifest": "/mnt/data/butong/datasets/omni/av50k_v1/output/h3_all_success_512x320_integer_stride_v1/manifest.jsonl",
            "height": None,
            "width": None,
            "num_frames": None,
            # Five iterations per bucket keep the 5:1 critic/student cycle
            # aligned while alternating all PCM durations at fixed 512x320.
            "size_schedule": [f"512x320x{frames}" for frames in (124, 243, 362) for _ in range(5)],
        },
        "schedule": {
            "inference_q": [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0],
            "num_inference_steps": 9,
            "video_shift": 12.0,
            "train_video_shift": 12.0,
            "audio_shift": 3.0,
        },
        "dmd": {
            "rollout_steps": 8,
            "cycle_rollout_steps": False,
            "rollout_solver": "renoise",
            "student_update_frequency": 5,
        },
        "pcm": {"enabled": True, "data_mode": "data", "loss_weight": 10.0, "warmup_steps": 1},
    }
    pcm_av_320_traj6 = deepcopy(pcm_av_320_traj8)
    pcm_av_320_traj6["job"]["name"] = "h3_av50k_pcm10_dmd_320p_traj6_vshift12"
    pcm_av_320_traj6["model"]["config"]["h3"]["schedule"].update(
        inference_q=[1.0, 5 / 6, 4 / 6, 0.5, 2 / 6, 1 / 6, 0.0],
        num_inference_steps=7,
    )
    pcm_av_320_traj6["model"]["config"]["h3"]["dmd"]["rollout_steps"] = 6
    pcm_av_320_traj8_cycle = deepcopy(pcm_av_320_traj8)
    pcm_av_320_traj8_cycle["job"]["name"] = "h3_av50k_pcm10_dmd_320p_traj8_cycle_vshift12"
    pcm_av_320_traj8_cycle["model"]["config"]["h3"]["dmd"]["cycle_rollout_steps"] = True
    pcm_av_320_traj8_cycle_shift6_pcm50 = deepcopy(pcm_av_320_traj8_cycle)
    pcm_av_320_traj8_cycle_shift6_pcm50["job"]["name"] = "h3_av50k_pcm50_dmd_320p_traj8_cycle_vshift6"
    pcm_av_320_traj8_cycle_shift6_pcm50["model"]["config"]["h3"]["schedule"].update(
        video_shift=6.0, train_video_shift=6.0,
    )
    pcm_av_320_traj8_cycle_shift6_pcm50["model"]["config"]["h3"]["pcm"]["loss_weight"] = 50.0
    pcm_av_320_traj6_cycle_shift6_pcm50 = deepcopy(pcm_av_320_traj8_cycle_shift6_pcm50)
    pcm_av_320_traj6_cycle_shift6_pcm50["job"]["name"] = "h3_av50k_pcm50_dmd_320p_traj6_cycle_vshift6"
    pcm_av_320_traj6_cycle_shift6_pcm50["model"]["config"]["h3"]["schedule"].update(
        inference_q=[1.0, 5 / 6, 4 / 6, 0.5, 2 / 6, 1 / 6, 0.0],
        num_inference_steps=7,
    )
    pcm_av_320_traj6_cycle_shift6_pcm50["model"]["config"]["h3"]["dmd"]["rollout_steps"] = 6
    # Full non-stride AV cache; five fake updates per generator update.
    non_stride_experiments = []
    for steps, base in ((8, pcm_av_320_traj8_cycle_shift6_pcm50),
                        (6, pcm_av_320_traj6_cycle_shift6_pcm50)):
        experiment = deepcopy(base)
        name = f"h3_av50k_pcm50_dmd_320p_non_stride_traj{steps}_cycle_vshift6_f5g1"
        experiment["job"]["name"] = name
        experiment["trainer"].update(max_iter=2000, logging_iter=1)
        experiment["optimizer"]["lr"] = 2e-6
        experiment["model"]["config"]["optimizer_fake_score"]["lr"] = 1e-6
        native = experiment["model"]["config"]["h3"]
        native["data"].update(
            manifest="/mnt/data/butong/codebase_h3/data/h3_av50k_512x320_non_stride_v1/manifest.jsonl",
            size_schedule=[f"512x320x{frames}" for frames in (124, 243, 362) for _ in range(6)],
        )
        # This is a period of six optimizer iterations: 1 G and 5 fake.
        native["dmd"]["student_update_frequency"] = 6
        non_stride_experiments.append((name, experiment))
    # Fresh four-step comparisons, preserving PCM50 and fake:G=5:1.
    for video_shift in (6, 12):
        experiment = deepcopy(non_stride_experiments[0][1])
        name = f"h3_av50k_pcm50_dmd_320p_non_stride_traj4_cycle_vshift{video_shift}_f5g1"
        experiment["job"]["name"] = name
        native = experiment["model"]["config"]["h3"]
        native["dmd"]["rollout_steps"] = 4
        native["schedule"].update(
            inference_q=[1.0, 0.75, 0.5, 0.25, 0.0], num_inference_steps=5,
            video_shift=float(video_shift), train_video_shift=float(video_shift),
        )
        non_stride_experiments.append((name, experiment))
    # Separate score-noising shift from the four-step generation trajectory.
    experiment = deepcopy(next(node for name, node in non_stride_experiments
                               if name == "h3_av50k_pcm50_dmd_320p_non_stride_traj4_cycle_vshift6_f5g1"))
    name = "h3_av50k_pcm50_dmd_320p_non_stride_traj4_cycle_vshift6_score4_f5g1"
    experiment["job"]["name"] = name
    experiment["model"]["config"]["h3"]["schedule"]["train_video_shift"] = 4.0
    non_stride_experiments.append((name, experiment))
    # RCM/Wan2.1 p_D=UniformShift(3): full base-q range for both DMD
    # score evaluation and fake RF training, independent of rollout shift=6.
    experiment = deepcopy(experiment)
    name = "h3_av50k_pcm50_dmd_320p_non_stride_traj4_cycle_vshift6_score3_fullrange_f5g1"
    experiment["job"]["name"] = name
    experiment["model"]["config"]["h3"]["schedule"].update(
        train_video_shift=3.0, train_q_min=0.0, train_q_max=1.0,
    )
    non_stride_experiments.append((name, experiment))
    # User-selected follow-up: retain full-range sampling, reduce only
    # video score shift to 2; video rollout shift=6 and audio shift=3.
    experiment = deepcopy(experiment)
    name = "h3_av50k_pcm50_dmd_320p_non_stride_traj4_cycle_vshift6_score2_fullrange_f5g1"
    experiment["job"]["name"] = name
    experiment["model"]["config"]["h3"]["schedule"]["train_video_shift"] = 2.0
    non_stride_experiments.append((name, experiment))
    cs = ConfigStore.instance()
    for video_shift, pcm_enabled in ((6, True), (12, True), (6, False)):
        node = deepcopy(experiment)
        mode = "pcm50_dmd" if pcm_enabled else "pure_dmd"
        name = f"h3_av50k_{mode}_320p_non_stride_traj4_cycle_vshift{video_shift}_scorev2a2_fullrange_f5g1"
        node["job"]["name"] = name
        native = node["model"]["config"]["h3"]
        native["schedule"].update(video_shift=float(video_shift), train_video_shift=2.0, train_audio_shift=2.0)
        native["pcm"].update(enabled=pcm_enabled, loss_weight=50.0 if pcm_enabled else 0.0,
                             warmup_steps=1 if pcm_enabled else 0)
        non_stride_experiments.append((name, node))
    staged = deepcopy(next(node for name, node in non_stride_experiments
                           if name == "h3_av50k_pcm50_dmd_320p_non_stride_traj4_cycle_vshift12_scorev2a2_fullrange_f5g1"))
    name = "h3_av50k_pcm500_fake100_pure_dmd2000_traj4_vshift12"
    staged["job"]["name"] = name
    staged["trainer"].update(max_iter=2500, logging_iter=1)
    staged["trainer"]["callbacks"] = {"fixed_prompt": {"every_n": 100}}
    native = staged["model"]["config"]["h3"]
    native["method"] = "h3_pcm_warmup_dmd"
    native["pcm"].update(warmup_steps=500, warmup_only=True)
    native["staged"] = {"fake_warmup_steps": 100, "pcm_lr_warmup_updates": 50,
                        "dmd_lr_warmup_updates": 20, "pcm_lr_min_ratio": 0.1}
    # Six consecutive batches per duration also decouple duration from rollout depth.
    non_stride_experiments.append((name, staged))
    for name, node in (("h3_synthetic_audio_dmd", pure),
                       ("h3_synthetic_audio_pcm_dmd", pcm),
                       ("h3_synthetic_audio_dmd_512x320", high),
                       ("h3_synthetic_audio_dmd_mixed_3_3_1", mixed),
                       ("h3_av50k_pcm10_dmd_size_3_3_1", pcm_av_size),
                       ("h3_av50k_pcm10_dmd_320p_traj8_vshift12", pcm_av_320_traj8),
                       ("h3_av50k_pcm10_dmd_320p_traj8_cycle_vshift12", pcm_av_320_traj8_cycle),
                       ("h3_av50k_pcm50_dmd_320p_traj8_cycle_vshift6", pcm_av_320_traj8_cycle_shift6_pcm50),
                       ("h3_av50k_pcm50_dmd_320p_traj6_cycle_vshift6", pcm_av_320_traj6_cycle_shift6_pcm50),
                       ("h3_av50k_pcm10_dmd_320p_traj6_vshift12", pcm_av_320_traj6),
                       *non_stride_experiments):
        cs.store(group="experiment", name=name, package="_global_", node=node)
