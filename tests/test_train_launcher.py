import os
import subprocess


def launch(mode, **variables):
    env = os.environ.copy()
    for key in (
        "NNODES",
        "NPROC_PER_NODE",
        "NODE_RANK",
        "RANK",
        "MASTER_ADDR",
        "MAX_STEPS",
        "MODE",
        "CONFIG",
        "OUTPUT_DIR",
        "MANIFEST",
        "FSDP_SHARD_SIZE",
        "BATCH_SIZE",
        "GRAD_ACCUM",
        "MAX_ITER",
        "LR",
        "CRITIC_LR",
        "SAVE_EVERY",
        "RDZV_ENDPOINT",
        "RDZV_ID",
    ):
        env.pop(key, None)
    env.update(DRY_RUN="1", MODE=mode, MASTER_ADDR="127.0.0.1", NODE_RANK="7")
    env.update(variables)
    return subprocess.run(
        ["bash", "scripts/train_h3.sh"], env=env, text=True, capture_output=True
    )


def test_launcher_32gpu_full_and_local_scope():
    full = launch("train")
    assert full.returncode == 0, full.stderr
    assert "--nnodes 8 --nproc-per-node 4" in full.stdout
    assert "--rdzv-backend c10d --rdzv-endpoint 127.0.0.1:29503" in full.stdout
    assert "trainer.max_iter=3000" in full.stdout
    assert "outputs/h3_dmd_synthetic_audio_full_v4_shift5" in full.stdout
    assert "h3_promptset_synthetic_audio_v1/manifest.jsonl" in full.stdout
    assert (
        "model.lora_rank=0" in full.stdout and "distributed.fsdp_size=32" in full.stdout
    )
    smoke = launch("cluster-smoke")
    assert "trainer.max_iter=3" in smoke.stdout
    local = launch("local-smoke")
    assert local.returncode == 0
    assert "--nnodes 1 --nproc-per-node 4 --standalone" in local.stdout
    assert (
        "model.lora_rank=64" in local.stdout and "trainer.max_iter=6" in local.stdout
    )
    assert "checkpoint.save_on_exit=false" in local.stdout
    changed = launch(
        "train",
        NNODES="4",
        NPROC_PER_NODE="4",
        FSDP_SHARD_SIZE="8",
        BATCH_SIZE="1",
        MAX_ITER="21",
        LR="4e-6",
        CRITIC_LR="7e-6",
        SAVE_EVERY="7",
    )
    assert changed.returncode == 0, changed.stderr
    assert "--nnodes 4 --nproc-per-node 4" in changed.stdout
    assert "distributed.fsdp_size=8" in changed.stdout
    assert "trainer.max_iter=21" in changed.stdout
    assert "data.batch_size=1" in changed.stdout
    assert "optimizer.lr=4e-6" in changed.stdout
    assert "model.config.optimizer_fake_score.lr=7e-6" in changed.stdout
    assert "checkpoint.save_iter=7" in changed.stdout
    assert launch("train", NNODES="1").returncode == 0
    assert launch("train", FSDP_SHARD_SIZE="10").returncode != 0
    assert launch("train", BATCH_SIZE="0").returncode != 0


def test_launcher_pcm_is_opt_in_and_maps_options():
    default = launch("train")
    assert "pcm.enabled=true" not in default.stdout
    pcm = launch(
        "train",
        PCM_ENABLED="1",
        PCM_DATA_MODE="rollout",
        PCM_LOSS_WEIGHT="0.2",
        PCM_PHASE_BOUNDARIES_Q="[1,0.5,0]",
        PCM_STEPS_PER_PHASE="12",
        PCM_GRID_INTERPOLATION="sigma",
        GRAD_CLIP_ENABLED="true",
        PCM_SKIP_STEPS="2",
        PCM_WARMUP_STEPS="100",
        PCM_WARMUP_ONLY="true",
        STUDENT_UPDATE_FREQ="3",
    )
    assert pcm.returncode == 0, pcm.stderr
    for expected in (
        "pcm.enabled=true",
        "pcm.data_mode=rollout",
        "pcm.loss_weight=0.2",
        "pcm.steps_per_phase=12",
        "pcm.grid_interpolation=sigma",
        "training.grad_clip_enabled=true",
        "pcm.skipping_interval_steps=2",
        "pcm.warmup_steps=100",
        "pcm.warmup_only=true",
        "dmd.student_update_frequency=3",
    ):
        assert expected in pcm.stdout
    assert launch("train", PCM_ENABLED="typo").returncode != 0


def test_launcher_can_select_registered_experiment():
    result = launch("train", EXPERIMENT="h3_synthetic_audio_dmd")
    assert result.returncode == 0, result.stderr
    assert "experiment=h3_synthetic_audio_dmd" in result.stdout
    assert "--config rcm/configs/registry_h3.py" in result.stdout
    assert "-m scripts.train" in result.stdout


def test_video_training_and_sampling_shifts_are_independent_overrides():
    result = launch('train', DMD_VIDEO_SHIFT='6', RENOISE_VIDEO_SHIFT='5')
    assert result.returncode == 0
    assert 'schedule.train_video_shift=6' in result.stdout
    assert 'schedule.video_shift=5' in result.stdout
    assert 'schedule.audio_shift=' not in result.stdout
