#!/usr/bin/env bash
# PCM real-AV bootstrap, fake bootstrap, then prompt-only cross-resolution DMD.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export EXPERIMENT="${EXPERIMENT:-h3_pcm500_fake100_crossres_dmd2000_320to480_traj4_vshift6_scorev2a2}"
export METHOD=h3_pcm_warmup_crossres_dmd
export HEIGHT=320 WIDTH=512 SCORE_HEIGHT=480 SCORE_WIDTH=768
export NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-32}" BATCH_SIZE=1 GRAD_ACCUM=1
export LR="${LR:-2e-6}" CRITIC_LR="${CRITIC_LR:-1e-6}"
export MAX_ITER="${MAX_ITER:-2500}" LOG_EVERY=1
export PCM_ENABLED=true PCM_DATA_MODE=data PCM_LOSS_WEIGHT="${PCM_LOSS_WEIGHT:-50}"
export PCM_GRID_INTERPOLATION="${PCM_GRID_INTERPOLATION:-sigma}"
export GRAD_CLIP_ENABLED="${GRAD_CLIP_ENABLED:-true}" GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export PCM_WARMUP_STEPS="${PCM_WARMUP_STEPS:-500}" PCM_WARMUP_ONLY=true STUDENT_UPDATE_FREQ=6
export DMD_ROLLOUT_SOLVER=renoise DMD_CYCLE_STEPS=true
export RENOISE_VIDEO_SHIFT="${RENOISE_VIDEO_SHIFT:-6}"
export DMD_VIDEO_SHIFT=2 DMD_AUDIO_SHIFT=2
export FIXED_PROMPT_EVERY="${FIXED_PROMPT_EVERY:-100}"
export MANIFEST="${MANIFEST:-$PWD/data/h3_av50k_512x320_non_stride_v1/manifest.jsonl}"
DMD_MANIFEST="${DMD_MANIFEST:-$PWD/data/h3_idle450k_fast_v3_text50_20261011/manifest.jsonl}"
if [[ "${DRY_RUN:-0}" != 1 && "${DRY_RUN:-0}" != true ]]; then
  if [[ ! -s "$(dirname "$DMD_MANIFEST")/CACHE_COMPLETE.json" ]]; then
    echo "DMD PE cache is not fully validated: $(dirname "$DMD_MANIFEST")/CACHE_COMPLETE.json" >&2
    exit 2
  fi
fi
export MAPPER_KIND=dysample
export MAPPER_CHECKPOINT="${MAPPER_CHECKPOINT:-$PWD/outputs/h3-crossres-dysample-artifacts-20261010/dysample_best_iter000013000_from_latest000013800.pt}"
export MAPPER_SHA256="${MAPPER_SHA256:-d2823b5b9b8d4cac4df29e2f5c539e048599a5c1dc4069297f191bf5810e2cf3}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/h3-pcm500-crossres-dmd-idle450k-20261011}"
exec bash scripts/train_h3.sh "model.config.h3.data.dmd_manifest=$DMD_MANIFEST" "model.config.h3.staged.pcm_gradient_accumulation_steps=${PCM_GRAD_ACCUM:-1}" "$@"
