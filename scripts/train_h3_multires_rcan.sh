#!/usr/bin/env bash
# 512x320 x3 -> 768x480 x1; all DMD/fake scores at 768x480.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export EXPERIMENT=h3_multires_pure_dmd_320_320_320_480_rcan_traj4_vshift12_scorev6a3
export METHOD=h3_multires_dmd
export HEIGHT=320 WIDTH=512 SCORE_HEIGHT=480 SCORE_WIDTH=768
export NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-32}" BATCH_SIZE="${BATCH_SIZE:-1}" GRAD_ACCUM=1
export LR="${LR:-2e-6}" CRITIC_LR="${CRITIC_LR:-1e-6}"
export MAX_ITER="${MAX_ITER:-2000}" LOG_EVERY=1
export STUDENT_UPDATE_FREQ=6 PCM_ENABLED=false
export RENOISE_VIDEO_SHIFT=12 DMD_VIDEO_SHIFT=6 DMD_AUDIO_SHIFT=3
export DMD_ROLLOUT_SOLVER=renoise DMD_CYCLE_STEPS=true
export GRAD_CLIP_ENABLED=true GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
export FIXED_PROMPT_EVERY="${FIXED_PROMPT_EVERY:-100}" FIXED_PROMPT_AT_START=true
export MANIFEST="${MANIFEST:-$PWD/data/h3_idle450k_fast_v3_text50_20261011/manifest.jsonl}"
export MAPPER_KIND=rcan
: "${MAPPER_CHECKPOINT:?Set MAPPER_CHECKPOINT to the agreed validated RCAN artifact}"
if [[ "${DRY_RUN:-0}" != 1 && "${DRY_RUN:-0}" != true ]]; then
  test -s "$(dirname "$MANIFEST")/CACHE_COMPLETE.json"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/h3-multires-rcan-pure-dmd-320x3-480x1-vshift12-scorev6a3}"
exec bash scripts/train_h3.sh "$@"
