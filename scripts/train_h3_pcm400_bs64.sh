#!/usr/bin/env bash
# Diagnostic: 400 native-320p PCM optimizer updates, effective global BS=64.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MANIFEST="${MANIFEST:-$PWD/data/h3_av50k_512x320_non_stride_alignment_validated_v2/manifest.jsonl}"
export PCM_WARMUP_STEPS=400 MAX_ITER=400 PCM_GRAD_ACCUM=2
export PCM_GRID_INTERPOLATION=sigma GRAD_CLIP_ENABLED=true GRAD_CLIP_NORM=1.0
export FIXED_PROMPT_AT_START=true FIXED_PROMPT_EVERY=100
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/h3-pcm400-bs64-sigmagrid-320p-traj4-vshift6-4x8-20261012}"
exec bash scripts/train_h3_pcm_crossres.sh "$@"
