#!/usr/bin/env bash
# Aligned non-stride native-320p PCM400 (BS64), fake100, crossres pure DMD2000 (BS32).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MANIFEST="${MANIFEST:-$PWD/data/h3_av50k_512x320_non_stride_alignment_validated_v2/manifest.jsonl}"
export PCM_WARMUP_STEPS=400 MAX_ITER="${MAX_ITER:-2500}" PCM_GRAD_ACCUM=2
export MAPPER_SHA256="${MAPPER_SHA256:-81a7b04cd00c6301bf8ec7b361c36d8c4dc048cf8af06212c28a47ed82a635cd}"
export PCM_GRID_INTERPOLATION=sigma GRAD_CLIP_ENABLED=true GRAD_CLIP_NORM=1.0
export FIXED_PROMPT_AT_START=true FIXED_PROMPT_EVERY=100
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/h3-pcm400-bs64-aligned26985-crossres-dmd2000-4x8-20261012}"
exec bash scripts/train_h3_pcm_crossres.sh "$@"
