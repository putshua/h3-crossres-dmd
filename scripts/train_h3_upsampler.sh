#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:$PWD/.venv/extra-site-packages${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-$PWD/.venv/bin/python}"
index="${PAIR_INDEX:?Set PAIR_INDEX to an immutable SQLite snapshot}"
output="${OUTPUT_DIR:-$PWD/outputs/h3-upsampler-320to480-dysample}"
args=(--standalone)
if (( ${NNODES:-1} > 1 )); then
  args=(--rdzv-backend c10d --rdzv-endpoint "${RDZV_ENDPOINT:?Set RDZV_ENDPOINT}"
        --rdzv-id "${RDZV_ID:-h3-upsampler}"
        --rdzv-conf "join_timeout=${RDZV_TIMEOUT:-900}${RDZV_IS_HOST:+,is_host=$RDZV_IS_HOST}")
fi
exec "$python_bin" -m torch.distributed.run --nnodes "${NNODES:-1}" \
  --nproc-per-node "${NPROC_PER_NODE:-4}" "${args[@]}" \
  -m scripts.train_h3_upsampler train --config "${CONFIG:-configs/h3_upsampler_train_320to480.json}" \
  --index "$index" --output "$output" --set \
  "batch_size=${BATCH_SIZE:-8}" "num_workers=${NUM_WORKERS:-2}" \
  "learning_rate=${LR:-0.0002}" "max_steps=${MAX_ITER:-20000}" \
  "mapper_kind=${MAPPER_KIND:-dysample}" "mapper_width=${MAPPER_WIDTH:-64}" \
  "crop_latent_frames=${CROP_LATENT_FRAMES:-5}" "validation_samples=${VALIDATION_SAMPLES:-192}" \
  "validation_every=${VALIDATION_EVERY:-500}" "save_every=${SAVE_EVERY:-100}" "$@"
