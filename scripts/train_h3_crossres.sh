#!/usr/bin/env bash
# Registered native-320p rollout / pretrained neural-480p AV-score experiment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'HELP'
H3 cross-resolution pure DMD: four native 512x320 steps, 768x480 AV score.
Default: cycle backward-sim exits 1..4, renoise, video/audio rollout shift 6/3,
score shift 2/2, G/f LR 2e-6/1e-6, five fake updates per G, FSDP32, BS1.
First prepare matched pairs, then set PRETRAIN_MAPPER=1 to train and validate
the mapper before DMD. An existing accepted MAPPER_CHECKPOINT skips pretraining.
NNODES=4 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=32 BATCH_SIZE=1 \
  RDZV_ENDPOINT=<rendezvous-ip>:29503 PRETRAIN_MAPPER=1 bash scripts/train_h3_crossres.sh
Set DRY_RUN=1 to print the training command without encoding/loading weights.
MAPPER_KIND=dysample|carafe|liif|rcan, MAPPER_CHECKPOINT, MAPPER_SHA256,
PAIRED_MANIFEST, MAPPER_OUTPUT_DIR; ordinary train_h3.sh environment overrides
also apply. This wrapper pins the accepted 320p/480p geometry and pure-DMD mode.
HELP
  exit 0
fi
export EXPERIMENT="${EXPERIMENT:-h3_crossres_pure_dmd_320to480_traj4_vshift6_scorev2a2}"
export METHOD=h3_crossres_dmd
export HEIGHT=320 WIDTH=512 SCORE_HEIGHT=480 SCORE_WIDTH=768
export NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-32}" BATCH_SIZE="${BATCH_SIZE:-1}"
export LR="${LR:-2e-6}" CRITIC_LR="${CRITIC_LR:-1e-6}"
export MAX_ITER="${MAX_ITER:-2000}" LOG_EVERY="${LOG_EVERY:-1}"
export STUDENT_UPDATE_FREQ=6 PCM_ENABLED=false
export RENOISE_VIDEO_SHIFT="${RENOISE_VIDEO_SHIFT:-6}"
export DMD_VIDEO_SHIFT="${DMD_VIDEO_SHIFT:-2}" DMD_AUDIO_SHIFT="${DMD_AUDIO_SHIFT:-2}"
export DMD_ROLLOUT_SOLVER=renoise DMD_CYCLE_STEPS=true
export FIXED_PROMPT_EVERY="${FIXED_PROMPT_EVERY:-100}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/h3-crossres-pure-dmd-320to480-traj4-vshift6}"
export RDZV_ID="${RDZV_ID:-h3-crossres-pure-dmd-320to480-traj4}"
export MANIFEST="${MANIFEST:-$PWD/data/h3_av50k_512x320_non_stride_v1/manifest.jsonl}"
export MAPPER_KIND="${MAPPER_KIND:-dysample}"
export PAIRED_MANIFEST="${PAIRED_MANIFEST:-$PWD/data/h3_crossres_320_480_v1/pairs.jsonl}"
mapper_dir="${MAPPER_OUTPUT_DIR:-$PWD/outputs/h3-crossres-mapper-$MAPPER_KIND}"
export MAPPER_CHECKPOINT="${MAPPER_CHECKPOINT:-$mapper_dir/frozen.pt}"
python_bin="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export PYTHONPATH="$PWD:$PWD/.venv/extra-site-packages${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${DRY_RUN:-0}" != 1 && "${DRY_RUN:-0}" != true ]]; then
  if [[ ! -f "$MAPPER_CHECKPOINT" ]]; then
    if [[ "${PRETRAIN_MAPPER:-0}" != 1 ]]; then
      echo "Validated mapper missing: $MAPPER_CHECKPOINT. Build paired cache, then set PRETRAIN_MAPPER=1." >&2
      exit 2
    fi
    rendezvous=(--standalone)
    if (( NNODES > 1 )); then
      endpoint="${RDZV_ENDPOINT:-${MASTER_ADDR:?Set MASTER_ADDR or RDZV_ENDPOINT}:${MASTER_PORT:-29503}}"
      rendezvous=(--rdzv-backend c10d --rdzv-endpoint "$endpoint"
                  --rdzv-id "${RDZV_ID:-h3-crossres}-mapper"
                  --rdzv-conf "join_timeout=${RDZV_TIMEOUT:-900}${RDZV_IS_HOST:+,is_host=$RDZV_IS_HOST}")
    fi
    "$python_bin" -m torch.distributed.run --nnodes "$NNODES" --nproc-per-node "$NPROC_PER_NODE" \
      "${rendezvous[@]}" -m scripts.pretrain_h3_score_upsampler --output "$(dirname "$MAPPER_CHECKPOINT")" -- \
      "experiment=$EXPERIMENT" "model.config.h3.crossres.mapper_kind=$MAPPER_KIND" \
      "model.config.h3.crossres.paired_manifest=$PAIRED_MANIFEST"
  fi
  export MAPPER_SHA256="${MAPPER_SHA256:-$("$python_bin" -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$MAPPER_CHECKPOINT")}" 
fi
exec bash scripts/train_h3.sh "$@"
