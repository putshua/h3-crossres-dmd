#!/usr/bin/env bash
# Full-parameter H3 DMD. Run the same command on every allocated node:
# NNODES=8 NPROC_PER_NODE=4 FSDP_SHARD_SIZE=32 BATCH_SIZE=1 bash scripts/train_h3.sh
set -euo pipefail
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
  cat <<'HELP'
Usage: NNODES=8 NPROC_PER_NODE=4 FSDP_SHARD_SIZE=32 BATCH_SIZE=1 bash scripts/train_h3.sh
Defaults: full-parameter H3 DMD, synthetic_audio_v1, 512x288, 4-step,
          5/10/15s alternating, 3000 iterations, 8 nodes x 4 GPUs, FSDP32.
          Student LR=1e-5, critic LR=2e-6 (independent overrides).
          PyTorch CUDA fused AdamW + FP32 master/moments for both roles.
Environment overrides:
  NNODES, NPROC_PER_NODE, FSDP_SHARD_SIZE, BATCH_SIZE, GRAD_ACCUM,
  MAX_ITER (MAX_STEPS alias), LR, CRITIC_LR, WEIGHT_DECAY, GRAD_CLIP_ENABLED, GRAD_CLIP_NORM,
  SAVE_EVERY, SAVE_LATEST_EVERY (100), LOG_EVERY, NUM_WORKERS, EMA_ENABLED,
  FIXED_PROMPT_ENABLED (true), FIXED_PROMPT_EVERY (250), FIXED_PROMPT_AT_START (true, before training),
  FIXED_PROMPT_MANIFEST, FIXED_PROMPT_COUNT (32), FIXED_PROMPT_FRAMES ('[243]'),
  FIXED_PROMPT_SEED (1), FIXED_PROMPT_DECODE (true, false saves AV latents only),
  DMD_VIDEO_SHIFT (5), RENOISE_VIDEO_SHIFT (5); audio shift remains 3,
  STUDENT_UPDATE_FREQ, DMD_UPDATE_SCHEDULE, DMD_ROLLOUT_SOLVER,
  DMD_CYCLE_STEPS, DMD_LOSS_WEIGHT, LOSS_REDUCTION,
  PCM_ENABLED (default false), PCM_LOSS_WEIGHT, PCM_DATA_MODE,
  PCM_PHASE_BOUNDARIES_Q, PCM_STEPS_PER_PHASE, PCM_SKIP_STEPS,
  PCM_WARMUP_STEPS, PCM_WARMUP_ONLY,
  SCORE_HEIGHT, SCORE_WIDTH, MAPPER_KIND, MAPPER_CHECKPOINT, MAPPER_SHA256,
  PAIRED_MANIFEST (only the independently registered h3_crossres_dmd method),
  MASTER_ADDR, MASTER_PORT, RDZV_ENDPOINT, RDZV_ID, RDZV_TIMEOUT,
  RDZV_IS_HOST (optional c10d TCPStore host flag, true on rendezvous node),
  PYTHON_BIN, CONFIG (Hydra registry .py), EXPERIMENT (registered name), METHOD,
  MANIFEST (MANIFEST_PATH alias), PRETRAINED_PATH,
  HEIGHT (default 288), WIDTH (default 512), MIXED_RESOLUTION (1 enables per-sample resolution schedule),
  OUTPUT_DIR (or OUTPUT_ROOT/JOB_NAME), RESUME_PATH (RESUME alias),
  PREFLIGHT_SAMPLES (0=all), PREFLIGHT_WORKERS, DRY_RUN=1.
Modes: MODE=train (default), cluster-smoke (3 full-parameter iterations),
       local-smoke (defaults to 1x4 GPUs, LoRA64, 6 iterations, no final checkpoint).
Multi-node: set MASTER_ADDR or RDZV_ENDPOINT identically on all nodes.
            c10d assigns node ranks; NODE_RANK is not required.
Current alternating packed layouts require BATCH_SIZE=1 and GRAD_ACCUM=1.
Trailing Hydra key=value or legacy --set key=value overrides settings; full/local parameter scope
is selected by MODE. DRY_RUN prints the command without starting training.
HELP
  exit 0
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mode="${MODE:-train}"
case "$mode" in
  # DCP auto-resumes latest_checkpoint.txt in an existing output directory.
  # Fresh run: new video training/rollout noise schedules.
  train) default_nodes=8; default_iters=3000; default_job=h3_dmd_synthetic_audio_full_v4_shift5 ;;
  cluster-smoke) default_nodes=8; default_iters=3; default_job=h3_dmd_synthetic_audio_cluster_smoke_v4_shift5 ;;
  local-smoke) default_nodes=1; default_iters=6; default_job=h3_dmd_synthetic_audio_local_smoke_v4_shift5 ;;
  *) echo 'MODE must be train, cluster-smoke, or local-smoke' >&2; exit 2 ;;
esac
NNODES="${NNODES:-$default_nodes}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
FSDP_SHARD_SIZE="${FSDP_SHARD_SIZE:-32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
MAX_ITER="${MAX_ITER:-${MAX_STEPS:-$default_iters}}"
LR="${LR:-1.0e-5}"
CRITIC_LR="${CRITIC_LR:-2.0e-6}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
CONFIG="${CONFIG:-rcm/configs/registry_h3.py}"
EXPERIMENT="${EXPERIMENT:-h3_synthetic_audio_dmd}"
MANIFEST="${MANIFEST:-${MANIFEST_PATH:-/mnt/data/butong/datasets/longvideo_clean_pipeline/output/h3_promptset_synthetic_audio_v1/manifest.jsonl}}"
HEIGHT="${HEIGHT:-288}"
WIDTH="${WIDTH:-512}"
PRETRAINED_PATH="${PRETRAINED_PATH:-/mnt/data/butong/mmh3/MiniMax-H3/FL2VA}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT:-$PWD/outputs}/${JOB_NAME:-$default_job}}"
RESUME_PATH="${RESUME_PATH:-${RESUME:-}}"
MASTER_PORT="${MASTER_PORT:-29503}"
RDZV_ID="${RDZV_ID:-h3-dmd-synthetic-audio}"
RDZV_TIMEOUT="${RDZV_TIMEOUT:-900}"
DRY_RUN="${DRY_RUN:-0}"
for name in NNODES NPROC_PER_NODE FSDP_SHARD_SIZE BATCH_SIZE GRAD_ACCUM MAX_ITER SAVE_EVERY MASTER_PORT RDZV_TIMEOUT; do
  if [[ ! "${!name}" =~ ^[1-9][0-9]*$ ]]; then echo "$name must be a positive integer" >&2; exit 2; fi
done
TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
EFFECTIVE_FSDP=$((FSDP_SHARD_SIZE < TOTAL_GPUS ? FSDP_SHARD_SIZE : TOTAL_GPUS))
if (( TOTAL_GPUS % EFFECTIVE_FSDP )); then
  echo "Total GPUs ($TOTAL_GPUS) must be divisible by effective FSDP size ($EFFECTIVE_FSDP)" >&2; exit 2
fi
case "$DRY_RUN" in 1|true|True) print_only=1 ;; 0|false|False) print_only=0 ;; *) echo 'DRY_RUN must be 0/1 or false/true' >&2; exit 2 ;; esac
if (( NNODES == 1 )); then
  rendezvous=(--standalone)
else
  if [[ -z "${MASTER_ADDR:-}" && -z "${RDZV_ENDPOINT:-}" && "$print_only" == 0 ]]; then
    echo 'Multi-node training requires MASTER_ADDR or RDZV_ENDPOINT, identical on all nodes' >&2; exit 2
  fi
  rdzv_conf="join_timeout=$RDZV_TIMEOUT"
  if [[ -n "${RDZV_IS_HOST:-}" ]]; then
    case "$RDZV_IS_HOST" in
      1|true|True) rdzv_conf+=",is_host=true" ;;
      0|false|False) rdzv_conf+=",is_host=false" ;;
      *) echo 'RDZV_IS_HOST must be 0/1 or false/true' >&2; exit 2 ;;
    esac
  fi
  rendezvous=(--rdzv-backend c10d
    --rdzv-endpoint "${RDZV_ENDPOINT:-${MASTER_ADDR:-localhost}:$MASTER_PORT}"
    --rdzv-id "$RDZV_ID" --rdzv-conf "$rdzv_conf")
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
# Avoid all ranks writing bytecode into the shared source/environment tree.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/h3-python-cache}"
export PYTHONPATH="$PWD:$PWD/.venv/extra-site-packages${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
args=(
  --set "model.pretrained_path=$PRETRAINED_PATH" --set "data.manifest=$MANIFEST"
  --set "data.batch_size=$BATCH_SIZE" --set "data.num_workers=$NUM_WORKERS"
  --set "distributed.fsdp_size=$FSDP_SHARD_SIZE"
  --set "training.gradient_accumulation_steps=$GRAD_ACCUM"
  --set "training.max_steps=$MAX_ITER" --set "training.learning_rate=$LR"
  --set "training.critic_learning_rate=$CRITIC_LR"
  --set "training.weight_decay=${WEIGHT_DECAY:-0.01}"
  --set "training.max_grad_norm=${GRAD_CLIP_NORM:-1.0}"
  --set "checkpoint.output_dir=$OUTPUT_DIR" --set "checkpoint.save_every=$SAVE_EVERY"
  "checkpoint.save_latest_iter=${SAVE_LATEST_EVERY:-100}"
  --set "logging.log_every=${LOG_EVERY:-10}")
case "${MIXED_RESOLUTION:-0}" in
  1|true|True)
    args+=(--set "data.height=null" --set "data.width=null") ;;
  0|false|False)
    args+=(--set "data.height=$HEIGHT" --set "data.width=$WIDTH") ;;
  *) echo 'MIXED_RESOLUTION must be 0/1 or false/true' >&2; exit 2 ;;
esac
if [[ -n "$RESUME_PATH" ]]; then args+=(--set "checkpoint.resume=$RESUME_PATH"); fi
# Only explicitly supplied environment values override CONFIG. PCM remains off
# in the default Hydra experiment; changing datasets/configs does not force rollout PCM.
for mapping in \
  METHOD:method \
  DMD_VIDEO_SHIFT:schedule.train_video_shift DMD_AUDIO_SHIFT:schedule.train_audio_shift RENOISE_VIDEO_SHIFT:schedule.video_shift \
  STUDENT_UPDATE_FREQ:dmd.student_update_frequency \
  DMD_UPDATE_SCHEDULE:dmd.update_schedule DMD_ROLLOUT_SOLVER:dmd.rollout_solver \
  DMD_CYCLE_STEPS:dmd.cycle_rollout_steps DMD_LOSS_WEIGHT:dmd.dmd_loss_weight \
  LOSS_REDUCTION:dmd.loss_reduction GRAD_CLIP_ENABLED:training.grad_clip_enabled PCM_ENABLED:pcm.enabled \
  PCM_LOSS_WEIGHT:pcm.loss_weight PCM_DATA_MODE:pcm.data_mode \
  PCM_PHASE_BOUNDARIES_Q:pcm.phase_boundaries_q PCM_STEPS_PER_PHASE:pcm.steps_per_phase PCM_GRID_INTERPOLATION:pcm.grid_interpolation \
  PCM_SKIP_STEPS:pcm.skipping_interval_steps PCM_WARMUP_STEPS:pcm.warmup_steps \
  PCM_WARMUP_ONLY:pcm.warmup_only \
  SCORE_HEIGHT:crossres.score_height SCORE_WIDTH:crossres.score_width \
  MAPPER_KIND:crossres.mapper_kind MAPPER_CHECKPOINT:crossres.mapper_checkpoint \
  MAPPER_SHA256:crossres.mapper_sha256 PAIRED_MANIFEST:crossres.paired_manifest; do
  name="${mapping%%:*}"; key="${mapping#*:}"
  if [[ -n "${!name:-}" ]]; then
    value="${!name}"
    case "$name" in
      PCM_ENABLED|PCM_WARMUP_ONLY|DMD_CYCLE_STEPS|GRAD_CLIP_ENABLED)
        case "$value" in
          1|true|True) value=true ;; 0|false|False) value=false ;;
          *) echo "$name must be 0/1 or false/true" >&2; exit 2 ;;
        esac ;;
    esac
    args+=(--set "$key=$value")
  fi
done
args+=("$@")
# MODE changes only parameter scope and smoke checkpoint behavior, never FSDP size.
if [[ "$mode" == local-smoke ]]; then
  args+=(--set model.lora_rank=64 --set model.lora_alpha=64
    --set model.critic_lora_rank=64 --set model.critic_lora_alpha=64
    --set model.lora_path=null --set checkpoint.save_on_exit=false
    --set checkpoint.save_every=1000000)
else
  args+=(--set model.lora_rank=0 --set model.critic_lora_rank=0 --set model.lora_path=null)
fi
args+=(--set distributed.strategy=fsdp2 --set distributed.sequence_parallel_size=1)
# Translate the established --set interface to upstream Hydra paths.
# Native Hydra overrides (e.g. optimizer.betas=[0.0,0.999]) also pass through.
hydra_args=(--config "$CONFIG" -- "experiment=$EXPERIMENT")
for mapping in FIXED_PROMPT_ENABLED:enabled FIXED_PROMPT_EVERY:every_n \
  FIXED_PROMPT_AT_START:run_at_start FIXED_PROMPT_MANIFEST:manifest \
  FIXED_PROMPT_COUNT:num_prompts FIXED_PROMPT_FRAMES:num_frames \
  FIXED_PROMPT_SEED:seed FIXED_PROMPT_DECODE:decode; do
  name="${mapping%%:*}"; key="${mapping#*:}"
  if [[ -n "${!name:-}" ]]; then
    value="${!name}"
    case "$name" in
      FIXED_PROMPT_ENABLED|FIXED_PROMPT_AT_START|FIXED_PROMPT_DECODE)
        case "$value" in
          1|true|True) value=true ;; 0|false|False) value=false ;;
          *) echo "$name must be 0/1 or false/true" >&2; exit 2 ;;
        esac ;;
    esac
    hydra_args+=("trainer.callbacks.fixed_prompt.$key=$value")
  fi
done
while (( ${#args[@]} )); do
  if [[ "${args[0]}" != --set ]]; then
    [[ "${args[0]}" == -- ]] || hydra_args+=("${args[0]}")
    args=("${args[@]:1}"); continue
  fi
  (( ${#args[@]} >= 2 )) || { echo '--set needs key=value' >&2; exit 2; }
  setting="${args[1]}"; key="${setting%%=*}"; value="${setting#*=}"
  case "$key" in
    training.max_steps) key=trainer.max_iter ;;
    training.seed) key=trainer.seed ;;
    training.gradient_accumulation_steps) key=trainer.grad_accum_iter ;;
    training.learning_rate) key=optimizer.lr ;;
    training.critic_learning_rate) key=model.config.optimizer_fake_score.lr ;;
    training.weight_decay)
      hydra_args+=("model.config.optimizer_fake_score.weight_decay=$value")
      key=optimizer.weight_decay ;;
    checkpoint.save_every) key=checkpoint.save_iter ;;
    checkpoint.save_on_exit) key=checkpoint.save_on_exit ;;
    checkpoint.resume) key=checkpoint.load_path ;;
    logging.log_every) key=trainer.logging_iter ;;
    *) key="model.config.h3.$key" ;;
  esac
  hydra_args+=("$key=$value"); args=("${args[@]:2}")
done
if [[ -n "${EMA_ENABLED:-}" ]]; then
  case "$EMA_ENABLED" in
    1|true|True) hydra_args+=(model.config.ema.enabled=true) ;;
    0|false|False) hydra_args+=(model.config.ema.enabled=false) ;;
    *) echo 'EMA_ENABLED must be 0/1 or false/true' >&2; exit 2 ;;
  esac
fi
export IMAGINAIRE_OUTPUT_PATH_LOCAL="$OUTPUT_DIR"
command=("$PYTHON_BIN" -m torch.distributed.run --nnodes "$NNODES" --nproc-per-node "$NPROC_PER_NODE"
  "${rendezvous[@]}" -m scripts.train "${hydra_args[@]}")
printf 'H3 DMD: mode=%s, nodes=%s x GPUs=%s, FSDP=%s (effective=%s), batch=%s, accum=%s, max_iter=%s\n' \
  "$mode" "$NNODES" "$NPROC_PER_NODE" "$FSDP_SHARD_SIZE" "$EFFECTIVE_FSDP" "$BATCH_SIZE" "$GRAD_ACCUM" "$MAX_ITER"
printf '%q ' "${command[@]}"
printf '\n'
if (( print_only )); then exit 0; fi
"$PYTHON_BIN" -m h3dmd.cli.preflight_upstream --world-size "$TOTAL_GPUS" --local-gpus "$NPROC_PER_NODE" \
  --samples "${PREFLIGHT_SAMPLES:-96}" --workers "${PREFLIGHT_WORKERS:-8}" "${hydra_args[@]}"
mkdir -p "$OUTPUT_DIR"
exec "${command[@]}"
