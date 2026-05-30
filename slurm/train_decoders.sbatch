#!/bin/bash
#SBATCH --job-name=zero-hack-decoders
#SBATCH --array=0-0
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Slurm job for the GPT decoder neural model.
#
#   task 0 -> gpt          (explicit GPT-style decoder blocks)
#
# Outputs:
#   outputs/models/<dataset>/<model>_holdout_<family>/best.pt
#   outputs/preds/<dataset>/holdout_<family>/{id,ood}/<model>/nextstep.csv
#   outputs/metrics/<dataset>/...
#
# Submit:
#   sbatch slurm/train_decoders.sbatch
#
# Common overrides:
#   DATASET=valid_s100k
#   HOLDOUT_FAMILY=ic
#   EPOCHS=30
#   PATIENCE=5
#   BATCH_SIZE=128
#   LR=3e-4
#   MAX_CONTEXT=192
#   LIMIT_PER_FAMILY=       # optional smoke cap
#   SLURM_OUTPUT_ROOT=outputs
#   EXTRA_ARGS="--d-model 256 --num-layers 4"

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

MODELS=(gpt)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$TASK_ID]}"

DATASET="${DATASET:-valid_s005k}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
SLURM_OUTPUT_ROOT="${SLURM_OUTPUT_ROOT:-outputs}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-4}"
MAX_CONTEXT="${MAX_CONTEXT:-192}"
K="${K:-5}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm "$SLURM_OUTPUT_ROOT"/{models,preds,metrics}

ARGS=(
  --model "$MODEL"
  --dataset "$DATASET"
  --generated-root "$GENERATED_ROOT"
  --eval-root "$EVAL_ROOT"
  --model-root "$SLURM_OUTPUT_ROOT/models"
  --preds-root "$SLURM_OUTPUT_ROOT/preds"
  --metrics-root "$SLURM_OUTPUT_ROOT/metrics"
  --holdout-family "$HOLDOUT_FAMILY"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --max-context "$MAX_CONTEXT"
  --k "$K"
  --device cuda
)

if [[ -n "${LIMIT_PER_FAMILY:-}" ]]; then
  ARGS+=(--limit-per-family "$LIMIT_PER_FAMILY")
fi

if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi

if [[ -n "${MAX_EVAL_BATCHES:-}" ]]; then
  ARGS+=(--max-eval-batches "$MAX_EVAL_BATCHES")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "task_id=%s model=%s dataset=%s holdout=%s output_root=%s epochs=%s\n" \
  "$TASK_ID" "$MODEL" "$DATASET" "$HOLDOUT_FAMILY" "$SLURM_OUTPUT_ROOT" "$EPOCHS"

time "$RUNNER" uv run python -m zero_hack.models.decoder_training "${ARGS[@]}"

find "$SLURM_OUTPUT_ROOT" -maxdepth 5 -type f | sort
