#!/bin/bash
#SBATCH --job-name=zero-hack-train
#SBATCH --array=0-0
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=120GB # Fair share on Leonardo: 120GB * gpus-per-task
#SBATCH --cpus-per-task=8 # Fair share on Leonardo: 8 * gpus-per-task
#SBATCH --time=4:00:00 # Time limit in HH:MM:SS, up to 24:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Slurm array job for training the LSTM next-step baseline.
#
#   task 0 -> LSTM         (zero_hack.models.lstm.train)
#
# The model trains on generated train/valid splits and evaluates next-step on
# fixed data/eval/<dataset>/holdout_<family>/{id,ood}.
#
# Submit:
#   sbatch slurm/train_models.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s005k          # which generated dataset to train on
#   OUTPUT_ROOT=data/generated   # where the dataset raw/ dirs live
#   MODEL_ROOT=outputs/models    # where checkpoints are written
#   EVAL_ROOT=data/eval
#   PREDS_ROOT=outputs/preds
#   METRICS_ROOT=outputs/metrics
#   HOLDOUT_FAMILY=ic
#   EPOCHS=10
#   BATCH_SIZE=128
#   LR=3e-3
#   MAX_CONTEXT=192
#   LIMIT_PER_FAMILY=            # cap records per family (smoke runs)
#   K=3                          # top-k for the eval summary
#   EXTRA_ARGS="--device cuda"   # appended verbatim to the trainer CLI
#   UV_CACHE_DIR=$SCRATCH/.cache/uv
#   UV_INSTALL_DIR=$SCRATCH/.local/bin
#   LEONARDO_LOAD_MODULES=0

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

MODELS=(lstm)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$TASK_ID]}"

DATASET="${DATASET:-valid_s005k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/generated}"
MODEL_ROOT="${MODEL_ROOT:-outputs/models}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
PREDS_ROOT="${PREDS_ROOT:-outputs/preds}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-3}"
MAX_CONTEXT="${MAX_CONTEXT:-192}"
K="${K:-3}"
RUNNER="${RUNNER:-srun}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"

SPLITS_DIR="${OUTPUT_ROOT}/${DATASET}/splits"
METHOD_NAME="$MODEL"
MODEL_RUN_NAME="${METHOD_NAME}_holdout_${HOLDOUT_FAMILY}"
REPORT_DIR="${METRICS_ROOT}/${DATASET}/${MODEL_RUN_NAME}"
CKPT="${MODEL_ROOT}/${DATASET}/${MODEL_RUN_NAME}/best.pt"
OUT_DIR="$(dirname "$CKPT")"

mkdir -p outputs/slurm "$OUT_DIR"

ARGS=(
  --splits-dir "$SPLITS_DIR"
  --dataset "$DATASET"
  --eval-root "$EVAL_ROOT"
  --preds-root "$PREDS_ROOT"
  --metrics-root "$METRICS_ROOT"
  --holdout-family "$HOLDOUT_FAMILY"
  --method-name "$METHOD_NAME"
  --report-dir "$REPORT_DIR"
  --checkpoint-path "$CKPT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --max-context "$MAX_CONTEXT"
  --k "$K"
)

if [[ -n "${LIMIT_PER_FAMILY:-}" ]]; then
  ARGS+=(--limit-per-family "$LIMIT_PER_FAMILY")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "task_id=%s model=%s dataset=%s holdout=%s splits=%s out=%s epochs=%s\n" \
  "$TASK_ID" "$MODEL" "$DATASET" "$HOLDOUT_FAMILY" "$SPLITS_DIR" "$OUT_DIR" "$EPOCHS"

time "$RUNNER" uv run python -m "zero_hack.models.${MODEL}.train" "${ARGS[@]}"

ls -lh "$OUT_DIR"
