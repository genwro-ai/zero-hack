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

# Slurm array job for training the next-step neural baselines.
#
#   task 0 -> LSTM         (zero_hack.models.lstm.train)
#
# Each task trains on the chosen dataset's train/valid/test splits, logs
# train/val loss per epoch, writes learning history, and writes the next-step
# eval report under
#   outputs/models/<dataset>/<model>/{history.json,results.json,results.md}
#
# Submit:
#   sbatch slurm/train_models.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s005k          # which generated dataset to train on
#   OUTPUT_ROOT=data/generated   # where the dataset raw/ dirs live
#   MODEL_ROOT=outputs/models    # where checkpoints are written
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
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-3}"
MAX_CONTEXT="${MAX_CONTEXT:-192}"
K="${K:-3}"
RUNNER="${RUNNER:-srun}"

SPLITS_DIR="${OUTPUT_ROOT}/${DATASET}/splits"
REPORT_DIR="${MODEL_ROOT}/${DATASET}"
OUT_DIR="${REPORT_DIR}/${MODEL}"

mkdir -p outputs/slurm "$OUT_DIR"

ARGS=(
  --splits-dir "$SPLITS_DIR"
  --report-dir "$REPORT_DIR"
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

printf "task_id=%s model=%s dataset=%s splits=%s out=%s epochs=%s\n" \
  "$TASK_ID" "$MODEL" "$DATASET" "$SPLITS_DIR" "$OUT_DIR" "$EPOCHS"

time "$RUNNER" uv run python -m "zero_hack.models.${MODEL}.train" "${ARGS[@]}"

ls -lh "$OUT_DIR"
