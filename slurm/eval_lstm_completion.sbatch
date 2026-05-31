#!/bin/bash
#SBATCH --job-name=zero-hack-lstm-completion
#SBATCH --array=0-1
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err
# Array job: END emails arrive per task; change to FAIL to reduce volume.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Free-running Task 2 completion eval on fixed data/eval holdout views for
# the LSTM checkpoints produced by slurm/train_lstm.sbatch.
#
#   task 0 -> lstm_teacher_forcing
#   task 1 -> lstm_scheduled_sampling
#
# Submit both:
#   sbatch slurm/eval_lstm_completion.sbatch
# Submit only the scheduled-sampling eval:
#   sbatch --array=1 slurm/eval_lstm_completion.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s100k
#   HOLDOUT_FAMILY=ic
#   EVAL_ROOT=data/eval
#   MODEL_ROOT=outputs/models
#   PREDS_ROOT=outputs/preds
#   METRICS_ROOT=outputs/metrics
#   VIEWS="id ood"
#   MAX_STEPS=400                # generation length cap
#   EXTRA_ARGS="--device cuda"   # appended verbatim to the eval CLI

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

RUNS=(teacher_forcing scheduled_sampling)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN="${RUNS[$TASK_ID]}"

DATASET="${DATASET:-valid_s100k}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
MODEL_ROOT="${MODEL_ROOT:-outputs/models}"
PREDS_ROOT="${PREDS_ROOT:-outputs/preds}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
VIEWS="${VIEWS:-id ood}"
MAX_STEPS="${MAX_STEPS:-400}"
RUNNER="${RUNNER:-srun}"

RUN_NAME="lstm_${RUN}"
MODEL_RUN_NAME="${RUN_NAME}_holdout_${HOLDOUT_FAMILY}"
CKPT="${MODEL_ROOT}/${DATASET}/${MODEL_RUN_NAME}/best.pt"
OUT_DIR="${METRICS_ROOT}/${DATASET}/holdout_${HOLDOUT_FAMILY}"

mkdir -p outputs/slurm "$OUT_DIR"

if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT (run slurm/train_lstm.sbatch first)" >&2
  exit 1
fi

ARGS=(
  --checkpoint "$CKPT"
  --dataset "$DATASET"
  --holdout-family "$HOLDOUT_FAMILY"
  --eval-root "$EVAL_ROOT"
  --preds-root "$PREDS_ROOT"
  --metrics-root "$METRICS_ROOT"
  --views $VIEWS
  --method-name "$RUN_NAME"
  --max-steps "$MAX_STEPS"
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "task_id=%s run=%s dataset=%s holdout=%s ckpt=%s out=%s\n" \
  "$TASK_ID" "$RUN" "$DATASET" "$HOLDOUT_FAMILY" "$CKPT" "$OUT_DIR"

time "$RUNNER" uv run python scripts/eval_lstm_completion.py "${ARGS[@]}"

find "$OUT_DIR" -maxdepth 3 -type f | sort
