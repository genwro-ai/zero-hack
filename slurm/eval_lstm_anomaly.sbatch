#!/bin/bash
#SBATCH --job-name=zero-hack-lstm-anomaly
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%x_%A.out
#SBATCH --error=outputs/slurm/%x_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Anomaly-detection (Task 3) eval for the LSTM, via sequence likelihood.
#
# This evaluates whichever checkpoints exist against fixed
# data/eval/<dataset>/holdout_<family>/{id,ood} anomaly sets.
#
# Submit:
#   sbatch slurm/eval_lstm_anomaly.sbatch
# Evaluate a pre-existing checkpoint dir (e.g. the old outputs/.../lstm/best.pt):
#   RUN_NAMES=lstm sbatch slurm/eval_lstm_anomaly.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s005k
#   HOLDOUT_FAMILY=ic
#   MODEL_ROOT=outputs/models
#   PREDS_ROOT=outputs/preds
#   METRICS_ROOT=outputs/metrics
#   EVAL_ROOT=data/eval
#   RUN_NAMES="lstm_teacher_forcing lstm_scheduled_sampling"  # checkpoint dirs to score
#   VIEWS="id ood"
#   THRESHOLD=                   # fixed avg-logprob threshold; empty = tune on calibration
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

DATASET="${DATASET:-valid_s005k}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"
MODEL_ROOT="${MODEL_ROOT:-outputs/models}"
PREDS_ROOT="${PREDS_ROOT:-outputs/preds}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
RUN_NAMES="${RUN_NAMES:-lstm_teacher_forcing lstm_scheduled_sampling}"
VIEWS="${VIEWS:-id ood}"
RUNNER="${RUNNER:-srun}"

EVAL_DIR="${EVAL_ROOT}/${DATASET}/holdout_${HOLDOUT_FAMILY}"

mkdir -p outputs/slurm "$EVAL_DIR"

if [[ ! -f "${EVAL_DIR}/id/eval_input_anomaly.csv" || \
      ! -f "${EVAL_DIR}/ood/eval_input_anomaly.csv" ]]; then
  echo "missing fixed anomaly eval set under ${EVAL_DIR}; run scripts/make_all_eval_sets.py first" >&2
  exit 2
fi

THRESHOLD_ARGS=()
if [[ -n "${THRESHOLD:-}" ]]; then
  THRESHOLD_ARGS=(--threshold "$THRESHOLD")
fi

for RUN in $RUN_NAMES; do
  MODEL_RUN_NAME="${RUN}_holdout_${HOLDOUT_FAMILY}"
  CKPT="${MODEL_ROOT}/${DATASET}/${MODEL_RUN_NAME}/best.pt"
  OUT_DIR="${METRICS_ROOT}/${DATASET}/holdout_${HOLDOUT_FAMILY}"
  if [[ ! -f "$CKPT" ]]; then
    echo "skip ${RUN}: checkpoint not found at ${CKPT}"
    continue
  fi
  mkdir -p "$OUT_DIR"
  echo "evaluating ${RUN} -> ${OUT_DIR}"

  ARGS=(
    --checkpoint "$CKPT"
    --dataset "$DATASET"
    --holdout-family "$HOLDOUT_FAMILY"
    --eval-root "$EVAL_ROOT"
    --preds-root "$PREDS_ROOT"
    --metrics-root "$METRICS_ROOT"
    --views $VIEWS
    --method-name "$RUN"
    "${THRESHOLD_ARGS[@]}"
  )
  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=($EXTRA_ARGS)
  fi

  time "$RUNNER" uv run python scripts/eval_lstm_anomaly.py "${ARGS[@]}"
  find "$OUT_DIR" -maxdepth 3 -type f | sort
done
