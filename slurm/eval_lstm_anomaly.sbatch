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
# This is a single (non-array) job so the anomaly eval set is generated exactly
# once before any checkpoint is scored. It then loops over the run names and
# evaluates whichever checkpoints exist (skipping missing ones).
#
#   1. make_eval_set.py  -> data/eval/<dataset>/{eval_input_anomaly,anomaly_truth}.csv
#   2. eval_lstm_anomaly.py per checkpoint -> ROC-AUC + best-F1 + validator baseline
#
# Submit:
#   sbatch slurm/eval_lstm_anomaly.sbatch
# Evaluate a pre-existing checkpoint dir (e.g. the old outputs/.../lstm/best.pt):
#   RUN_NAMES=lstm sbatch slurm/eval_lstm_anomaly.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s005k
#   SPLITS_ROOT=data/generated
#   MODEL_ROOT=outputs/models
#   METRICS_ROOT=outputs/metrics
#   EVAL_ROOT=data/eval
#   RUN_NAMES="lstm_teacher_forcing lstm_scheduled_sampling"  # checkpoint dirs to score
#   N_ANOMALY_VALID=300          # valid sequences/family in the anomaly set
#   N_ANOMALY_INVALID=300        # corrupted (invalid) sequences/family
#   THRESHOLD=                   # fixed avg-logprob threshold; empty = sweep for best F1
#   SEED=1729
#   FORCE_REGEN=0                # 1 = regenerate the anomaly set even if present
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
SPLITS_ROOT="${SPLITS_ROOT:-data/generated}"
MODEL_ROOT="${MODEL_ROOT:-outputs/models}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
RUN_NAMES="${RUN_NAMES:-lstm_teacher_forcing lstm_scheduled_sampling}"
N_ANOMALY_VALID="${N_ANOMALY_VALID:-300}"
N_ANOMALY_INVALID="${N_ANOMALY_INVALID:-300}"
SEED="${SEED:-1729}"
FORCE_REGEN="${FORCE_REGEN:-0}"
RUNNER="${RUNNER:-srun}"

SPLITS_DIR="${SPLITS_ROOT}/${DATASET}/splits"
EVAL_DIR="${EVAL_ROOT}/${DATASET}"

mkdir -p outputs/slurm "$EVAL_DIR"

# 1. Generate the anomaly eval set once (unless it already exists).
if [[ "$FORCE_REGEN" == "1" || ! -f "${EVAL_DIR}/eval_input_anomaly.csv" ]]; then
  echo "generating anomaly eval set -> ${EVAL_DIR}"
  uv run python scripts/make_eval_set.py \
    --splits-dir "$SPLITS_DIR" \
    --n-anomaly-valid "$N_ANOMALY_VALID" \
    --n-anomaly-invalid "$N_ANOMALY_INVALID" \
    --seed "$SEED" \
    --out-dir "$EVAL_DIR"
else
  echo "reusing existing anomaly eval set at ${EVAL_DIR}"
fi

# 2. Evaluate every checkpoint that exists.
THRESHOLD_ARGS=()
if [[ -n "${THRESHOLD:-}" ]]; then
  THRESHOLD_ARGS=(--threshold "$THRESHOLD")
fi

for RUN in $RUN_NAMES; do
  CKPT="${MODEL_ROOT}/${DATASET}/${RUN}/best.pt"
  OUT_DIR="${METRICS_ROOT}/${DATASET}/${RUN}/anomaly"
  if [[ ! -f "$CKPT" ]]; then
    echo "skip ${RUN}: checkpoint not found at ${CKPT}"
    continue
  fi
  mkdir -p "$OUT_DIR"
  echo "evaluating ${RUN} -> ${OUT_DIR}"

  ARGS=(
    --checkpoint "$CKPT"
    --eval-dir "$EVAL_DIR"
    --out "$OUT_DIR"
    "${THRESHOLD_ARGS[@]}"
  )
  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=($EXTRA_ARGS)
  fi

  time "$RUNNER" uv run python scripts/eval_lstm_anomaly.py "${ARGS[@]}"
  ls -lh "$OUT_DIR"
done
