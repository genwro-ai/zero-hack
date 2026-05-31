#!/bin/bash
#SBATCH --job-name=zero-hack-lofo
#SBATCH --array=0-5
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
# This is an array job, so END emails arrive per-task (one per fold/model);
# switch to FAIL only if you want to reduce email volume.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Leave-one-family-out (LOFO) training array. Each task trains on 2 of the 3
# product families and holds the 3rd out entirely for evaluation. Both the LSTM
# and the Transformer are covered across the array.
#
# Index -> (model, holdout family) mapping (--array=0-5):
#   MODEL = MODELS[SLURM_ARRAY_TASK_ID % 2]   with MODELS=(lstm transformer)
#   FAM   = FAMILIES[SLURM_ARRAY_TASK_ID / 2] with FAMILIES=(mosfet igbt ic)
#
#   idx 0 -> (lstm,        mosfet)
#   idx 1 -> (transformer, mosfet)
#   idx 2 -> (lstm,        igbt)
#   idx 3 -> (transformer, igbt)
#   idx 4 -> (lstm,        ic)
#   idx 5 -> (transformer, ic)
#
# Submit the full LOFO sweep (both models x 3 folds):
#   sbatch slurm/train_lofo.sbatch
#
# Pin a single fold for BOTH models (override the family via env). With
# HOLDOUT_FAMILY set, FAM is forced to that family and only the model index
# matters, so submit just the first two tasks:
#   HOLDOUT_FAMILY=ic sbatch --array=0-1 slurm/train_lofo.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s005k          # which generated dataset's splits to train on
#   SPLITS_ROOT=data/generated   # where the dataset splits/ dirs live
#   MODEL_ROOT=outputs/models     # where checkpoints are written
#   METRICS_ROOT=outputs/metrics  # where reports/metrics are written
#   EPOCHS=15
#   BATCH_SIZE=128
#   LR=3e-3                       # LSTM only (transformer uses its tuned default)
#   MAX_CONTEXT=192
#   K=5                           # top-k for the next-step report
#   HOLDOUT_FAMILY=               # mosfet | igbt | ic (pins the fold)
#   EXTRA_ARGS="--device cuda"    # appended verbatim to the trainer CLI

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

RUNNER="${RUNNER:-srun}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

MODELS=(lstm transformer)
FAMILIES=(mosfet igbt ic)
MODEL="${MODELS[$((TASK_ID % 2))]}"
FAM="${FAMILIES[$((TASK_ID / 2))]}"

# Pin the fold to a single family if requested.
if [[ -n "${HOLDOUT_FAMILY:-}" ]]; then
  FAM="$HOLDOUT_FAMILY"
fi

DATASET="${DATASET:-valid_s005k}"
SPLITS_ROOT="${SPLITS_ROOT:-data/generated}"
MODEL_ROOT="${MODEL_ROOT:-outputs/models}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-3}"
MAX_CONTEXT="${MAX_CONTEXT:-192}"
K="${K:-5}"

SPLITS_DIR="${SPLITS_ROOT}/${DATASET}/splits"

mkdir -p outputs/slurm

if [[ "$MODEL" == "lstm" ]]; then
  CKPT="${MODEL_ROOT}/${DATASET}/lstm_holdout_${FAM}/best.pt"
  REPORT_DIR="${METRICS_ROOT}/${DATASET}/lstm_holdout_${FAM}"
  mkdir -p "$(dirname "$CKPT")" "$REPORT_DIR"

  ARGS=(
    --splits-dir "$SPLITS_DIR"
    --holdout-family "$FAM"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --max-context "$MAX_CONTEXT"
    --k "$K"
    --checkpoint-path "$CKPT"
    --report-dir "$REPORT_DIR"
  )

  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=($EXTRA_ARGS)
  fi

  printf "task_id=%s model=%s family=%s dataset=%s splits=%s ckpt=%s epochs=%s\n" \
    "$TASK_ID" "$MODEL" "$FAM" "$DATASET" "$SPLITS_DIR" "$CKPT" "$EPOCHS"

  time "$RUNNER" uv run python -m zero_hack.models.lstm.train "${ARGS[@]}"

  ls -lh "$(dirname "$CKPT")"
else
  # The transformer trainer auto-saves to
  #   <model-root>/<dataset>/transformer_holdout_<family>/best.pt
  # and writes id/ood next-step metrics under <metrics-root>, so we do not pass
  # an explicit checkpoint/report path here. No --lr: it has a tuned default.
  OUT_DIR="${MODEL_ROOT}/${DATASET}/transformer_holdout_${FAM}"

  ARGS=(
    --model transformer
    --dataset "$DATASET"
    --splits-dir "$SPLITS_DIR"
    --holdout-family "$FAM"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --max-context "$MAX_CONTEXT"
    --k "$K"
    --model-root "$MODEL_ROOT"
    --metrics-root "$METRICS_ROOT"
  )

  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=($EXTRA_ARGS)
  fi

  printf "task_id=%s model=%s family=%s dataset=%s splits=%s out=%s epochs=%s\n" \
    "$TASK_ID" "$MODEL" "$FAM" "$DATASET" "$SPLITS_DIR" "$OUT_DIR" "$EPOCHS"

  time "$RUNNER" uv run python -m zero_hack.models.decoder_training "${ARGS[@]}"
fi
