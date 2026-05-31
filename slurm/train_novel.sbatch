#!/bin/bash
#SBATCH --job-name=zero-hack-novel
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
# Array job: END emails arrive per task; change to FAIL to reduce email volume.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Novel-family LOFO training array. Trains both the LSTM and the Transformer on
# the novel-family splits (data/generated/novel/splits/), leave-one-family-out.
# Each novel profile is relabelled to its base family token by build_novel_splits:
#   novel_mixed  -> mosfet
#   novel_sparse -> igbt
#   novel_dense  -> ic
# This mirrors train_lofo.sbatch but points at the novel dataset namespace.
#
# NOTE: The 3 novel profiles are density variants of ONE novel-grammar distribution,
# so leave-one-novel-family-out here measures in-distribution generalisation, NOT
# true OOD to an unseen grammar. The stronger experiment is training on the real
# families + novel augmentation and holding out a real family; treat this as a
# baseline / ablation on the novel-grammar data alone.
#
# Index -> (model, holdout family) mapping (--array=0-5):
#   MODEL = MODELS[SLURM_ARRAY_TASK_ID % 2]   with MODELS=(lstm transformer)
#   FAM   = FAMILIES[SLURM_ARRAY_TASK_ID / 2] with FAMILIES=(mosfet igbt ic)
#
#   idx 0 -> (lstm,        mosfet)   holds out novel_mixed
#   idx 1 -> (transformer, mosfet)   holds out novel_mixed
#   idx 2 -> (lstm,        igbt)     holds out novel_sparse
#   idx 3 -> (transformer, igbt)     holds out novel_sparse
#   idx 4 -> (lstm,        ic)       holds out novel_dense
#   idx 5 -> (transformer, ic)       holds out novel_dense
#
# Prereq (run once on a login node to build the splits):
#   uv run python -m scripts.build_novel_splits
#
# Submit the full sweep (both models x 3 folds):
#   sbatch slurm/train_novel.sbatch
#
# Pin a single fold for BOTH models (override the family via env). With
# HOLDOUT_FAMILY set, FAM is forced to that family and only the model index
# matters, so submit just the first two tasks:
#   HOLDOUT_FAMILY=ic sbatch --array=0-1 slurm/train_novel.sbatch
#
# Submit a single fold (e.g. novel_dense holdout) for both models:
#   sbatch --array=4,5 slurm/train_novel.sbatch
#
# Optional environment overrides:
#   DATASET=novel                # output namespace under outputs/models|metrics
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

DATASET="${DATASET:-novel}"
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

if [[ ! -f "${SPLITS_DIR}/MOSFET_train.csv" ]]; then
  echo "novel splits not found in ${SPLITS_DIR}; run: uv run python -m scripts.build_novel_splits" >&2
  exit 1
fi

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

  ls -lh "${MODEL_ROOT}/${DATASET}/transformer_holdout_${FAM}" 2>/dev/null || true
fi
