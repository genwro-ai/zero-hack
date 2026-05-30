#!/bin/bash
#SBATCH --job-name=zero-hack-tf-pseudo
#SBATCH --array=0-2
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err
# Array job: END emails arrive per task; change to FAIL to reduce volume.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Train the Transformer on the PSEUDO-family grammars (domain randomization),
# leave-one-pseudo-grammar-out. Each pseudo profile is relabelled to its base
# family (see scripts/build_pseudo_splits.py), so this reuses decoder_training's
# transformer path + early stopping + id/ood next-step eval unchanged.
#
# Prereq (run once): build the pseudo splits on a LOGIN node (needs the data):
#   uv run python -m scripts.build_pseudo_splits
#
# Array index -> held-out pseudo grammar (the OOD test grammar):
#   0 -> mosfet  (holds out pseudo_low_power)
#   1 -> igbt    (holds out pseudo_power_vertical)
#   2 -> ic      (holds out pseudo_logic_dense)
#
# Submit all three folds:   sbatch slurm/train_transformer_pseudo.sbatch
# One fold:                 sbatch --array=2 slurm/train_transformer_pseudo.sbatch
#
# Env overrides:
#   DATASET=pseudo            # output namespace under outputs/models|metrics
#   SPLITS_DIR=data/generated/pseudo/splits
#   EPOCHS=60  PATIENCE=8  BATCH_SIZE=128  LR=3e-4  MAX_CONTEXT=192  K=5
#   D_MODEL / NHEAD / NUM_LAYERS / DIM_FEEDFORWARD  (transformer size)
#   FAMILY_DROPOUT=0.0        # >0 trains a usable <FAMILY_UNKNOWN> mode
#   EXTRA_ARGS="..."          # appended verbatim

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

FAMILIES=(mosfet igbt ic)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FAM="${FAMILIES[$TASK_ID]}"

DATASET="${DATASET:-pseudo}"
SPLITS_DIR="${SPLITS_DIR:-data/generated/pseudo/splits}"
MODEL_ROOT="${MODEL_ROOT:-outputs/models}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
EPOCHS="${EPOCHS:-60}"
PATIENCE="${PATIENCE:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-4}"
MAX_CONTEXT="${MAX_CONTEXT:-192}"
K="${K:-5}"
FAMILY_DROPOUT="${FAMILY_DROPOUT:-0.0}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm

if [[ ! -f "${SPLITS_DIR}/MOSFET_train.csv" ]]; then
  echo "pseudo splits not found in ${SPLITS_DIR}; run: uv run python -m scripts.build_pseudo_splits" >&2
  exit 1
fi

ARGS=(
  --model transformer
  --dataset "$DATASET"
  --splits-dir "$SPLITS_DIR"
  --holdout-family "$FAM"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --max-context "$MAX_CONTEXT"
  --k "$K"
  --family-dropout "$FAMILY_DROPOUT"
  --model-root "$MODEL_ROOT"
  --metrics-root "$METRICS_ROOT"
)

# Optional transformer-size overrides, only added when set.
[[ -n "${D_MODEL:-}" ]] && ARGS+=(--d-model "$D_MODEL")
[[ -n "${NHEAD:-}" ]] && ARGS+=(--nhead "$NHEAD")
[[ -n "${NUM_LAYERS:-}" ]] && ARGS+=(--num-layers "$NUM_LAYERS")
[[ -n "${DIM_FEEDFORWARD:-}" ]] && ARGS+=(--dim-feedforward "$DIM_FEEDFORWARD")

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "task_id=%s holdout=%s dataset=%s epochs=%s splits=%s\n" \
  "$TASK_ID" "$FAM" "$DATASET" "$EPOCHS" "$SPLITS_DIR"

time "$RUNNER" uv run python -m zero_hack.models.decoder_training "${ARGS[@]}"

ls -lh "${MODEL_ROOT}/${DATASET}/transformer_holdout_${FAM}" 2>/dev/null || true
