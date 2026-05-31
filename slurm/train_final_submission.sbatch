#!/bin/bash
#SBATCH --job-name=zero-hack-final
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%A.out
#SBATCH --error=outputs/slurm/%x_%A.err

set -euo pipefail

# Final-submission GPT training run.
#
# Trains one model on all MOSFET + IGBT + IC valid_s100k training data plus
# the augmented rule-valid dataset. This is not a holdout experiment.
#
# Submit:
#   sbatch slurm/train_final_submission.sbatch
#
# Main overrides:
#   DATASET=valid_s100k
#   FINAL_DATASET=final_submission
#   METHOD_NAME=final_submission_gpt
#   PHASE_LOSS_WEIGHT=0.1
#   EPOCHS=20
#   PATIENCE=5
#   EXTRA_ARGS="--d-model 384 --num-layers 6"

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

DATASET="${DATASET:-valid_s100k}"
BASE_COUNT="${BASE_COUNT:-100000}"
FINAL_DATASET="${FINAL_DATASET:-final_submission}"
AUGMENT_COUNT="${AUGMENT_COUNT:-50000}"
AUGMENT_DATASET="${AUGMENT_DATASET:-augmented_s050k}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
SLURM_OUTPUT_ROOT="${SLURM_OUTPUT_ROOT:-outputs}"
RUNNER="${RUNNER:-srun}"

METHOD_NAME="${METHOD_NAME:-final_submission_gpt}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-2e-4}"
MAX_CONTEXT="${MAX_CONTEXT:-256}"
K="${K:-5}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-2000}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-300}"
SFT_AUGMENT_LIMIT="${SFT_AUGMENT_LIMIT:-20000}"
PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.1}"
FAMILY_DROPOUT="${FAMILY_DROPOUT:-0.25}"
STEP_DROPOUT="${STEP_DROPOUT:-0.05}"

D_MODEL="${D_MODEL:-256}"
NHEAD="${NHEAD:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1024}"
DROPOUT="${DROPOUT:-0.1}"
SEED="${SEED:-1729}"

mkdir -p outputs/slurm "$SLURM_OUTPUT_ROOT"/{models,preds,metrics}

family_file_stem() {
  case "$1" in
    mosfet) printf "MOSFET" ;;
    igbt) printf "IGBT" ;;
    ic) printf "IC" ;;
    *)
      printf "Unknown family: %s\n" "$1" >&2
      exit 2
      ;;
  esac
}

family_seed_offset() {
  case "$1" in
    mosfet) printf "1" ;;
    igbt) printf "2" ;;
    ic) printf "3" ;;
  esac
}

ensure_base_family_raw() {
  local family="$1"
  local stem
  local out_file
  local seed
  stem="$(family_file_stem "$family")"
  out_file="$GENERATED_ROOT/$DATASET/raw/${stem}.csv"
  seed="$((BASE_COUNT + $(family_seed_offset "$family")))"

  if [[ -s "$out_file" && "${FORCE_BASE:-0}" != "1" ]]; then
    printf "base raw exists: %s\n" "$out_file"
    return 0
  fi

  mkdir -p "$(dirname "$out_file")"
  printf "generating base family=%s count=%s seed=%s -> %s\n" \
    "$family" "$BASE_COUNT" "$seed" "$out_file"
  time "$RUNNER" uv run python data/industrial/generate_sequences.py \
    --family "$family" \
    --count "$BASE_COUNT" \
    --seed "$seed" \
    --output "$out_file"
}

ensure_base_splits() {
  local splits_dir="$GENERATED_ROOT/$DATASET/splits"
  if [[ -s "$splits_dir/MOSFET_train.csv" && "${FORCE_SPLITS:-0}" != "1" ]]; then
    printf "base splits exist: %s\n" "$splits_dir"
    return 0
  fi

  ensure_base_family_raw mosfet
  ensure_base_family_raw igbt
  ensure_base_family_raw ic

  printf "creating base splits -> %s\n" "$splits_dir"
  time "$RUNNER" uv run python scripts/create_dataset_splits.py \
    --dataset "$DATASET" \
    --input-dir "$GENERATED_ROOT/$DATASET/raw" \
    --output-dir "$splits_dir" \
    --force
}

ensure_augmentation() {
  local out_csv="$GENERATED_ROOT/$AUGMENT_DATASET/raw.csv"
  if [[ -s "$out_csv" && "${FORCE_AUGMENT:-0}" != "1" ]]; then
    printf "augmentation data exists: %s\n" "$out_csv"
    return 0
  fi

  printf "generating augmentation count=%s -> %s\n" "$AUGMENT_COUNT" "$out_csv"
  time "$RUNNER" uv run python scripts/generate_augmented_sequences.py \
    --count "$AUGMENT_COUNT" \
    --dataset "$AUGMENT_DATASET" \
    --seed "${AUGMENT_SEED:-4250}"
}

ensure_base_splits
ensure_augmentation

ARGS=(
  --dataset "$FINAL_DATASET"
  --splits-dir "$GENERATED_ROOT/$DATASET/splits"
  --model-root "$SLURM_OUTPUT_ROOT/models"
  --preds-root "$SLURM_OUTPUT_ROOT/preds"
  --metrics-root "$SLURM_OUTPUT_ROOT/metrics"
  --holdout-family none
  --method-name "$METHOD_NAME"
  --augment-train-csv "$GENERATED_ROOT/$AUGMENT_DATASET/raw.csv"
  --augment-family-mode unknown
  --augment-limit "$SFT_AUGMENT_LIMIT"
  --tasks next_step
  --skip-fixed-eval
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --max-context "$MAX_CONTEXT"
  --k "$K"
  --phase-loss-weight "$PHASE_LOSS_WEIGHT"
  --family-dropout "$FAMILY_DROPOUT"
  --step-dropout "$STEP_DROPOUT"
  --max-train-batches "$MAX_TRAIN_BATCHES"
  --max-eval-batches "$MAX_EVAL_BATCHES"
  --d-model "$D_MODEL"
  --nhead "$NHEAD"
  --num-layers "$NUM_LAYERS"
  --dim-feedforward "$DIM_FEEDFORWARD"
  --dropout "$DROPOUT"
  --seed "$SEED"
  --device cuda
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "final submission train dataset=%s final_dataset=%s method=%s phase_loss=%s\n" \
  "$DATASET" "$FINAL_DATASET" "$METHOD_NAME" "$PHASE_LOSS_WEIGHT"

time "$RUNNER" uv run python -u -m zero_hack.models.gpt.train "${ARGS[@]}"

find "$SLURM_OUTPUT_ROOT/models/$FINAL_DATASET" -maxdepth 3 -type f | sort
