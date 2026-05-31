#!/bin/bash
#SBATCH --job-name=zero-hack-gpt-phase
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# End-to-end GPT run for the Industrial AI holdout protocol:
#   1. Ensure the 100k-per-family base dataset exists.
#   2. Generate 50k extra valid "unseen" sequences with scripts/generate_unseen_data.py.
#   3. Build 1000-row eval views for next-step/completion and anomaly for every holdout.
#   4. Train one GPT per holdout family with next-token + next-phase auxiliary loss.
#      Slurm array tasks run the three holdouts in parallel, one GPU each.
#
# Submit:
#   sbatch slurm/train_gpt_phase_augmented.sbatch
#   sbatch --array=2 slurm/train_gpt_phase_augmented.sbatch   # only holdout=ic
#
# Main overrides:
#   DATASET=valid_s100k
#   BASE_COUNT=100000
#   UNSEEN_COUNT=50000
#   RUN_DATASET=valid_s100k_unseen_s050k
#   EPOCHS=40
#   BATCH_SIZE=256
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
UNSEEN_COUNT="${UNSEEN_COUNT:-50000}"
UNSEEN_DATASET="${UNSEEN_DATASET:-unseen_s050k}"
RUN_DATASET="${RUN_DATASET:-${DATASET}_unseen_s050k}"

GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
SLURM_OUTPUT_ROOT="${SLURM_OUTPUT_ROOT:-outputs/slurm_runs}"
RUNNER="${RUNNER:-srun}"

HOLDOUT_FAMILIES="${HOLDOUT_FAMILIES:-mosfet igbt ic}"
HOLDOUTS=(mosfet igbt ic)
TASK_ID="${SLURM_ARRAY_TASK_ID:-}"
if [[ -n "$TASK_ID" ]]; then
  HOLDOUT_FAMILY="${HOLDOUTS[$TASK_ID]}"
else
  HOLDOUT_FAMILY=""
fi
FRACTIONS="${FRACTIONS:-0.6 0.8}"
EVAL_VALID_TOTAL="${EVAL_VALID_TOTAL:-1000}"
EVAL_ANOMALY_TOTAL="${EVAL_ANOMALY_TOTAL:-1000}"
EVAL_SEED="${EVAL_SEED:-1729}"

EPOCHS="${EPOCHS:-40}"
PATIENCE="${PATIENCE:-6}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-2e-4}"
MAX_CONTEXT="${MAX_CONTEXT:-256}"
K="${K:-5}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-2000}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-300}"
SFT_AUGMENT_LIMIT="${SFT_AUGMENT_LIMIT:-20000}"
PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.0}"
FAMILY_DROPOUT="${FAMILY_DROPOUT:-0.25}"
STEP_DROPOUT="${STEP_DROPOUT:-0.05}"
METHOD_NAME="${METHOD_NAME:-gpt_phase_augmented}"

D_MODEL="${D_MODEL:-256}"
NHEAD="${NHEAD:-8}"
NUM_LAYERS="${NUM_LAYERS:-4}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1024}"
DROPOUT="${DROPOUT:-0.1}"

mkdir -p outputs/slurm "$SLURM_OUTPUT_ROOT"/{models,preds,metrics} "$EVAL_ROOT"

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

ensure_unseen() {
  local out_csv="$GENERATED_ROOT/$UNSEEN_DATASET/raw.csv"
  if [[ -s "$out_csv" && "${FORCE_UNSEEN:-0}" != "1" ]]; then
    printf "unseen augmentation exists: %s\n" "$out_csv"
    return 0
  fi

  printf "generating unseen augmentation count=%s -> %s\n" "$UNSEEN_COUNT" "$out_csv"
  time "$RUNNER" uv run python scripts/generate_unseen_data.py \
    --count "$UNSEEN_COUNT" \
    --dataset "$UNSEEN_DATASET" \
    --seed "${UNSEEN_SEED:-4250}"
}

wait_for_file() {
  local path="$1"
  local label="$2"
  local waited=0
  local timeout="${WAIT_FOR_SHARED_DATA_SECS:-7200}"
  while [[ ! -s "$path" ]]; do
    if (( waited >= timeout )); then
      printf "timed out waiting for %s: %s\n" "$label" "$path" >&2
      exit 2
    fi
    printf "waiting for %s: %s\n" "$label" "$path"
    sleep 30
    waited="$((waited + 30))"
  done
}

ensure_or_wait_shared_data() {
  local splits_done="$GENERATED_ROOT/$DATASET/splits/MOSFET_train.csv"
  local unseen_done="$GENERATED_ROOT/$UNSEEN_DATASET/raw.csv"

  if [[ -z "$TASK_ID" || "$TASK_ID" == "0" ]]; then
    ensure_base_splits
    ensure_unseen
    return 0
  fi

  wait_for_file "$splits_done" "base splits from array task 0"
  wait_for_file "$unseen_done" "unseen augmentation from array task 0"
}

count_words() {
  local count=0
  for _item in $*; do
    count="$((count + 1))"
  done
  printf "%s" "$count"
}

make_eval_view() {
  local holdout="$1"
  local view="$2"
  shift 2
  local families=("$@")
  local family_count="${#families[@]}"
  local fraction_count
  local n_valid
  local n_anomaly_valid
  local n_anomaly_invalid

  fraction_count="$(count_words $FRACTIONS)"
  n_valid="$((EVAL_VALID_TOTAL / (family_count * fraction_count)))"
  n_anomaly_valid="$((EVAL_ANOMALY_TOTAL / (family_count * 2)))"
  n_anomaly_invalid="$n_anomaly_valid"

  printf "making eval holdout=%s view=%s families=%s n_valid_per_family=%s anomaly=%s+%s\n" \
    "$holdout" "$view" "${families[*]}" "$n_valid" "$n_anomaly_valid" "$n_anomaly_invalid"

  time "$RUNNER" uv run python scripts/make_eval_set.py \
    --splits-dir "$GENERATED_ROOT/$DATASET/splits" \
    --out-dir "$EVAL_ROOT/$RUN_DATASET/holdout_$holdout/$view" \
    --holdout-family "$holdout" \
    --eval-families "${families[@]}" \
    --n-valid "$n_valid" \
    --n-anomaly-valid "$n_anomaly_valid" \
    --n-anomaly-invalid "$n_anomaly_invalid" \
    --seed "$EVAL_SEED" \
    --fractions $FRACTIONS
}

make_eval_for_holdout() {
  local holdout="$1"
  local id_families=()
  local family
  for family in mosfet igbt ic; do
    if [[ "$family" != "$holdout" ]]; then
      id_families+=("$family")
    fi
  done

  make_eval_view "$holdout" id "${id_families[@]}"
  make_eval_view "$holdout" ood "$holdout"
}

train_holdout() {
  local holdout="$1"
  local args=(
    --dataset "$RUN_DATASET"
    --splits-dir "$GENERATED_ROOT/$DATASET/splits"
    --eval-root "$EVAL_ROOT"
    --model-root "$SLURM_OUTPUT_ROOT/models"
    --preds-root "$SLURM_OUTPUT_ROOT/preds"
    --metrics-root "$SLURM_OUTPUT_ROOT/metrics"
    --holdout-family "$holdout"
    --method-name "$METHOD_NAME"
    --augment-train-csv "$GENERATED_ROOT/$UNSEEN_DATASET/raw.csv"
    --augment-family-mode unknown
    --augment-limit "$SFT_AUGMENT_LIMIT"
    --tasks next_step completion anomaly
    --eval-views id ood
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --max-context "$MAX_CONTEXT"
    --k "$K"
    --phase-loss-weight "$PHASE_LOSS_WEIGHT"
    --family-dropout "$FAMILY_DROPOUT"
    --step-dropout "$STEP_DROPOUT"
    --device cuda
    --d-model "$D_MODEL"
    --nhead "$NHEAD"
    --num-layers "$NUM_LAYERS"
    --dim-feedforward "$DIM_FEEDFORWARD"
    --dropout "$DROPOUT"
  )

  args+=(--max-train-batches "$MAX_TRAIN_BATCHES")
  args+=(--max-eval-batches "$MAX_EVAL_BATCHES")

  if [[ -n "${EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    args+=($EXTRA_ARGS)
  fi

  printf "training holdout=%s run_dataset=%s method=%s\n" \
    "$holdout" "$RUN_DATASET" "$METHOD_NAME"
  time "$RUNNER" uv run python -u -m zero_hack.models.gpt.train "${args[@]}"
}

printf "dataset=%s run_dataset=%s unseen=%s holdouts=%s\n" \
  "$DATASET" "$RUN_DATASET" "$UNSEEN_DATASET" "$HOLDOUT_FAMILIES"

ensure_or_wait_shared_data

if [[ -n "$HOLDOUT_FAMILY" ]]; then
  make_eval_for_holdout "$HOLDOUT_FAMILY"
  train_holdout "$HOLDOUT_FAMILY"
else
  for holdout in $HOLDOUT_FAMILIES; do
    make_eval_for_holdout "$holdout"
    train_holdout "$holdout"
  done
fi

find "$SLURM_OUTPUT_ROOT" -maxdepth 6 -type f | sort
