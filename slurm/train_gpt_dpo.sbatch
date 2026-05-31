#!/bin/bash
#SBATCH --job-name=zero-hack-gpt-dpo
#SBATCH --array=0-2
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# DPO fine-tune the SFT GPT checkpoints produced by:
#   sbatch slurm/train_gpt_phase_augmented.sbatch
#
# Submit all holdouts:
#   sbatch slurm/train_gpt_dpo.sbatch
#
# Defaults run a longer DPO pass:
#   PAIRS=100000
#   EPOCHS=10
#   DPO_METHOD_NAME=gpt_dpo_10ep_100k
#
# If DPO pairs are missing, this script generates them for the current holdout.
# The default pair file includes PAIRS in the filename so larger runs do not
# accidentally reuse older smaller pair files.

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
AUGMENT_DATASET="${AUGMENT_DATASET:-augmented_s050k}"
RUN_DATASET="${RUN_DATASET:-${DATASET}_augmented_s050k}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
DPO_ROOT="${DPO_ROOT:-data/generated/dpo_pairs}"
SLURM_OUTPUT_ROOT="${SLURM_OUTPUT_ROOT:-outputs}"
RUNNER="${RUNNER:-srun}"

HOLDOUTS=(mosfet igbt ic)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
HOLDOUT_FAMILY="${HOLDOUTS[$TASK_ID]}"

SFT_METHOD_NAME="${SFT_METHOD_NAME:-gpt_phase_augmented}"
DPO_METHOD_NAME="${DPO_METHOD_NAME:-gpt_dpo_10ep_100k}"
CHECKPOINT="${CHECKPOINT:-$SLURM_OUTPUT_ROOT/models/$RUN_DATASET/${SFT_METHOD_NAME}_holdout_$HOLDOUT_FAMILY/best.pt}"
SPLITS_DIR="$GENERATED_ROOT/$DATASET/splits"
AUGMENT_CSV="$GENERATED_ROOT/$AUGMENT_DATASET/raw.csv"
PAIRS="${PAIRS:-100000}"
PAIRS_PATH="${PAIRS_PATH:-$DPO_ROOT/$RUN_DATASET/holdout_$HOLDOUT_FAMILY/pairs_${PAIRS}.jsonl}"
MAX_TRIES="${MAX_TRIES:-16}"
INVALID_WEIGHT="${INVALID_WEIGHT:-0.9}"
VALID_MISMATCH_WEIGHT="${VALID_MISMATCH_WEIGHT:-0.1}"
AUGMENT_LIMIT="${AUGMENT_LIMIT:-20000}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-6}"
BETA="${BETA:-0.1}"
SFT_WEIGHT="${SFT_WEIGHT:-0.3}"
FAMILY_DROPOUT="${FAMILY_DROPOUT:-0.15}"
STEP_DROPOUT="${STEP_DROPOUT:-0.05}"
K="${K:-5}"

mkdir -p outputs/slurm "$SLURM_OUTPUT_ROOT"/{models,preds,metrics} "$(dirname "$PAIRS_PATH")"

if [[ ! -s "$CHECKPOINT" ]]; then
  printf "missing SFT checkpoint: %s\n" "$CHECKPOINT" >&2
  printf "run sbatch slurm/train_gpt_phase_augmented.sbatch first\n" >&2
  exit 2
fi

if [[ ! -s "$PAIRS_PATH" || "${FORCE_DPO_PAIRS:-0}" == "1" ]]; then
  printf "generating missing DPO pairs holdout=%s -> %s\n" "$HOLDOUT_FAMILY" "$PAIRS_PATH"
  time "$RUNNER" uv run python scripts/generate_dpo_pairs.py \
    --splits-dir "$SPLITS_DIR" \
    --holdout-family "$HOLDOUT_FAMILY" \
    --augment-train-csv "$AUGMENT_CSV" \
    --augment-limit "$AUGMENT_LIMIT" \
    --augment-family-mode unknown \
    --pairs "$PAIRS" \
    --invalid-weight "$INVALID_WEIGHT" \
    --valid-mismatch-weight "$VALID_MISMATCH_WEIGHT" \
    --max-tries "$MAX_TRIES" \
    --out "$PAIRS_PATH" \
    --seed "${DPO_SEED:-1729}"
fi

ARGS=(
  --checkpoint "$CHECKPOINT"
  --pairs "$PAIRS_PATH"
  --dataset "$RUN_DATASET"
  --splits-dir "$SPLITS_DIR"
  --eval-root "$EVAL_ROOT"
  --model-root "$SLURM_OUTPUT_ROOT/models"
  --preds-root "$SLURM_OUTPUT_ROOT/preds"
  --metrics-root "$SLURM_OUTPUT_ROOT/metrics"
  --holdout-family "$HOLDOUT_FAMILY"
  --method-name "$DPO_METHOD_NAME"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --beta "$BETA"
  --sft-weight "$SFT_WEIGHT"
  --family-dropout "$FAMILY_DROPOUT"
  --step-dropout "$STEP_DROPOUT"
  --k "$K"
  --device cuda
)

if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi

if [[ -n "${PAIR_LIMIT:-}" ]]; then
  ARGS+=(--pair-limit "$PAIR_LIMIT")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "DPO training holdout=%s checkpoint=%s pairs=%s\n" \
  "$HOLDOUT_FAMILY" "$CHECKPOINT" "$PAIRS_PATH"

time "$RUNNER" uv run python -u -m zero_hack.models.gpt.dpo_train "${ARGS[@]}"

find "$SLURM_OUTPUT_ROOT/models/$RUN_DATASET" -maxdepth 2 -type f -name "best.pt" | sort
