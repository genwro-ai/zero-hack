#!/bin/bash
#SBATCH --job-name=zero-hack-gpt-bare
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

# Bare GPT baseline over the unified holdout protocol.
# Each array task trains on one leave-family-out split, using 20k train records
# sampled from valid_s100k by default, then evaluates Tasks 1/2 on Industrial
# variant eval sets and Task 3 with threshold calibration.
#
# Submit:
#   sbatch slurm/train_gpt_bare.sbatch
#   sbatch --array=2 slurm/train_gpt_bare.sbatch   # only holdout=ic
#
# Main overrides:
#   DATASET=valid_s100k
#   TRAIN_SAMPLES=20000
#   EPOCHS=15
#   BATCH_SIZE=256
#   MAX_TRAIN_BATCHES=2000
#   MAX_EVAL_BATCHES=300
#   EXTRA_ARGS="--d-model 256 --num-layers 4"

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

HOLDOUTS=(mosfet igbt ic)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
HOLDOUT_FAMILY="${HOLDOUTS[$TASK_ID]}"

DATASET="${DATASET:-valid_s100k}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20000}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
SLURM_OUTPUT_ROOT="${SLURM_OUTPUT_ROOT:-outputs}"
METHOD_NAME="${METHOD_NAME:-gpt_bare}"
TASKS="${TASKS:-next_step completion anomaly}"
EVAL_VIEWS="${EVAL_VIEWS:-id ood}"
EVAL_FAMILY_MODE="${EVAL_FAMILY_MODE:-holdout_unknown}"

EPOCHS="${EPOCHS:-15}"
PATIENCE="${PATIENCE:-5}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"
MAX_CONTEXT="${MAX_CONTEXT:-192}"
K="${K:-5}"
MAX_COMPLETION_STEPS="${MAX_COMPLETION_STEPS:-240}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-2000}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-300}"

D_MODEL="${D_MODEL:-128}"
NHEAD="${NHEAD:-4}"
NUM_LAYERS="${NUM_LAYERS:-3}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-512}"
DROPOUT="${DROPOUT:-0.1}"
SEED="${SEED:-1729}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm "$SLURM_OUTPUT_ROOT"/{models,preds,metrics}

ARGS=(
  --dataset "$DATASET"
  --generated-root "$GENERATED_ROOT"
  --eval-root "$EVAL_ROOT"
  --model-root "$SLURM_OUTPUT_ROOT/models"
  --preds-root "$SLURM_OUTPUT_ROOT/preds"
  --metrics-root "$SLURM_OUTPUT_ROOT/metrics"
  --holdout-family "$HOLDOUT_FAMILY"
  --method-name "$METHOD_NAME"
  --tasks $TASKS
  --eval-views $EVAL_VIEWS
  --eval-family-mode "$EVAL_FAMILY_MODE"
  --train-samples "$TRAIN_SAMPLES"
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --max-context "$MAX_CONTEXT"
  --k "$K"
  --max-completion-steps "$MAX_COMPLETION_STEPS"
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

printf "dataset=%s holdout=%s method=%s train_samples=%s tasks=%s views=%s\n" \
  "$DATASET" "$HOLDOUT_FAMILY" "$METHOD_NAME" "$TRAIN_SAMPLES" "$TASKS" "$EVAL_VIEWS"

time "$RUNNER" uv run python -m zero_hack.models.gpt.train "${ARGS[@]}"
