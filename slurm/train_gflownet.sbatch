#!/bin/bash
#SBATCH --job-name=zero-hack-gflownet
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail

# Prefix-conditioned GFlowNet process-flow model.
#
# Default run:
#   - dataset: data/generated/valid_s010k/splits
#   - holdout family: ic
#   - train cap: 5k/family, so about 10k train sequences with one family held out
#   - epochs: 200
#
# Submit:
#   sbatch slurm/train_gflownet.sbatch
#
# Common overrides:
#   DATASET=valid_s100k
#   HOLDOUT_FAMILY=mosfet
#   LIMIT_PER_FAMILY=5000
#   EPOCHS=200
#   BATCH_SIZE=128
#   GFN_BATCH_SIZE=16
#   LR=3e-4
#   COMPLETION_SAMPLES=4
#   SLURM_OUTPUT_ROOT=outputs/slurm_runs
#   EXTRA_ARGS="--d-model 256 --num-layers 4 --gfn-weight 0.2"

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

DATASET="${DATASET:-valid_s010k}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
SLURM_OUTPUT_ROOT="${SLURM_OUTPUT_ROOT:-outputs/slurm_runs}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GFN_BATCH_SIZE="${GFN_BATCH_SIZE:-16}"
COMPLETION_SAMPLES="${COMPLETION_SAMPLES:-4}"
LR="${LR:-3e-4}"
MAX_CONTEXT="${MAX_CONTEXT:-192}"
LIMIT_PER_FAMILY="${LIMIT_PER_FAMILY:-5000}"
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
  --limit-per-family "$LIMIT_PER_FAMILY"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --gfn-batch-size "$GFN_BATCH_SIZE"
  --completion-samples "$COMPLETION_SAMPLES"
  --lr "$LR"
  --max-context "$MAX_CONTEXT"
  --device cuda
)

if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi

if [[ -n "${MAX_EVAL_BATCHES:-}" ]]; then
  ARGS+=(--max-eval-batches "$MAX_EVAL_BATCHES")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "model=gflownet dataset=%s holdout=%s limit_per_family=%s epochs=%s output_root=%s\n" \
  "$DATASET" "$HOLDOUT_FAMILY" "$LIMIT_PER_FAMILY" "$EPOCHS" "$SLURM_OUTPUT_ROOT"

time "$RUNNER" uv run python -m zero_hack.models.gflownet.train "${ARGS[@]}"

find "$SLURM_OUTPUT_ROOT" -maxdepth 6 -type f | sort
