#!/bin/bash
#SBATCH --job-name=zero-hack-classic
#SBATCH --array=0-2
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Slurm array job for n-gram / VLMC hyperparameter search.
# Each task evaluates one holdout family, using 20k train records sampled from
# the valid_s100k generated training split by default.
#
# Submit:
#   sbatch slurm/eval_classic_search.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s100k
#   TRAIN_SAMPLES=20000
#   MODELS="ngram vlmc"
#   N_VALUES="3 5 7"
#   ALPHA_VALUES="0.2 0.4 0.8"
#   VIEWS="id ood"
#   TASKS="next_step completion anomaly"
#   RANK_BY=next_step_mrr
#   UV_CACHE_DIR=$SCRATCH/.cache/uv
#   UV_INSTALL_DIR=$SCRATCH/.local/bin
#   LEONARDO_LOAD_MODULES=0

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
MODELS="${MODELS:-ngram vlmc}"
N_VALUES="${N_VALUES:-3 5 7}"
ALPHA_VALUES="${ALPHA_VALUES:-0.2 0.4 0.8}"
VIEWS="${VIEWS:-id ood}"
TASKS="${TASKS:-next_step completion anomaly}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
PREDS_ROOT="${PREDS_ROOT:-outputs/preds}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
SEED="${SEED:-1729}"
RANK_BY="${RANK_BY:-next_step_mrr}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm "$PREDS_ROOT" "$METRICS_ROOT"

ARGS=(
  --generated-root "$GENERATED_ROOT"
  --eval-root "$EVAL_ROOT"
  --preds-root "$PREDS_ROOT"
  --metrics-root "$METRICS_ROOT"
  --datasets "$DATASET"
  --holdout-families "$HOLDOUT_FAMILY"
  --models $MODELS
  --views $VIEWS
  --tasks $TASKS
  --train-samples "$TRAIN_SAMPLES"
  --n-values $N_VALUES
  --alpha-values $ALPHA_VALUES
  --rank-by "$RANK_BY"
  --seed "$SEED"
)

printf "dataset=%s holdout=%s train_samples=%s models=%s n_values=%s alpha_values=%s\n" \
  "$DATASET" "$HOLDOUT_FAMILY" "$TRAIN_SAMPLES" "$MODELS" "$N_VALUES" "$ALPHA_VALUES"

time "$RUNNER" uv run python scripts/run_holdout_experiments.py "${ARGS[@]}"
