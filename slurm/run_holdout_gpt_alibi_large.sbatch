#!/bin/bash
#SBATCH --job-name=zero-hack-holdout-gpt-large
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Holdout experiments for a LARGER GPT-ALiBi decoder.
#
# Same pipeline as slurm/run_holdout_gpt_alibi.sbatch (scripts/run_holdout_experiments.py
# --models gpt), but with a much bigger GPTNextStepModel to test whether the
# baseline model was simply under-capacity. nhead is fixed at 8 in the runner and
# feed-forward is 4*d_model, so d_model must stay divisible by 8.
#
#   default here -> d_model=512  layers=12  batch=48  lr=3e-4  (~50M params)
#   baseline     -> d_model=256  layers=4   batch=128 lr=6e-4  (~3M params)
#
# Predictions land under outputs/preds/... and metrics under outputs/metrics/...,
# tagged "gpt" — same as the base script. Use a separate PREDS_ROOT/METRICS_ROOT
# (or copy results aside) if you want to keep both runs' numbers side by side.
#
# Submit:
#   DATASETS=valid_s010k sbatch slurm/run_holdout_gpt_alibi_large.sbatch
# Single family:
#   DATASETS=valid_s010k HOLDOUT_FAMILIES=ic sbatch slurm/run_holdout_gpt_alibi_large.sbatch
# Even bigger (xl, ~85M params) via overrides:
#   GPT_D_MODEL=768 GPT_LAYERS=12 GPT_BATCH_SIZE=32 GPT_LR=2e-4 \
#     DATASETS=valid_s010k sbatch slurm/run_holdout_gpt_alibi_large.sbatch
#
# Optional environment overrides:
#   DATASETS="valid_s010k"               # default: all discovered datasets
#   HOLDOUT_FAMILIES="mosfet igbt ic"    # default: all families
#   VIEWS="id ood"                       # default: id ood
#   TASKS="next_step completion anomaly" # default: all tasks
#   GENERATED_ROOT=data/generated
#   EVAL_ROOT=data/eval
#   PREDS_ROOT=outputs/preds
#   METRICS_ROOT=outputs/metrics
#   GPT_EPOCHS=40
#   GPT_PATIENCE=5
#   GPT_D_MODEL=512
#   GPT_LAYERS=12
#   GPT_BATCH_SIZE=48
#   GPT_LR=3e-4
#   GPT_MAX_CONTEXT=256
#   PHASE_LOSS_WEIGHT=0.0
#   TRAIN_SIZES="100 250 500"            # scaling sweep (per family); default: full split
#   ANOMALY_METHOD=likelihood
#   SAVE_CHECKPOINTS=1                   # pass --save-checkpoints (recommended for big runs)
#   SEED=1729
#   NUM_WORKERS=8
#   EXTRA_ARGS="..."                     # appended verbatim to the CLI
#   UV_CACHE_DIR=$SCRATCH/.cache/uv
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

GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
PREDS_ROOT="${PREDS_ROOT:-outputs/preds}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1729}"
ANOMALY_METHOD="${ANOMALY_METHOD:-likelihood}"
# Larger model defaults (vs the base script's 256/4/128/6e-4).
GPT_EPOCHS="${GPT_EPOCHS:-40}"
GPT_PATIENCE="${GPT_PATIENCE:-5}"
GPT_D_MODEL="${GPT_D_MODEL:-512}"
GPT_LAYERS="${GPT_LAYERS:-12}"
GPT_BATCH_SIZE="${GPT_BATCH_SIZE:-48}"
GPT_LR="${GPT_LR:-3e-4}"
GPT_MAX_CONTEXT="${GPT_MAX_CONTEXT:-256}"
PHASE_LOSS_WEIGHT="${PHASE_LOSS_WEIGHT:-0.0}"
RUNNER="${RUNNER:-srun}"

if (( GPT_D_MODEL % 8 != 0 )); then
  echo "GPT_D_MODEL ($GPT_D_MODEL) must be divisible by nhead=8." >&2
  exit 1
fi

mkdir -p outputs/slurm

ARGS=(
  --models gpt
  --generated-root "$GENERATED_ROOT"
  --eval-root "$EVAL_ROOT"
  --preds-root "$PREDS_ROOT"
  --metrics-root "$METRICS_ROOT"
  --anomaly-method "$ANOMALY_METHOD"
  --gpt-epochs "$GPT_EPOCHS"
  --gpt-patience "$GPT_PATIENCE"
  --gpt-d-model "$GPT_D_MODEL"
  --gpt-layers "$GPT_LAYERS"
  --gpt-batch-size "$GPT_BATCH_SIZE"
  --gpt-lr "$GPT_LR"
  --gpt-max-context "$GPT_MAX_CONTEXT"
  --phase-loss-weight "$PHASE_LOSS_WEIGHT"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
)

if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(--datasets ${DATASETS})
fi
if [[ -n "${HOLDOUT_FAMILIES:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(--holdout-families ${HOLDOUT_FAMILIES})
fi
if [[ -n "${VIEWS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(--views ${VIEWS})
fi
if [[ -n "${TASKS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(--tasks ${TASKS})
fi
if [[ -n "${TRAIN_SIZES:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=(--train-sizes ${TRAIN_SIZES})
fi
if [[ "${SAVE_CHECKPOINTS:-0}" == "1" ]]; then
  ARGS+=(--save-checkpoints)
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "models=gpt(LARGE) datasets=%s holdout=%s views=%s tasks=%s epochs=%s d_model=%s layers=%s batch=%s lr=%s\n" \
  "${DATASETS:-<all>}" "${HOLDOUT_FAMILIES:-<all>}" "${VIEWS:-id ood}" \
  "${TASKS:-<all>}" "$GPT_EPOCHS" "$GPT_D_MODEL" "$GPT_LAYERS" "$GPT_BATCH_SIZE" "$GPT_LR"

# setup_uv.sh already provisioned the venv with `uv sync --no-dev`; --no-sync
# reuses it as-is so `uv run` does not re-sync the dev group (pytest, pygments,
# ...) which is uncached on the offline compute nodes.
time "$RUNNER" uv run --no-sync python scripts/run_holdout_experiments.py "${ARGS[@]}"
