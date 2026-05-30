#!/bin/bash
#SBATCH --job-name=zero-hack-lstm
#SBATCH --array=0-1
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err
# Array job: END emails arrive per task; change to FAIL to reduce volume.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# LSTM-only training array. Two runs go to DISTINCT checkpoint dirs so neither
# overwrites the other and they can be compared on the completion eval:
#
#   task 0 -> teacher forcing      -> outputs/models/<dataset>/lstm_teacher_forcing_holdout_<family>/best.pt
#   task 1 -> scheduled sampling   -> outputs/models/<dataset>/lstm_scheduled_sampling_holdout_<family>/best.pt
#
# Submit both:
#   sbatch slurm/train_lstm.sbatch
# Submit only the scheduled-sampling run:
#   sbatch --array=1 slurm/train_lstm.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s005k          # which generated dataset's splits to train on
#   SPLITS_ROOT=data/generated   # where the dataset splits/ dirs live
#   EVAL_ROOT=data/eval           # fixed Industrial eval sets
#   MODEL_ROOT=outputs/models     # where checkpoints are written
#   PREDS_ROOT=outputs/preds      # where method predictions are written
#   METRICS_ROOT=outputs/metrics  # where next-step reports are written
#   EPOCHS=10
#   BATCH_SIZE=128
#   LR=3e-3
#   MAX_CONTEXT=192
#   K=3                           # top-k for the next-step report
#   SS_MAX_PROB=0.25              # scheduled-sampling max own-prediction prob
#   SS_SCHEDULE=linear            # linear | constant
#   LIMIT_PER_FAMILY=             # cap records per family (smoke runs)
#   HOLDOUT_FAMILY=ic             # mosfet | igbt | ic (leave-one-family-out)
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

RUNS=(teacher_forcing scheduled_sampling)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN="${RUNS[$TASK_ID]}"

DATASET="${DATASET:-valid_s005k}"
SPLITS_ROOT="${SPLITS_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
MODEL_ROOT="${MODEL_ROOT:-outputs/models}"
PREDS_ROOT="${PREDS_ROOT:-outputs/preds}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-3e-3}"
MAX_CONTEXT="${MAX_CONTEXT:-192}"
K="${K:-3}"
SS_MAX_PROB="${SS_MAX_PROB:-0.25}"
SS_SCHEDULE="${SS_SCHEDULE:-linear}"
RUNNER="${RUNNER:-srun}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"

SPLITS_DIR="${SPLITS_ROOT}/${DATASET}/splits"
RUN_NAME="lstm_${RUN}"
MODEL_RUN_NAME="${RUN_NAME}_holdout_${HOLDOUT_FAMILY}"
CKPT="${MODEL_ROOT}/${DATASET}/${MODEL_RUN_NAME}/best.pt"
REPORT_DIR="${METRICS_ROOT}/${DATASET}/${MODEL_RUN_NAME}"

mkdir -p outputs/slurm "$(dirname "$CKPT")" "$REPORT_DIR"

ARGS=(
  --splits-dir "$SPLITS_DIR"
  --dataset "$DATASET"
  --eval-root "$EVAL_ROOT"
  --preds-root "$PREDS_ROOT"
  --metrics-root "$METRICS_ROOT"
  --holdout-family "$HOLDOUT_FAMILY"
  --method-name "$RUN_NAME"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --max-context "$MAX_CONTEXT"
  --k "$K"
  --checkpoint-path "$CKPT"
  --report-dir "$REPORT_DIR"
)

if [[ "$RUN" == "scheduled_sampling" ]]; then
  ARGS+=(--scheduled-sampling --ss-max-prob "$SS_MAX_PROB" --ss-schedule "$SS_SCHEDULE")
fi

if [[ -n "${LIMIT_PER_FAMILY:-}" ]]; then
  ARGS+=(--limit-per-family "$LIMIT_PER_FAMILY")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "task_id=%s run=%s dataset=%s holdout=%s splits=%s ckpt=%s epochs=%s\n" \
  "$TASK_ID" "$RUN" "$DATASET" "$HOLDOUT_FAMILY" "$SPLITS_DIR" "$CKPT" "$EPOCHS"

time "$RUNNER" uv run python -m zero_hack.models.lstm.train "${ARGS[@]}"

ls -lh "$(dirname "$CKPT")"
