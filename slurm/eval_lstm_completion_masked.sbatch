#!/bin/bash
#SBATCH --job-name=zero-hack-lstm-completion-masked
#SBATCH --array=0-1
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err
# Array job: END emails arrive per task; change to FAIL to reduce volume.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Free-running sequence-completion (Task 2) eval WITH the ViolationMask applied
# (--enforce-rules): rule-violating next steps are masked to -inf during
# generation, so completions stay process-valid by construction.
#
# This mirrors slurm/eval_lstm_completion.sbatch exactly but writes to a
# separate `completion_masked` dir, so you get a clean A/B against the unmasked
# baseline. Quick promise check: compare generated_route_validity_rate and the
# completion metrics between the two output dirs.
#
#   task 0 -> lstm_teacher_forcing
#   task 1 -> lstm_scheduled_sampling
#
# Submit both:
#   sbatch slurm/eval_lstm_completion_masked.sbatch
# Submit only the scheduled-sampling eval:
#   sbatch --array=1 slurm/eval_lstm_completion_masked.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s005k
#   SPLITS_ROOT=data/generated
#   MODEL_ROOT=outputs/models
#   METRICS_ROOT=outputs/metrics
#   FRACTIONS="0.6 0.8"          # prefix fractions to keep
#   MAX_EXAMPLES=100             # sequences per family per fraction (0 = all)
#   MAX_STEPS=400                # generation length cap
#   EXTRA_ARGS="--device cuda"   # appended verbatim to the eval CLI

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
MODEL_ROOT="${MODEL_ROOT:-outputs/models}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
FRACTIONS="${FRACTIONS:-0.6 0.8}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
MAX_STEPS="${MAX_STEPS:-400}"
RUNNER="${RUNNER:-srun}"

RUN_NAME="lstm_${RUN}"
SPLITS_DIR="${SPLITS_ROOT}/${DATASET}/splits"
CKPT="${MODEL_ROOT}/${DATASET}/${RUN_NAME}/best.pt"
OUT_DIR="${METRICS_ROOT}/${DATASET}/${RUN_NAME}/completion_masked"

mkdir -p outputs/slurm "$OUT_DIR"

if [[ ! -f "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT (run slurm/train_lstm.sbatch first)" >&2
  exit 1
fi

ARGS=(
  --checkpoint "$CKPT"
  --splits-dir "$SPLITS_DIR"
  --fractions $FRACTIONS
  --max-examples-per-family "$MAX_EXAMPLES"
  --max-steps "$MAX_STEPS"
  --enforce-rules
  --out "$OUT_DIR"
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "task_id=%s run=%s dataset=%s ckpt=%s out=%s (enforce_rules=1)\n" \
  "$TASK_ID" "$RUN" "$DATASET" "$CKPT" "$OUT_DIR"

time "$RUNNER" uv run python scripts/eval_lstm_completion.py "${ARGS[@]}"

ls -lh "$OUT_DIR"
