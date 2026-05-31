#!/bin/bash
#SBATCH --job-name=zero-hack-submit
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=outputs/slurm/%x_%A.out
#SBATCH --error=outputs/slurm/%x_%A.err

set -euo pipefail

# Generate final organizer submission files:
#   nextstep.csv
#   completion.csv
#   anomaly.csv
#
# Default checkpoint:
#   outputs/models/final_submission/final_submission_gpt_all_families/best.pt

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

SLURM_OUTPUT_ROOT="${SLURM_OUTPUT_ROOT:-outputs}"
FINAL_DATASET="${FINAL_DATASET:-final_submission}"
METHOD_NAME="${METHOD_NAME:-final_submission_gpt}"
CHECKPOINT="${CHECKPOINT:-$SLURM_OUTPUT_ROOT/models/$FINAL_DATASET/${METHOD_NAME}_all_families/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$SLURM_OUTPUT_ROOT/submission/$METHOD_NAME}"
EVAL_INPUT_VALID="${EVAL_INPUT_VALID:-data/industrial/eval_input_valid.csv}"
EVAL_INPUT_ANOMALY="${EVAL_INPUT_ANOMALY:-data/industrial/eval_input_anomaly.csv}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
CALIBRATION_DATASET="${CALIBRATION_DATASET:-valid_s100k}"
K="${K:-5}"
MAX_COMPLETION_STEPS="${MAX_COMPLETION_STEPS:-240}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-64}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm "$OUTPUT_DIR"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --eval-input-valid "$EVAL_INPUT_VALID"
  --eval-input-anomaly "$EVAL_INPUT_ANOMALY"
  --output-dir "$OUTPUT_DIR"
  --eval-root "$EVAL_ROOT"
  --calibration-dataset "$CALIBRATION_DATASET"
  --k "$K"
  --max-completion-steps "$MAX_COMPLETION_STEPS"
  --score-batch-size "$SCORE_BATCH_SIZE"
  --device cuda
)

if [[ -n "${ANOMALY_THRESHOLD:-}" ]]; then
  ARGS+=(--anomaly-threshold "$ANOMALY_THRESHOLD")
fi

if [[ -n "${CALIBRATION_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  dirs=($CALIBRATION_DIRS)
  ARGS+=(--calibration-dirs "${dirs[@]}")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "predict final submission checkpoint=%s output_dir=%s\n" "$CHECKPOINT" "$OUTPUT_DIR"

time "$RUNNER" uv run python -u scripts/predict_gpt_submission.py "${ARGS[@]}"

ls -lh "$OUTPUT_DIR"/nextstep.csv "$OUTPUT_DIR"/completion.csv "$OUTPUT_DIR"/anomaly.csv
