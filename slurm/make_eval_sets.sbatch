#!/bin/bash
#SBATCH --job-name=zero-hack-evalsets
#SBATCH --array=0-5
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Slurm array job for creating local eval sets under data/eval.
#
# For each mixed dataset size, this creates standard and diverse eval suites:
#   data/eval/mixed_s005k_v40_a60/holdout_mosfet/standard/id
#   data/eval/mixed_s005k_v40_a60/holdout_mosfet/standard/ood
#   data/eval/mixed_s005k_v40_a60/holdout_mosfet/diverse/id
#   data/eval/mixed_s005k_v40_a60/holdout_mosfet/diverse/ood
#
# Each holdout directory uses the corresponding two families as train families
# in metadata. The id view evaluates those two families; the ood view evaluates
# the holdout family.
#
# Submit with:
#   sbatch slurm/make_eval_sets.sbatch
#
# Optional environment overrides:
#   GENERATED_ROOT=data/generated
#   EVAL_ROOT=data/eval
#   DATASETS="mixed_s005k_v40_a60 mixed_s010k_v40_a60 ..."
#   EVAL_SOURCES="test_standard test_diverse"   # use EVAL_SOURCES=test for legacy layout
#   HOLDOUT_FAMILIES="mosfet igbt ic"
#   N_VALID=100
#   N_ANOMALY_VALID=200
#   N_ANOMALY_INVALID=129
#   EVAL_SEED=1729
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

DEFAULT_DATASETS="mixed_s005k_v40_a60 mixed_s010k_v40_a60 mixed_s020k_v40_a60 mixed_s100k_v40_a60 mixed_s500k_v40_a60 mixed_s1000k_v40_a60"
# shellcheck disable=SC2206
DATASET_LIST=(${DATASETS:-$DEFAULT_DATASETS})

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASET_LIST[$TASK_ID]}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
EVAL_SOURCES="${EVAL_SOURCES:-test_standard test_diverse}"
HOLDOUT_FAMILIES="${HOLDOUT_FAMILIES:-mosfet igbt ic}"
N_VALID="${N_VALID:-100}"
N_ANOMALY_VALID="${N_ANOMALY_VALID:-200}"
N_ANOMALY_INVALID="${N_ANOMALY_INVALID:-129}"
EVAL_SEED="${EVAL_SEED:-1729}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm "$EVAL_ROOT"

printf "task_id=%s dataset=%s generated_root=%s eval_root=%s sources=%s holdouts=%s\n" \
  "$TASK_ID" "$DATASET" "$GENERATED_ROOT" "$EVAL_ROOT" "$EVAL_SOURCES" "$HOLDOUT_FAMILIES"

time "$RUNNER" uv run python scripts/make_all_eval_sets.py \
  --generated-root "$GENERATED_ROOT" \
  --datasets "$DATASET" \
  --out-root "$EVAL_ROOT" \
  --eval-sources $EVAL_SOURCES \
  --holdout-families $HOLDOUT_FAMILIES \
  --n-valid "$N_VALID" \
  --n-anomaly-valid "$N_ANOMALY_VALID" \
  --n-anomaly-invalid "$N_ANOMALY_INVALID" \
  --seed "$EVAL_SEED"

find "$EVAL_ROOT/$DATASET" -maxdepth 2 -type f -name "*.csv" -exec ls -lh {} +
