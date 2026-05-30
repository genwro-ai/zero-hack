#!/bin/bash
#SBATCH --job-name=zero-hack-mix-split
#SBATCH --array=0-5
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Slurm array job for creating 40/60 vanilla/augmented mixed splits.
#
# It expects:
#   data/generated/valid_s005k/raw/{MOSFET,IGBT,IC}.csv
#   data/generated/augmented_s005k/raw/{MOSFET,IGBT,IC}.csv
#
# It writes:
#   data/generated/mixed_s005k_v40_a60/splits/train.csv
#   data/generated/mixed_s005k_v40_a60/splits/valid.csv
#   data/generated/mixed_s005k_v40_a60/splits/test_standard.csv
#   data/generated/mixed_s005k_v40_a60/splits/test_diverse.csv
#
# Submit with:
#   sbatch slurm/create_mixed_dataset_splits.sbatch
#
# Optional environment overrides:
#   GENERATED_ROOT=data/generated
#   VANILLA_RATIO=0.4
#   AUGMENTED_RATIO=0.6
#   FAMILIES="mosfet igbt ic"
#   FORCE=1
#   SPLIT_SEED=1729
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

SIZES=(5000 10000 20000 100000 500000 1000000)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
COUNT="${SIZES[$TASK_ID]}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
VANILLA_RATIO="${VANILLA_RATIO:-0.4}"
AUGMENTED_RATIO="${AUGMENTED_RATIO:-0.6}"
FAMILIES="${FAMILIES:-mosfet igbt ic}"
SPLIT_SEED="${SPLIT_SEED:-1729}"
FORCE="${FORCE:-0}"
RUNNER="${RUNNER:-srun}"

dataset_label() {
  local prefix="$1"
  local count="$2"
  printf "%s_s%03dk" "$prefix" "$((count / 1000))"
}

ratio_label() {
  python - "$1" <<'PY'
import sys
print(f"{round(float(sys.argv[1]) * 100):02d}")
PY
}

VANILLA_DATASET="$(dataset_label valid "$COUNT")"
AUGMENTED_DATASET="$(dataset_label augmented "$COUNT")"
DATASET="${DATASET:-mixed_s$(printf "%03dk" "$((COUNT / 1000))")_v$(ratio_label "$VANILLA_RATIO")_a$(ratio_label "$AUGMENTED_RATIO")}"

ARGS=(
  --count "$COUNT"
  --dataset "$DATASET"
  --generated-root "$GENERATED_ROOT"
  --vanilla-dataset "$VANILLA_DATASET"
  --augmented-dataset "$AUGMENTED_DATASET"
  --vanilla-ratio "$VANILLA_RATIO"
  --augmented-ratio "$AUGMENTED_RATIO"
  --families $FAMILIES
  --seed "$SPLIT_SEED"
)

if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

mkdir -p outputs/slurm

printf "task_id=%s count=%s dataset=%s vanilla=%s augmented=%s generated_root=%s\n" \
  "$TASK_ID" "$COUNT" "$DATASET" "$VANILLA_DATASET" "$AUGMENTED_DATASET" "$GENERATED_ROOT"

time "$RUNNER" uv run python scripts/create_mixed_dataset_splits.py "${ARGS[@]}"

find "$GENERATED_ROOT/$DATASET/splits" -maxdepth 1 -type f -name "*.csv" -exec ls -lh {} +
