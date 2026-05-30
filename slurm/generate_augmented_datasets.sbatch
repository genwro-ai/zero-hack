#!/bin/bash
#SBATCH --job-name=zero-hack-aug-data
#SBATCH --array=0-17%6
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

# Slurm array job for generating augmented Industrial AI datasets.
#
# It creates:
#   data/generated/augmented_s005k/raw/MOSFET.csv
#   data/generated/augmented_s010k/raw/MOSFET.csv
#   ...
#
# Submit with:
#   sbatch slurm/generate_augmented_datasets.sbatch
#
# Optional environment overrides:
#   OUTPUT_ROOT=data/generated
#   FORCE=1
#   NO_VALIDATE=1
#   AUGMENTED_EXTRA_ARGS="--litho-cycles 6 --second-metal-layer random"
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
FAMILIES=(mosfet igbt ic)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SIZE_INDEX="$((TASK_ID / ${#FAMILIES[@]}))"
FAMILY_INDEX="$((TASK_ID % ${#FAMILIES[@]}))"

COUNT="${SIZES[$SIZE_INDEX]}"
FAMILY="${FAMILIES[$FAMILY_INDEX]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/generated}"
FORCE="${FORCE:-0}"
NO_VALIDATE="${NO_VALIDATE:-0}"
RUNNER="${RUNNER:-srun}"

dataset_label() {
  local count="$1"
  printf "augmented_s%03dk" "$((count / 1000))"
}

family_seed_offset() {
  case "$1" in
    mosfet) printf "101" ;;
    igbt) printf "102" ;;
    ic) printf "103" ;;
    *)
      printf "Unknown family: %s\n" "$1" >&2
      exit 2
      ;;
  esac
}

DATASET="$(dataset_label "$COUNT")"
SEED="$((COUNT + $(family_seed_offset "$FAMILY")))"

ARGS=(
  --count "$COUNT"
  --dataset "$DATASET"
  --output-root "$OUTPUT_ROOT"
  --families "$FAMILY"
  --seed "$SEED"
)

if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

if [[ "$NO_VALIDATE" == "1" ]]; then
  ARGS+=(--no-validate)
fi

if [[ -n "${AUGMENTED_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($AUGMENTED_EXTRA_ARGS)
fi

mkdir -p outputs/slurm

printf "task_id=%s count=%s family=%s dataset=%s seed=%s output_root=%s\n" \
  "$TASK_ID" "$COUNT" "$FAMILY" "$DATASET" "$SEED" "$OUTPUT_ROOT"

time "$RUNNER" uv run python scripts/generate_augmented_industrial.py "${ARGS[@]}"

find "$OUTPUT_ROOT/$DATASET/raw" -maxdepth 1 -type f -name "*.csv" -exec ls -lh {} +
