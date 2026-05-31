#!/bin/bash
#SBATCH --job-name=zero-hack-data
#SBATCH --array=0-8%6
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=120GB # Fair share on Leonardo: 120GB * gpus-per-task
#SBATCH --cpus-per-task=8 # Fair share on Leonardo: 8 * gpus-per-task
#SBATCH --time=0:30:00 # Time limit in HH:MM:SS, up to 24:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Slurm array job for generating valid Industrial AI datasets.
#
# It creates the same layout as local_scripts/generate_valid_datasets.sh:
#   data/generated/valid_s005k/raw/MOSFET.csv
#   data/generated/valid_s020k/raw/MOSFET.csv
#   ...
#
# Submit with:
#   sbatch slurm/generate_valid_datasets.sbatch
#
# Optional environment overrides:
#   OUTPUT_ROOT=data/generated
#   FORCE=1
#   VALIDATE_AFTER=1
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

SIZES=(5000 20000 100000)
FAMILIES=(mosfet igbt ic)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SIZE_INDEX="$((TASK_ID / ${#FAMILIES[@]}))"
FAMILY_INDEX="$((TASK_ID % ${#FAMILIES[@]}))"

COUNT="${SIZES[$SIZE_INDEX]}"
FAMILY="${FAMILIES[$FAMILY_INDEX]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/generated}"
GENERATOR="${GENERATOR:-data/industrial/generate_sequences.py}"
FORCE="${FORCE:-0}"
VALIDATE_AFTER="${VALIDATE_AFTER:-0}"
RUNNER="${RUNNER:-srun}"

dataset_label() {
  local count="$1"
  printf "valid_s%03dk" "$((count / 1000))"
}

family_file_name() {
  case "$1" in
    mosfet) printf "MOSFET" ;;
    igbt) printf "IGBT" ;;
    ic) printf "IC" ;;
    *)
      printf "Unknown family: %s\n" "$1" >&2
      exit 2
      ;;
  esac
}

family_seed_offset() {
  case "$1" in
    mosfet) printf "1" ;;
    igbt) printf "2" ;;
    ic) printf "3" ;;
    *)
      printf "Unknown family: %s\n" "$1" >&2
      exit 2
      ;;
  esac
}

LABEL="$(dataset_label "$COUNT")"
FAMILY_NAME="$(family_file_name "$FAMILY")"
SEED="$((COUNT + $(family_seed_offset "$FAMILY")))"
OUT_DIR="${OUTPUT_ROOT}/${LABEL}/raw"
OUT_FILE="${OUT_DIR}/${FAMILY_NAME}.csv"

mkdir -p "$OUT_DIR" outputs/slurm

if [[ -s "$OUT_FILE" && "$FORCE" != "1" ]]; then
  printf "skip existing: %s\n" "$OUT_FILE"
  exit 0
fi

printf "task_id=%s count=%s family=%s seed=%s output=%s\n" \
  "$TASK_ID" "$COUNT" "$FAMILY" "$SEED" "$OUT_FILE"

time "$RUNNER" uv run python "$GENERATOR" \
  --family "$FAMILY" \
  --count "$COUNT" \
  --seed "$SEED" \
  --output "$OUT_FILE"

if [[ "$VALIDATE_AFTER" == "1" ]]; then
  "$RUNNER" uv run python "$GENERATOR" --validate "$OUT_FILE"
fi

du -h "$OUT_FILE"
