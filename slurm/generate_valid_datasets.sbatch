#!/usr/bin/env bash
#SBATCH --job-name=zero-hack-data
#SBATCH --array=0-17%6
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Slurm array job for generating valid Industrial AI datasets.
#
# It creates the same layout as scripts/generate_valid_datasets.sh:
#   data/generated/valid_s005k/raw/MOSFET.csv
#   data/generated/valid_s010k/raw/MOSFET.csv
#   ...
#
# Submit through:
#   scripts/submit_generate_valid_datasets.sh
#
# Optional environment overrides:
#   OUTPUT_ROOT=/scratch/$USER/zero-hack/generated
#   FORCE=1
#   VALIDATE_AFTER=1
#   UV_CACHE_DIR=/scratch/$USER/.cache/uv

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

SIZES=(5000 10000 20000 100000 500000 1000000)
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

time uv run python "$GENERATOR" \
  --family "$FAMILY" \
  --count "$COUNT" \
  --seed "$SEED" \
  --output "$OUT_FILE"

if [[ "$VALIDATE_AFTER" == "1" ]]; then
  uv run python "$GENERATOR" --validate "$OUT_FILE"
fi

du -h "$OUT_FILE"
