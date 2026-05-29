#!/usr/bin/env bash
#SBATCH --job-name=zero-hack-split
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Slurm array job for creating train/valid/test splits from generated raw CSVs.
#
# It expects raw files like:
#   data/generated/valid_s005k/raw/MOSFET.csv
#   data/generated/valid_s005k/raw/IGBT.csv
#   data/generated/valid_s005k/raw/IC.csv
#
# Submit through:
#   scripts/submit_create_dataset_splits.sh
#
# Optional environment overrides:
#   OUTPUT_ROOT=/scratch/$USER/zero-hack/generated
#   FAMILIES="mosfet igbt ic"
#   FORCE=1
#   SPLIT_SEED=1729

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

DATASETS=(
  valid_s005k
  valid_s010k
  valid_s020k
  valid_s100k
  valid_s500k
  valid_s1000k
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASETS[$TASK_ID]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/generated}"
FAMILIES="${FAMILIES:-mosfet igbt ic}"
SPLIT_SEED="${SPLIT_SEED:-1729}"
FORCE="${FORCE:-0}"

INPUT_DIR="${OUTPUT_ROOT}/${DATASET}/raw"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/splits"

mkdir -p outputs/slurm

ARGS=(
  --dataset "$DATASET"
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --families $FAMILIES
  --seed "$SPLIT_SEED"
)

if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

printf "task_id=%s dataset=%s input=%s output=%s seed=%s\n" \
  "$TASK_ID" "$DATASET" "$INPUT_DIR" "$OUTPUT_DIR" "$SPLIT_SEED"

time uv run python scripts/create_dataset_splits.py "${ARGS[@]}"

find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.csv" -exec ls -lh {} +
