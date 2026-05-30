#!/usr/bin/env bash
set -euo pipefail

# Create train/valid/test splits for all generated dataset sizes.

DATASETS="${DATASETS:-valid_s005k valid_s010k valid_s020k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/generated}"
INDUSTRIAL_DIR="${INDUSTRIAL_DIR:-data/industrial}"
INCLUDE_INDUSTRIAL="${INCLUDE_INDUSTRIAL:-1}"
FAMILIES="${FAMILIES:-mosfet igbt ic}"
FORCE="${FORCE:-0}"
SPLIT_SEED="${SPLIT_SEED:-1729}"

usage() {
  cat <<'EOF'
Usage: local_scripts/create_all_dataset_splits.sh [options]

Options:
  --force                 overwrite existing split files
  --datasets "..."        space-separated dataset labels
  --families "..."        space-separated families
  --output-root PATH      generated-data root
  --industrial-dir PATH   provided industrial-data root
  --no-include-industrial use only generated raw CSVs
  --seed INT              split seed
  -h, --help              show this help

Environment variables with the same names are also supported:
  FORCE=1
  DATASETS="valid_s005k valid_s010k"
  FAMILIES="mosfet igbt ic"
  OUTPUT_ROOT=data/generated
  INDUSTRIAL_DIR=data/industrial
  INCLUDE_INDUSTRIAL=1
  SPLIT_SEED=1729
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --datasets)
      DATASETS="${2:?Missing value for --datasets}"
      shift 2
      ;;
    --families)
      FAMILIES="${2:?Missing value for --families}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?Missing value for --output-root}"
      shift 2
      ;;
    --industrial-dir)
      INDUSTRIAL_DIR="${2:?Missing value for --industrial-dir}"
      shift 2
      ;;
    --no-include-industrial)
      INCLUDE_INDUSTRIAL=0
      shift
      ;;
    --seed)
      SPLIT_SEED="${2:?Missing value for --seed}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf "Unknown argument: %s\n\n" "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for dataset in $DATASETS; do
  input_dir="${OUTPUT_ROOT}/${dataset}/raw"
  output_dir="${OUTPUT_ROOT}/${dataset}/splits"

  if [[ ! -d "$input_dir" ]]; then
    printf "skip missing raw dataset: %s\n" "$input_dir"
    continue
  fi

  args=(
    --dataset "$dataset"
    --input-dir "$input_dir"
    --output-dir "$output_dir"
    --industrial-dir "$INDUSTRIAL_DIR"
    --families $FAMILIES
    --seed "$SPLIT_SEED"
  )

  if [[ "$INCLUDE_INDUSTRIAL" != "1" ]]; then
    args+=(--no-include-industrial)
  fi

  if [[ "$FORCE" == "1" ]]; then
    args+=(--force)
  fi

  printf "\n==> creating splits for %s\n" "$dataset"
  uv run python scripts/create_dataset_splits.py "${args[@]}"
done
