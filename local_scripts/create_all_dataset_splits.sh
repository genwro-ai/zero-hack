#!/usr/bin/env bash
set -euo pipefail

# Create train/valid/test splits for all generated dataset sizes.

DATASETS="${DATASETS:-valid_s005k valid_s010k valid_s020k valid_s100k valid_s500k valid_s1000k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/generated}"
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
  --seed INT              split seed
  -h, --help              show this help

Environment variables with the same names are also supported:
  FORCE=1
  DATASETS="valid_s005k valid_s010k"
  FAMILIES="mosfet igbt ic"
  OUTPUT_ROOT=data/generated
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
    --families $FAMILIES
    --seed "$SPLIT_SEED"
  )

  if [[ "$FORCE" == "1" ]]; then
    args+=(--force)
  fi

  printf "\n==> creating splits for %s\n" "$dataset"
  uv run python scripts/create_dataset_splits.py "${args[@]}"
done
