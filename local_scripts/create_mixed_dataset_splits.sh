#!/usr/bin/env bash
set -euo pipefail

# Create 40/60 vanilla/augmented mixed splits for all requested dataset sizes.

SIZES="${SIZES:-5000 10000 20000 100000 500000 1000000}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
VANILLA_RATIO="${VANILLA_RATIO:-0.4}"
AUGMENTED_RATIO="${AUGMENTED_RATIO:-0.6}"
FAMILIES="${FAMILIES:-mosfet igbt ic}"
FORCE="${FORCE:-0}"
SPLIT_SEED="${SPLIT_SEED:-1729}"

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

usage() {
  cat <<'EOF'
Usage: local_scripts/create_mixed_dataset_splits.sh [options]

Options:
  --force                 overwrite existing split files
  --sizes "..."           space-separated sequence counts
  --families "..."        space-separated families
  --generated-root PATH   generated-data root
  --vanilla-ratio FLOAT   vanilla share in train/valid/test, default 0.4
  --augmented-ratio FLOAT augmented share in train/valid/test, default 0.6
  --seed INT              split seed
  -h, --help              show this help

Environment variables with the same names are also supported:
  FORCE=1
  SIZES="5000 10000"
  FAMILIES="mosfet igbt ic"
  GENERATED_ROOT=data/generated
  VANILLA_RATIO=0.4
  AUGMENTED_RATIO=0.6
  SPLIT_SEED=1729
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --sizes)
      SIZES="${2:?Missing value for --sizes}"
      shift 2
      ;;
    --families)
      FAMILIES="${2:?Missing value for --families}"
      shift 2
      ;;
    --generated-root)
      GENERATED_ROOT="${2:?Missing value for --generated-root}"
      shift 2
      ;;
    --vanilla-ratio)
      VANILLA_RATIO="${2:?Missing value for --vanilla-ratio}"
      shift 2
      ;;
    --augmented-ratio)
      AUGMENTED_RATIO="${2:?Missing value for --augmented-ratio}"
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

for size in $SIZES; do
  vanilla_dataset="$(dataset_label valid "$size")"
  augmented_dataset="$(dataset_label augmented "$size")"
  dataset="mixed_s$(printf "%03dk" "$((size / 1000))")_v$(ratio_label "$VANILLA_RATIO")_a$(ratio_label "$AUGMENTED_RATIO")"

  args=(
    --count "$size"
    --dataset "$dataset"
    --generated-root "$GENERATED_ROOT"
    --vanilla-dataset "$vanilla_dataset"
    --augmented-dataset "$augmented_dataset"
    --vanilla-ratio "$VANILLA_RATIO"
    --augmented-ratio "$AUGMENTED_RATIO"
    --families $FAMILIES
    --seed "$SPLIT_SEED"
  )

  if [[ "$FORCE" == "1" ]]; then
    args+=(--force)
  fi

  printf "\n==> creating mixed splits for %s from %s + %s\n" \
    "$dataset" "$vanilla_dataset" "$augmented_dataset"
  uv run python scripts/create_mixed_dataset_splits.py "${args[@]}"
done
