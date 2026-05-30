#!/usr/bin/env bash
set -euo pipefail

# Generate augmented Industrial AI process-route datasets for all requested
# sizes and families.

SIZES="${SIZES:-5000 10000 20000 100000 500000 1000000}"
FAMILIES="${FAMILIES:-mosfet igbt ic}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/generated}"
FORCE="${FORCE:-0}"
NO_VALIDATE="${NO_VALIDATE:-0}"
AUGMENTED_EXTRA_ARGS="${AUGMENTED_EXTRA_ARGS:-}"

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

usage() {
  cat <<'EOF'
Usage: local_scripts/generate_augmented_datasets.sh [options]

Options:
  --force                 overwrite existing raw CSV files
  --sizes "..."           space-separated sequence counts
  --families "..."        space-separated families
  --output-root PATH      generated-data root
  --no-validate           skip generator validator rejection
  --extra-args "..."      extra args for scripts/generate_augmented_industrial.py
  -h, --help              show this help

Environment variables with the same names are also supported:
  FORCE=1
  SIZES="5000 10000"
  FAMILIES="mosfet igbt ic"
  OUTPUT_ROOT=data/generated
  NO_VALIDATE=1
  AUGMENTED_EXTRA_ARGS="--litho-cycles 6 --second-metal-layer random"
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
    --output-root)
      OUTPUT_ROOT="${2:?Missing value for --output-root}"
      shift 2
      ;;
    --no-validate)
      NO_VALIDATE=1
      shift
      ;;
    --extra-args)
      AUGMENTED_EXTRA_ARGS="${2:?Missing value for --extra-args}"
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
  dataset="$(dataset_label "$size")"
  for family in $FAMILIES; do
    seed="$((size + $(family_seed_offset "$family")))"
    args=(
      --count "$size"
      --dataset "$dataset"
      --output-root "$OUTPUT_ROOT"
      --families "$family"
      --seed "$seed"
    )

    if [[ "$FORCE" == "1" ]]; then
      args+=(--force)
    fi

    if [[ "$NO_VALIDATE" == "1" ]]; then
      args+=(--no-validate)
    fi

    if [[ -n "$AUGMENTED_EXTRA_ARGS" ]]; then
      # shellcheck disable=SC2206
      args+=($AUGMENTED_EXTRA_ARGS)
    fi

    printf "\n==> generating augmented %s sequences for %s -> %s/%s/raw\n" \
      "$size" "$family" "$OUTPUT_ROOT" "$dataset"
    time uv run python scripts/generate_augmented_industrial.py "${args[@]}"
  done
done
