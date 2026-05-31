#!/usr/bin/env bash
set -euo pipefail

# Generate valid Industrial AI process-route datasets for all requested sizes
# and families. The generator validates each generated sequence before writing.

SIZES="${SIZES:-5000 20000 100000}"
FAMILIES="${FAMILIES:-mosfet igbt ic}"
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

generate_one() {
  local family="$1"
  local count="$2"
  local label
  local family_name
  local seed
  local out_dir
  local out_file

  label="$(dataset_label "$count")"
  family_name="$(family_file_name "$family")"
  seed="$((count + $(family_seed_offset "$family")))"
  out_dir="${OUTPUT_ROOT}/${label}/raw"
  out_file="${out_dir}/${family_name}.csv"

  mkdir -p "$out_dir"

  if [[ -s "$out_file" && "$FORCE" != "1" ]]; then
    printf "skip existing: %s\n" "$out_file"
    return 0
  fi

  printf "\n==> generating %s sequences for %s -> %s\n" "$count" "$family" "$out_file"
  time uv run python "$GENERATOR" \
    --family "$family" \
    --count "$count" \
    --seed "$seed" \
    --output "$out_file"

  if [[ "$VALIDATE_AFTER" == "1" ]]; then
    printf "==> validating %s\n" "$out_file"
    uv run python "$GENERATOR" --validate "$out_file"
  fi

  du -h "$out_file"
}

for size in $SIZES; do
  for family in $FAMILIES; do
    generate_one "$family" "$size"
  done
done
