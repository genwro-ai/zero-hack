#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
Usage:
  local_scripts/run_family_holdouts.sh <model-train-command> [args...]

Examples:
  local_scripts/run_family_holdouts.sh uv run python -m zero_hack.models.ngram.train --n 5
  local_scripts/run_family_holdouts.sh uv run python -m zero_hack.models.lstm.train --epochs 1 --device cpu
EOF
  exit 2
fi

for holdout in mosfet igbt ic; do
  printf "\n==> holdout_family=%s\n" "$holdout"
  "$@" --holdout-family "$holdout"
done
