#!/usr/bin/env bash
set -euo pipefail

# Pull experiment results from a directory under $SCRATCH on a remote server
# back into this repository. Only result directories are synced (outputs/,
# reports/), not the whole repo.
#
# Usage:
#   local_scripts/rsync_from_scratch.sh <leonardo-user>
#
# Defaults to LEONARDO:
#   REMOTE_HOST=login.leonardo.cineca.it
#
# Optional:
#   REMOTE_USER=...
#   REMOTE_HOST=...
#   REMOTE_DIR='$SCRATCH/zero-hack'
#   RESULT_DIRS="outputs reports"   # space-separated subdirs to pull back
#   DELETE=1
#   DRY_RUN=1
#   SSH_PORT=22
#
# Password auth is handled by ssh/rsync prompts. Do not put passwords in this
# script or commit them to the repository.

POSITIONAL_USER="${1:-}"
REMOTE_HOST="${REMOTE_HOST:-login.leonardo.cineca.it}"
REMOTE_USER="${REMOTE_USER:-$POSITIONAL_USER}"
if [[ -z "$REMOTE_USER" ]]; then
  printf "Usage: %s <leonardo-user>\n" "$0" >&2
  printf "Or set REMOTE_USER explicitly.\n" >&2
  exit 2
fi
REMOTE_DIR="${REMOTE_DIR:-\$SCRATCH/zero-hack}"
RESULT_DIRS="${RESULT_DIRS:-outputs reports}"
SSH_PORT="${SSH_PORT:-22}"
DELETE="${DELETE:-0}"
DRY_RUN="${DRY_RUN:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_ARGS=(
  --archive
  --compress
  --human-readable
  --progress
  --itemize-changes
  --relative
  --exclude ".DS_Store"
  --exclude "__pycache__/"
  --exclude "*.pyc"
)

if [[ "$DELETE" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

SSH_CMD=(ssh -p "$SSH_PORT")

# Build remote source list using --relative so the subdir structure is
# preserved on the local side (e.g. .../zero-hack/./outputs -> ROOT_DIR/outputs).
SOURCES=()
for dir in $RESULT_DIRS; do
  SOURCES+=("${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/./${dir}")
done

printf "pulling results [%s] from %s@%s:%s -> %s\n" \
  "$RESULT_DIRS" "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_DIR" "$ROOT_DIR"

rsync "${RSYNC_ARGS[@]}" \
  -e "${SSH_CMD[*]}" \
  "${SOURCES[@]}" \
  "${ROOT_DIR}/"
