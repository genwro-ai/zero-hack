#!/usr/bin/env bash
set -euo pipefail

# Sync a directory under $SCRATCH on a remote server back to this repository.
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
#   DELETE=1
#   DRY_RUN=1
#   INCLUDE_GIT=1
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
SSH_PORT="${SSH_PORT:-22}"
DELETE="${DELETE:-0}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_GIT="${INCLUDE_GIT:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RSYNC_ARGS=(
  --archive
  --compress
  --human-readable
  --progress
  --itemize-changes
  --exclude ".DS_Store"
  --exclude ".ruff_cache/"
  --exclude ".venv/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude "data/generated/"
  --exclude "outputs/slurm/"
)

if [[ "$DELETE" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

if [[ "$INCLUDE_GIT" != "1" ]]; then
  RSYNC_ARGS+=(--exclude ".git/")
fi

SSH_CMD=(ssh -p "$SSH_PORT")

printf "syncing %s@%s:%s -> %s\n" "$REMOTE_USER" "$REMOTE_HOST" "$REMOTE_DIR" "$ROOT_DIR"

rsync "${RSYNC_ARGS[@]}" \
  -e "${SSH_CMD[*]}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/" \
  "${ROOT_DIR}/"
