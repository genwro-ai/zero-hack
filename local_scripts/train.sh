#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=src

# Train a single ALiBi GPT decoder and save a checkpoint (early stopping on valid loss).
# Checkpoint -> outputs/models/<dataset>/gpt_decoder_holdout_<family>/best.pt
#
# Usage:   bash local_scripts/train.sh
# Override env: DATASET, HOLDOUT, LIMIT, EPOCHS, D_MODEL, LAYERS
#   DATASET=valid_s020k HOLDOUT=ic LIMIT=10000 bash local_scripts/train.sh
# Any extra flags are passed straight to the trainer:
#   bash local_scripts/train.sh --lr 3e-4 --batch-size 128

DATASET="${DATASET:-valid_s100k}"
HOLDOUT="${HOLDOUT:-ic}"
LIMIT="${LIMIT:-10000}"
EPOCHS="${EPOCHS:-30}"
D_MODEL="${D_MODEL:-256}"
LAYERS="${LAYERS:-4}"

uv run python src/zero_hack/models/gpt/train.py \
  --dataset "$DATASET" \
  --holdout-family "$HOLDOUT" \
  --limit-per-family "$LIMIT" \
  --epochs "$EPOCHS" \
  --d-model "$D_MODEL" \
  --num-layers "$LAYERS" \
  --nhead 8 \
  --dim-feedforward $((D_MODEL * 4)) \
  --max-context 256 \
  "$@"
