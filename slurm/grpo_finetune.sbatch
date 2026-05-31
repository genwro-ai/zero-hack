#!/bin/bash
#SBATCH --job-name=zero-hack-grpo
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail

# GRPO (Group Relative Policy Optimization) finetuning of a pretrained next-step
# model (LSTM or transformer decoder) using the process-rule validator as a
# verifiable reward, for the sequence-completion objective.
#
# Submit:
#   CKPT=outputs/models/valid_s005k/lstm_teacher_forcing/best.pt \
#   HOLDOUT_FAMILY=ic sbatch slurm/grpo_finetune.sbatch
#
# Required environment:
#   CKPT                          # pretrained policy checkpoint (.pt) to finetune
#
# Optional environment overrides:
#   DATASET=valid_s005k           # which generated dataset's splits to draw prompts from
#   SPLITS_ROOT=data/generated    # where the dataset splits/ dirs live
#   HOLDOUT_FAMILY=ic             # mosfet | igbt | ic (kept unseen; not a prompt family)
#   GROUP_SIZE=8                  # completions sampled per prompt (the GRPO group)
#   STEPS=200                     # number of GRPO updates
#   LR=1e-5
#   KL_COEF=0.02                  # KL(policy||frozen-reference) weight
#   OUT=                          # output checkpoint (default best_grpo.pt next to CKPT)
#   EXTRA_ARGS="--temperature 1.0 --w-block 0.8 --w-length 0.4"  # appended verbatim to the CLI

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

if [[ -z "${CKPT:-}" ]]; then
  echo "CKPT is required (path to the pretrained policy checkpoint)." >&2
  exit 1
fi

DATASET="${DATASET:-valid_s005k}"
SPLITS_ROOT="${SPLITS_ROOT:-data/generated}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"
GROUP_SIZE="${GROUP_SIZE:-8}"
STEPS="${STEPS:-200}"
LR="${LR:-1e-5}"
KL_COEF="${KL_COEF:-0.02}"
RUNNER="${RUNNER:-srun}"

SPLITS_DIR="${SPLITS_ROOT}/${DATASET}/splits"
OUT="${OUT:-$(dirname "$CKPT")/best_grpo.pt}"

mkdir -p outputs/slurm "$(dirname "$OUT")"

ARGS=(
  --checkpoint "$CKPT"
  --dataset "$DATASET"
  --splits-dir "$SPLITS_DIR"
  --holdout-family "$HOLDOUT_FAMILY"
  --group-size "$GROUP_SIZE"
  --steps "$STEPS"
  --lr "$LR"
  --kl-coef "$KL_COEF"
  --out "$OUT"
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "ckpt=%s dataset=%s holdout=%s group_size=%s steps=%s lr=%s kl=%s out=%s\n" \
  "$CKPT" "$DATASET" "$HOLDOUT_FAMILY" "$GROUP_SIZE" "$STEPS" "$LR" "$KL_COEF" "$OUT"

# setup_uv.sh already provisioned the venv with `uv sync --no-dev`; --no-sync
# reuses it as-is so `uv run` does not re-sync the dev group (pytest, pygments,
# ...) which is uncached on the offline compute nodes.
time "$RUNNER" uv run --no-sync python scripts/grpo_finetune.py "${ARGS[@]}"

ls -lh "$(dirname "$OUT")"
