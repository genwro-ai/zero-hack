#!/bin/bash
#SBATCH --job-name=zero-hack-ns-compare
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail

# Compare bare GPT next-step predictions with the neurosymbolic hard and shaped
# decoding heads.
#
# Default checkpoint assumes the GPT decoder was trained with:
#   sbatch --array=1 slurm/train_decoders.sbatch
#
# Submit:
#   sbatch slurm/compare_gpt_neurosymbolic.sbatch
#
# Common overrides:
#   DATASET=valid_s005k
#   HOLDOUT_FAMILY=ic
#   VIEW=ood
#   CHECKPOINT=outputs/slurm_runs/models/valid_s005k/gpt_decoder_holdout_ic/best.pt
#   OUTPUT=outputs/slurm_runs/neurosymbolic/valid_s005k/holdout_ic_ood.jsonl
#   LIMIT=100              # optional smoke cap
#   K=5
#   RUNNER=srun
#   LEONARDO_LOAD_MODULES=0

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

DATASET="${DATASET:-valid_s005k}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"
VIEW="${VIEW:-ood}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
SLURM_OUTPUT_ROOT="${SLURM_OUTPUT_ROOT:-outputs/slurm_runs}"
CHECKPOINT="${CHECKPOINT:-$SLURM_OUTPUT_ROOT/models/$DATASET/gpt_decoder_holdout_$HOLDOUT_FAMILY/best.pt}"
OUTPUT="${OUTPUT:-$SLURM_OUTPUT_ROOT/neurosymbolic/$DATASET/holdout_${HOLDOUT_FAMILY}_${VIEW}.jsonl}"
K="${K:-5}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm "$(dirname "$OUTPUT")"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --dataset "$DATASET"
  --holdout-family "$HOLDOUT_FAMILY"
  --view "$VIEW"
  --eval-root "$EVAL_ROOT"
  --output "$OUTPUT"
  --k "$K"
  --device cuda
)

if [[ -n "${LIMIT:-}" ]]; then
  ARGS+=(--limit "$LIMIT")
fi

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ARGS+=($EXTRA_ARGS)
fi

printf "dataset=%s holdout=%s view=%s checkpoint=%s output=%s k=%s\n" \
  "$DATASET" "$HOLDOUT_FAMILY" "$VIEW" "$CHECKPOINT" "$OUTPUT" "$K"

time "$RUNNER" uv run python scripts/compare_gpt_neurosymbolic.py "${ARGS[@]}"

ls -lh "$OUTPUT"
