#!/bin/bash
#SBATCH --job-name=gpt-ensemble-train
#SBATCH --array=0-19%20
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Train one ensemble member per array task (seed = SLURM_ARRAY_TASK_ID).
#
# Submit:
#   sbatch slurm/train_gpt_ensemble.sbatch
# Then evaluate (after the array finishes):
#   sbatch --dependency=afterok:<TRAIN_JOBID> slurm/eval_gpt_ensemble.sbatch
#
# Overrides (env): SPLITS_DIR, HOLDOUT_FAMILY, LIMIT_PER_FAMILY, MEMBERS_DIR,
#                  EPOCHS, D_MODEL, LAYERS. Change member count via --array.

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi
if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/setup_uv.sh

SPLITS_DIR="${SPLITS_DIR:-data/generated/valid_s020k/splits}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"
MEMBERS_DIR="${MEMBERS_DIR:-outputs/ensemble/${HOLDOUT_FAMILY}/members}"
EPOCHS="${EPOCHS:-6}"
D_MODEL="${D_MODEL:-256}"
LAYERS="${LAYERS:-4}"
SEED="${SLURM_ARRAY_TASK_ID:-0}"
RUNNER="${RUNNER:-srun}"

EXTRA=()
[[ -n "${HOLDOUT_FAMILY}" ]] && EXTRA+=(--holdout-family "${HOLDOUT_FAMILY}")
[[ -n "${LIMIT_PER_FAMILY:-}" ]] && EXTRA+=(--limit-per-family "${LIMIT_PER_FAMILY}")

$RUNNER uv run python scripts/train_gpt_member.py \
  --splits-dir "${SPLITS_DIR}" \
  --seed "${SEED}" \
  --out "${MEMBERS_DIR}/member_$(printf '%03d' "${SEED}").pt" \
  --epochs "${EPOCHS}" --d-model "${D_MODEL}" --layers "${LAYERS}" \
  "${EXTRA[@]}"
