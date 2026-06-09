#!/bin/bash
#SBATCH --job-name=gpt-ensemble-eval
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# Average the trained members' probabilities and score the holdout (id + ood).
# Requires the eval set built first:
#   DATASETS=valid_s020k HOLDOUT_FAMILIES=ic bash local_scripts/make_eval_sets.sh
#   (or scripts/make_all_eval_sets.py --datasets valid_s020k --holdout-families ic)
#
# Submit (after the training array):
#   sbatch --dependency=afterok:<TRAIN_JOBID> slurm/eval_gpt_ensemble.sbatch
#
# Overrides (env): DATASET, SPLITS_DIR, HOLDOUT_FAMILY, MEMBERS_DIR, EVAL_DIR, OUT, MAX_MEMBERS

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi
if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/setup_uv.sh

DATASET="${DATASET:-valid_s020k}"
HOLDOUT_FAMILY="${HOLDOUT_FAMILY:-ic}"
SPLITS_DIR="${SPLITS_DIR:-data/generated/${DATASET}/splits}"
MEMBERS_DIR="${MEMBERS_DIR:-outputs/ensemble/${HOLDOUT_FAMILY}/members}"
OUT="${OUT:-outputs/ensemble/${HOLDOUT_FAMILY}/eval}"
RUNNER="${RUNNER:-srun}"

# 1-vs-N on our held-out split, scored with the official data/industrial/eval_metrics.py.
$RUNNER uv run python scripts/eval_gpt_ensemble.py \
  --members-dir "${MEMBERS_DIR}" --splits-dir "${SPLITS_DIR}" \
  --holdout-family "${HOLDOUT_FAMILY}" --out "${OUT}"
