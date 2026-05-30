#!/bin/bash
#SBATCH --job-name=llama-exp
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

# From-scratch HuggingFace LlamaForCausalLM (~6M params, RoPE+RMSNorm+SwiGLU), no family token,
# leave-one-family-out at 10k/family. Per holdout: train once, then evaluate next-step + anomaly
# (surprisal) once and completion TWICE (plain greedy vs grammar-masked greedy) -> ablation.
# Step representation is compositional (word + char embeddings, tied) so held-out-family steps get
# real vectors instead of a single <UNK>.
# Outputs under outputs/llama (separate from other experiments). Needs `transformers` (uv sync on login).
#
# Submit:  sbatch slurm/run_llama_experiment.sbatch
# Result:  outputs/llama/valid_s100k/average.json + per-holdout results.json + printed table.
# To parallelise, submit one family at a time: HOLDOUT_FAMILIES=ic sbatch slurm/run_llama_experiment.sbatch

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi
if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source slurm/setup_uv.sh

DATASET="${DATASET:-valid_s100k}"
LIMIT="${LIMIT:-10000}"
HOLDOUT_FAMILIES="${HOLDOUT_FAMILIES:-mosfet igbt ic}"
EPOCHS="${EPOCHS:-20}"
RUNNER="${RUNNER:-srun}"

if [[ "${MAKE_EVAL_SETS:-1}" == "1" ]]; then
  DATASETS="$DATASET" HOLDOUT_FAMILIES="$HOLDOUT_FAMILIES" bash local_scripts/make_eval_sets.sh
fi

$RUNNER uv run python scripts/run_llama_experiment.py \
  --datasets "$DATASET" \
  --holdout-families $HOLDOUT_FAMILIES \
  --limit-per-family "$LIMIT" \
  --epochs "$EPOCHS" \
  --patience 5 --min-delta 0.0 \
  --grammar-nextstep \
  --out outputs/llama
