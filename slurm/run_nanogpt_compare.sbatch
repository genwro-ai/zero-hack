#!/bin/bash
#SBATCH --job-name=nanogpt-cmp
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --array=0-1
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

# Compare our ALiBi GPT (model=gpt) vs NanoGPT (absolute PE) via decoder_training.py,
# leave-one-family-out at 10k/family. Array task 0 = gpt, task 1 = nanogpt; each runs all 3
# holdouts. Next-step (Task 1) id/ood metrics land per architecture under OUT/metrics.
#
# Submit:  sbatch slurm/run_nanogpt_compare.sbatch
# Compare: OUT/metrics/<dataset>/holdout_<fam>/{id,ood}/{gpt,nanogpt}/next_step.json

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
EPOCHS="${EPOCHS:-30}"
D_MODEL="${D_MODEL:-256}"
LAYERS="${LAYERS:-4}"
OUT="${OUT:-outputs/nanogpt_compare}"
RUNNER="${RUNNER:-srun}"
MODELS=(gpt nanogpt)
ARCH="${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}"

if [[ "${MAKE_EVAL_SETS:-1}" == "1" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    DATASETS="$DATASET" HOLDOUT_FAMILIES="$HOLDOUT_FAMILIES" bash local_scripts/make_eval_sets.sh
    touch "data/eval/${DATASET}/.eval_done"
  else
    echo "task ${SLURM_ARRAY_TASK_ID}: waiting for eval sets ..."
    until [[ -f "data/eval/${DATASET}/.eval_done" ]]; do sleep 10; done
  fi
fi

for HOLDOUT in $HOLDOUT_FAMILIES; do
  echo "=== ${ARCH} holdout=${HOLDOUT} ==="
  $RUNNER uv run python src/zero_hack/models/decoder_training.py \
    --model "$ARCH" \
    --dataset "$DATASET" \
    --holdout-family "$HOLDOUT" \
    --limit-per-family "$LIMIT" \
    --epochs "$EPOCHS" \
    --d-model "$D_MODEL" --num-layers "$LAYERS" --nhead 8 \
    --dim-feedforward $((D_MODEL * 4)) --max-context 256 \
    --eval-root data/eval \
    --preds-root "$OUT/preds" \
    --metrics-root "$OUT/metrics" \
    --model-root "$OUT/models"
done
