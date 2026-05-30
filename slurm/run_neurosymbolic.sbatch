#!/bin/bash
#SBATCH --job-name=neurosymbolic
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

# ALiBi GPT at 10k/family, leave-one-family-out, WITH vs WITHOUT neurosymbolic decoding
# (mask next steps that violate the 10 process rules during prediction/completion).
# Array task 0 = plain decoding, task 1 = symbolic-masked. Same trained model, different inference.
#
# Submit:  sbatch slurm/run_neurosymbolic.sbatch
# Compare: outputs/ns_off/... vs outputs/ns_on/... (next_step + completion gain expected)

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
GPT_EPOCHS="${GPT_EPOCHS:-30}"
GPT_PATIENCE="${GPT_PATIENCE:-4}"
GPT_D_MODEL="${GPT_D_MODEL:-256}"
GPT_LAYERS="${GPT_LAYERS:-4}"
RUNNER="${RUNNER:-srun}"

if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
  OUT=outputs/ns_off
  NS=()
else
  OUT=outputs/ns_on
  NS=(--neurosymbolic)
fi

if [[ "${MAKE_EVAL_SETS:-1}" == "1" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    DATASETS="$DATASET" HOLDOUT_FAMILIES="$HOLDOUT_FAMILIES" bash local_scripts/make_eval_sets.sh
    touch "data/eval/${DATASET}/.eval_done"
  else
    echo "task ${SLURM_ARRAY_TASK_ID}: waiting for eval sets ..."
    until [[ -f "data/eval/${DATASET}/.eval_done" ]]; do sleep 10; done
  fi
fi

echo "=== neurosymbolic=${SLURM_ARRAY_TASK_ID:-0} out=${OUT} ==="
$RUNNER uv run python scripts/run_holdout_experiments.py \
  --datasets "$DATASET" \
  --holdout-families $HOLDOUT_FAMILIES \
  --models gpt \
  --tasks next_step completion anomaly \
  --limit-per-family "$LIMIT" \
  "${NS[@]}" \
  --save-checkpoints \
  --preds-root "$OUT/preds" \
  --metrics-root "$OUT/metrics" \
  --plots-root "$OUT/plots" \
  --checkpoints-root "$OUT/checkpoints" \
  --gpt-epochs "$GPT_EPOCHS" --gpt-patience "$GPT_PATIENCE" \
  --gpt-d-model "$GPT_D_MODEL" --gpt-layers "$GPT_LAYERS"
