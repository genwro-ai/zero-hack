#!/bin/bash
#SBATCH --job-name=zero-hack-dpo-pairs
#SBATCH --array=0-2
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Generate DPO preference pairs for each two-family-train holdout.
#
# Submit:
#   sbatch slurm/generate_dpo_pairs.sbatch
#
# Outputs:
#   data/generated/dpo_pairs/<run_dataset>/holdout_<family>/pairs.jsonl

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
AUGMENT_COUNT="${AUGMENT_COUNT:-50000}"
AUGMENT_DATASET="${AUGMENT_DATASET:-augmented_s050k}"
RUN_DATASET="${RUN_DATASET:-${DATASET}_augmented_s050k}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
DPO_ROOT="${DPO_ROOT:-data/generated/dpo_pairs}"
PAIRS="${PAIRS:-100000}"
RUNNER="${RUNNER:-srun}"

HOLDOUTS=(mosfet igbt ic)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
HOLDOUT_FAMILY="${HOLDOUTS[$TASK_ID]}"
SPLITS_DIR="$GENERATED_ROOT/$DATASET/splits"
AUGMENT_CSV="$GENERATED_ROOT/$AUGMENT_DATASET/raw.csv"
OUT="$DPO_ROOT/$RUN_DATASET/holdout_$HOLDOUT_FAMILY/pairs.jsonl"

mkdir -p outputs/slurm "$(dirname "$OUT")"

if [[ ! -s "$SPLITS_DIR/MOSFET_train.csv" ]]; then
  printf "missing splits: %s\n" "$SPLITS_DIR" >&2
  printf "run slurm/train_gpt_phase_augmented.sbatch or the data split scripts first\n" >&2
  exit 2
fi

if [[ ! -s "$AUGMENT_CSV" || "${FORCE_AUGMENT:-0}" == "1" ]]; then
  printf "generating augmentation count=%s -> %s\n" "$AUGMENT_COUNT" "$AUGMENT_CSV"
  time "$RUNNER" uv run python scripts/generate_augmented_sequences.py \
    --count "$AUGMENT_COUNT" \
    --dataset "$AUGMENT_DATASET" \
    --seed "${AUGMENT_SEED:-4250}"
fi

if [[ -s "$OUT" && "${FORCE_DPO_PAIRS:-0}" != "1" ]]; then
  printf "DPO pairs exist: %s\n" "$OUT"
  exit 0
fi

printf "generating DPO pairs holdout=%s pairs=%s -> %s\n" \
  "$HOLDOUT_FAMILY" "$PAIRS" "$OUT"

time "$RUNNER" uv run python scripts/generate_dpo_pairs.py \
  --splits-dir "$SPLITS_DIR" \
  --holdout-family "$HOLDOUT_FAMILY" \
  --augment-train-csv "$AUGMENT_CSV" \
  --augment-family-mode unknown \
  --pairs "$PAIRS" \
  --out "$OUT" \
  --seed "${DPO_SEED:-1729}"

ls -lh "$OUT" "$OUT.meta.json"
