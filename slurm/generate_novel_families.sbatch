#!/bin/bash
#SBATCH --job-name=zero-hack-novel
#SBATCH --array=0-2
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err

set -euo pipefail

# Slurm array job for the leak-free novel-family OOD eval sets.
#
# Each array task generates one profile of genuinely novel, rule-valid,
# family-less process flows via scripts/generate_novel_families.py. Unlike the
# pseudo-family recombination, this composes sequences from atomic role-typed
# operation pools, so the flows satisfy process logic but match no official
# family backbone. Output is eval-only and must never enter training:
#   data/eval/novel_families/<profile>/raw.csv
#   data/eval/novel_families/<profile>/meta.json   # config + novelty audit
#
# Submit with:
#   sbatch slurm/generate_novel_families.sbatch
# Single profile:
#   sbatch --array=2 slurm/generate_novel_families.sbatch
#
# Optional environment overrides:
#   NOVEL_COUNT=5000                 # sequences per profile
#   OUT_ROOT=data/eval               # writes under <OUT_ROOT>/novel_families/
#   NOVEL_SEED=1729
#   REF_COUNT=2000                   # real-family refs for the novelty audit
#   UV_CACHE_DIR=$SCRATCH/.cache/uv
#   UV_INSTALL_DIR=$SCRATCH/.local/bin
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

PROFILES=(
  novel_sparse
  novel_mixed
  novel_dense
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
PROFILE="${PROFILES[$TASK_ID]}"
NOVEL_COUNT="${NOVEL_COUNT:-5000}"
OUT_ROOT="${OUT_ROOT:-data/eval}"
NOVEL_SEED="${NOVEL_SEED:-1729}"
REF_COUNT="${REF_COUNT:-2000}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm "$OUT_ROOT"

printf "task_id=%s profile=%s count=%s out_root=%s seed=%s\n" \
  "$TASK_ID" "$PROFILE" "$NOVEL_COUNT" "$OUT_ROOT" "$NOVEL_SEED"

time "$RUNNER" uv run python scripts/generate_novel_families.py \
  --profiles "$PROFILE" \
  --count "$NOVEL_COUNT" \
  --out-root "$OUT_ROOT" \
  --ref-count "$REF_COUNT" \
  --seed "$NOVEL_SEED"

find "$OUT_ROOT/novel_families/$PROFILE" -maxdepth 1 -type f -exec ls -lh {} +
