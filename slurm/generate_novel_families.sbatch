#!/bin/bash
#SBATCH --job-name=zero-hack-novel-gen
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=120GB # Fair share on Leonardo: 120GB * gpus-per-task
#SBATCH --cpus-per-task=8 # Fair share on Leonardo: 8 * gpus-per-task
#SBATCH --time=0:30:00 # Time limit in HH:MM:SS, up to 24:00:00
#SBATCH --output=outputs/slurm/%x_%A.out
#SBATCH --error=outputs/slurm/%x_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Standalone generation job for the NOVEL (family-less) process-grammar datasets.
# This runs ONLY scripts/generate_novel_families.py -- it does not split. Use it
# when you want the eval-only raw.csv files on disk without (re)building splits;
# to generate AND split in one job use slurm/build_novel_splits.sbatch instead.
#
# Unlike the pseudo-families (whole-block recombination), these flows are sampled
# atomically against the organizer validator + closed vocab, so they belong to no
# official family. Output (one dir per profile):
#   data/eval/novel_families/novel_mixed/raw.csv   + meta.json
#   data/eval/novel_families/novel_sparse/raw.csv  + meta.json
#   data/eval/novel_families/novel_dense/raw.csv   + meta.json
#
# CPU/IO only (no GPU math), but runs on a compute node so it relies on the
# login-warmed uv cache (see setup_uv.sh). Acceptance rate is ~100%, so 5k/profile
# completes well within the time limit.
#
# Submit:
#   sbatch slurm/generate_novel_families.sbatch
#
# Optional environment overrides:
#   COUNT=5000                                              # sequences per profile
#   SEED=42                                                 # generation seed
#   PROFILES="novel_mixed novel_sparse novel_dense"         # which profiles
#   REF_COUNT=400                                           # family seqs/family for the novelty audit
#   OUT_ROOT=data/eval/novel_families                       # output root
#   UV_CACHE_DIR=$SCRATCH/.cache/uv
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

COUNT="${COUNT:-5000}"
SEED="${SEED:-42}"
PROFILES="${PROFILES:-novel_mixed novel_sparse novel_dense}"
REF_COUNT="${REF_COUNT:-400}"
OUT_ROOT="${OUT_ROOT:-data/eval/novel_families}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm "$OUT_ROOT"

printf "generating novel families: count=%s seed=%s ref_count=%s profiles=[%s] out=%s\n" \
  "$COUNT" "$SEED" "$REF_COUNT" "$PROFILES" "$OUT_ROOT"

# shellcheck disable=SC2086
time "$RUNNER" uv run python scripts/generate_novel_families.py \
  --count "$COUNT" --seed "$SEED" --ref-count "$REF_COUNT" \
  --profiles $PROFILES --out-root "$OUT_ROOT"

echo "=== novel family outputs ==="
find "$OUT_ROOT" -maxdepth 2 -type f \( -name "raw.csv" -o -name "meta.json" \) -exec ls -lh {} +
