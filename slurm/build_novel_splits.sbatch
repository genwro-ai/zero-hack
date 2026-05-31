#!/bin/bash
#SBATCH --job-name=zero-hack-novel-splits
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

# Data-prep job for the novel-family eval flows: (1) generate the novel
# raw.csv files, then (2) assign arbitrary conditioning-family tokens +
# split them into train/valid/test that the transformer training pipeline
# consumes (slurm/train_transformer_pseudo.sbatch or similar).
#
# IMPORTANT: novel profiles are FAMILY-LESS (no base_family in meta.json).
# The profile→family mapping below (MOSFET/IGBT/IC) is an arbitrary eval
# construct that lets --holdout-family work syntactically.  Leave-one-
# (assigned)-family-out is closer to in-distribution than true OOD; the
# stronger experiment is to merge these splits into a real dataset split and
# hold out a real family.
#
# CPU/IO only (no GPU math) but runs on a compute node, so it relies on the
# login-warmed uv cache (see setup_uv.sh).
#
# Submit:
#   sbatch slurm/build_novel_splits.sbatch
# Reuse existing raw.csv (skip generation, only re-split):
#   SKIP_GENERATE=1 sbatch slurm/build_novel_splits.sbatch
#
# Optional environment overrides:
#   COUNT=5000                                           # sequences per profile
#   SEED=42                                              # generation seed
#   PROFILES="novel_mixed novel_sparse novel_dense"      # profiles to generate
#   IN_ROOT=data/eval/novel_families
#   OUT_ROOT=data/generated/novel/splits
#   TRAIN=0.8  VALID=0.1                                 # split fractions (rest -> test)
#   SKIP_GENERATE=0                                      # 1 = do not regenerate raw.csv

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
IN_ROOT="${IN_ROOT:-data/eval/novel_families}"
OUT_ROOT="${OUT_ROOT:-data/generated/novel/splits}"
TRAIN="${TRAIN:-0.8}"
VALID="${VALID:-0.1}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
RUNNER="${RUNNER:-srun}"

mkdir -p outputs/slurm

if [[ "$SKIP_GENERATE" != "1" ]]; then
  printf "generating novel families: count=%s seed=%s profiles=[%s]\n" "$COUNT" "$SEED" "$PROFILES"
  # shellcheck disable=SC2086
  time "$RUNNER" uv run python scripts/generate_novel_families.py \
    --count "$COUNT" --seed "$SEED" --profiles $PROFILES --out-root "$IN_ROOT"
else
  echo "SKIP_GENERATE=1: reusing existing raw.csv under $IN_ROOT"
fi

printf "building splits: in=%s out=%s train=%s valid=%s\n" "$IN_ROOT" "$OUT_ROOT" "$TRAIN" "$VALID"
time "$RUNNER" uv run python -m scripts.build_novel_splits \
  --in-root "$IN_ROOT" --out-root "$OUT_ROOT" --train "$TRAIN" --valid "$VALID"

echo "=== novel splits ==="
find "$OUT_ROOT" -maxdepth 1 -type f -name "*.csv" -exec ls -lh {} +
