#!/bin/bash
#SBATCH --job-name=zero-hack-split
#SBATCH --array=0-5
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks-per-node=1 # Number of `srun` tasks executed per node
#SBATCH --gpus-per-task=1 # Number of GPUs (up to 4 on Leonardo)
#SBATCH --mem=120GB # Fair share on Leonardo: 120GB * gpus-per-task
#SBATCH --cpus-per-task=8 # Fair share on Leonardo: 8 * gpus-per-task
#SBATCH --time=0:30:00 # Time limit in HH:MM:SS, up to 24:00:00
#SBATCH --output=outputs/slurm/%x_%A_%a.out
#SBATCH --error=outputs/slurm/%x_%A_%a.err
# Array job: END emails arrive per task; change to FAIL to reduce volume.
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Slurm array job for creating train/valid/test splits from generated raw CSVs.
#
# It expects raw files like:
#   data/generated/valid_s005k/raw/MOSFET.csv
#   data/generated/valid_s005k/raw/IGBT.csv
#   data/generated/valid_s005k/raw/IC.csv
#
# Submit with:
#   sbatch slurm/create_dataset_splits.sbatch
#
# Optional environment overrides:
#   OUTPUT_ROOT=data/generated
#   INDUSTRIAL_DIR=data/industrial
#   INCLUDE_INDUSTRIAL=1
#   FAMILIES="mosfet igbt ic"
#   FORCE=1
#   SPLIT_SEED=1729
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

DATASETS=(
  valid_s005k
  valid_s010k
  valid_s020k
  valid_s100k
  valid_s500k
  valid_s1000k
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASETS[$TASK_ID]}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/generated}"
INDUSTRIAL_DIR="${INDUSTRIAL_DIR:-data/industrial}"
INCLUDE_INDUSTRIAL="${INCLUDE_INDUSTRIAL:-1}"
FAMILIES="${FAMILIES:-mosfet igbt ic}"
SPLIT_SEED="${SPLIT_SEED:-1729}"
FORCE="${FORCE:-0}"
RUNNER="${RUNNER:-srun}"

INPUT_DIR="${OUTPUT_ROOT}/${DATASET}/raw"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/splits"

mkdir -p outputs/slurm

ARGS=(
  --dataset "$DATASET"
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --industrial-dir "$INDUSTRIAL_DIR"
  --families $FAMILIES
  --seed "$SPLIT_SEED"
)

if [[ "$INCLUDE_INDUSTRIAL" != "1" ]]; then
  ARGS+=(--no-include-industrial)
fi

if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

printf "task_id=%s dataset=%s input=%s output=%s seed=%s\n" \
  "$TASK_ID" "$DATASET" "$INPUT_DIR" "$OUTPUT_DIR" "$SPLIT_SEED"

time "$RUNNER" uv run python scripts/create_dataset_splits.py "${ARGS[@]}"

find "$OUTPUT_DIR" -maxdepth 1 -type f -name "*.csv" -exec ls -lh {} +
