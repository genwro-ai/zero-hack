#!/bin/bash
#SBATCH --job-name=zero-hack-lstm-anomaly-lr
#SBATCH --account=euhpc_d30_031
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc # Reservation for the hackathon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm/%x_%A.out
#SBATCH --error=outputs/slurm/%x_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lukasz.lenkiewicz28@gmail.com

set -euo pipefail

# Likelihood-ratio anomaly eval (Task 3) in the HOLDOUT / handoff setting.
#
# For each held-out family it builds (if missing) the two anomaly eval views and
# scores the LOFO checkpoint with four detectors so the ratio is comparable to
# the raw-likelihood baseline:
#   id  = the two TRAINED families  (in-distribution sanity)
#   ood = the HELD-OUT family       (the actual handoff test)
#
# Pipeline per (family, view):
#   1. make_eval_set.py     -> data/eval/<dataset>/holdout_<fam>/<view>/{eval_input_anomaly,anomaly_truth}.csv
#   2. eval_lstm_anomaly_lr -> outputs/metrics/<dataset>/lstm_holdout_<fam>/anomaly_lr/<view>/results.{json,md}
#
# Needs LOFO checkpoints from slurm/train_lofo.sbatch at
#   outputs/models/<dataset>/lstm_holdout_<fam>/best.pt
#
# Submit (all three folds):
#   sbatch slurm/eval_lstm_anomaly_lr.sbatch
# Pin a single fold:
#   HOLDOUT_FAMILY=ic sbatch slurm/eval_lstm_anomaly_lr.sbatch
#
# Optional environment overrides:
#   DATASET=valid_s005k
#   SPLITS_ROOT=data/generated
#   MODEL_ROOT=outputs/models
#   METRICS_ROOT=outputs/metrics
#   EVAL_ROOT=data/eval
#   RUN=lstm_holdout            # checkpoint dir prefix: <RUN>_<fam>/best.pt
#   VIEWS="id ood"              # which views to score
#   HOLDOUT_FAMILY=            # mosfet | igbt | ic (pins the fold; default all)
#   BG_NGRAM_N=1               # background n-gram order (1=unigram nuisance; >=2 hurts the ratio)
#   LAMBDA=1.0                 # weight on background log-prob in the raw ratio
#   N_ANOMALY_VALID=200
#   N_ANOMALY_INVALID=129
#   SEED=1729
#   FORCE_REGEN=0              # 1 = rebuild eval views even if present
#   EXTRA_ARGS="--device cuda" # appended verbatim to the eval CLI

if [[ -n "${SCRATCH:-}" ]]; then
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
fi

if [[ "${LEONARDO_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  module load python/3.11.7 || true
  module load profile/deeplrn || true
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

source slurm/setup_uv.sh

DATASET="${DATASET:-valid_s005k}"
SPLITS_ROOT="${SPLITS_ROOT:-data/generated}"
MODEL_ROOT="${MODEL_ROOT:-outputs/models}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
RUN="${RUN:-lstm_holdout}"
VIEWS="${VIEWS:-id ood}"
BG_NGRAM_N="${BG_NGRAM_N:-1}"
LAMBDA="${LAMBDA:-1.0}"
N_ANOMALY_VALID="${N_ANOMALY_VALID:-200}"
N_ANOMALY_INVALID="${N_ANOMALY_INVALID:-129}"
SEED="${SEED:-1729}"
FORCE_REGEN="${FORCE_REGEN:-0}"
RUNNER="${RUNNER:-srun}"

ALL_FAMILIES=(mosfet igbt ic)
if [[ -n "${HOLDOUT_FAMILY:-}" ]]; then
  FAMILIES=("$HOLDOUT_FAMILY")
else
  FAMILIES=("${ALL_FAMILIES[@]}")
fi

SPLITS_DIR="${SPLITS_ROOT}/${DATASET}/splits"
mkdir -p outputs/slurm

# The two trained families for a given holdout (used to build the `id` view).
other_families() {
  local holdout="$1"
  local out=()
  for fam in "${ALL_FAMILIES[@]}"; do
    [[ "$fam" != "$holdout" ]] && out+=("$fam")
  done
  echo "${out[@]}"
}

for FAM in "${FAMILIES[@]}"; do
  CKPT="${MODEL_ROOT}/${DATASET}/${RUN}_${FAM}/best.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "skip holdout=${FAM}: checkpoint not found at ${CKPT}"
    continue
  fi

  for VIEW in $VIEWS; do
    EVAL_DIR="${EVAL_ROOT}/${DATASET}/holdout_${FAM}/${VIEW}"

    if [[ "$VIEW" == "ood" ]]; then
      EVAL_FAMILIES=("$FAM")
    else
      # shellcheck disable=SC2207
      EVAL_FAMILIES=($(other_families "$FAM"))
    fi

    # 1. Build the eval view once (unless present).
    if [[ "$FORCE_REGEN" == "1" || ! -f "${EVAL_DIR}/eval_input_anomaly.csv" ]]; then
      echo "building eval view -> ${EVAL_DIR} (families: ${EVAL_FAMILIES[*]})"
      uv run python scripts/make_eval_set.py \
        --splits-dir "$SPLITS_DIR" \
        --holdout-family "$FAM" \
        --eval-families "${EVAL_FAMILIES[@]}" \
        --n-anomaly-valid "$N_ANOMALY_VALID" \
        --n-anomaly-invalid "$N_ANOMALY_INVALID" \
        --seed "$SEED" \
        --out-dir "$EVAL_DIR"
    else
      echo "reusing eval view at ${EVAL_DIR}"
    fi

    OUT_DIR="${METRICS_ROOT}/${DATASET}/${RUN}_${FAM}/anomaly_lr/${VIEW}"
    mkdir -p "$OUT_DIR"
    echo "scoring holdout=${FAM} view=${VIEW} -> ${OUT_DIR}"

    ARGS=(
      --checkpoint "$CKPT"
      --splits-dir "$SPLITS_DIR"
      --holdout-family "$FAM"
      --eval-dir "$EVAL_DIR"
      --bg-ngram-n "$BG_NGRAM_N"
      --lam "$LAMBDA"
      --out "$OUT_DIR"
    )
    if [[ -n "${EXTRA_ARGS:-}" ]]; then
      # shellcheck disable=SC2206
      ARGS+=($EXTRA_ARGS)
    fi

    time "$RUNNER" uv run python scripts/eval_lstm_anomaly_lr.py "${ARGS[@]}"
    ls -lh "$OUT_DIR"
  done
done
