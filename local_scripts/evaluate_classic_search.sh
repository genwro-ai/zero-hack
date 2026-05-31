#!/usr/bin/env bash
set -euo pipefail

# Evaluate n-gram and VLMC classic baselines on fixed holdout eval sets.
# Defaults train on 20k records sampled from valid_s100k.

DATASET="${DATASET:-valid_s100k}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20000}"
HOLDOUT_FAMILIES="${HOLDOUT_FAMILIES:-mosfet igbt ic}"
MODELS="${MODELS:-ngram vlmc}"
N_VALUES="${N_VALUES:-3 5 7}"
ALPHA_VALUES="${ALPHA_VALUES:-0.2 0.4 0.8}"
VIEWS="${VIEWS:-id ood}"
TASKS="${TASKS:-next_step completion anomaly}"
GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
PREDS_ROOT="${PREDS_ROOT:-outputs/preds}"
METRICS_ROOT="${METRICS_ROOT:-outputs/metrics}"
SEED="${SEED:-1729}"
RANK_BY="${RANK_BY:-next_step_mrr}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage: local_scripts/evaluate_classic_search.sh [options]

Options:
  --dataset NAME             generated dataset label
  --train-samples INT        total train records sampled after holdout filtering
  --holdout-families "..."   space-separated holdout families
  --models "..."             space-separated models: ngram vlmc
  --n-values "..."           n-gram orders / VLMC depths
  --alpha-values "..."       n-gram backoff alphas
  --views "..."              id, ood
  --tasks "..."              next_step, completion, anomaly
  --rank-by METRIC           summary sort column
  --dry-run                  print command without running it
  -h, --help                 show this help

Environment variables with the same names are also supported:
  DATASET=valid_s100k
  TRAIN_SAMPLES=20000
  N_VALUES="3 5 7"
  ALPHA_VALUES="0.2 0.4 0.8"
  RANK_BY=next_step_mrr
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:?Missing value for --dataset}"
      shift 2
      ;;
    --train-samples)
      TRAIN_SAMPLES="${2:?Missing value for --train-samples}"
      shift 2
      ;;
    --holdout-families)
      HOLDOUT_FAMILIES="${2:?Missing value for --holdout-families}"
      shift 2
      ;;
    --models)
      MODELS="${2:?Missing value for --models}"
      shift 2
      ;;
    --n-values)
      N_VALUES="${2:?Missing value for --n-values}"
      shift 2
      ;;
    --alpha-values)
      ALPHA_VALUES="${2:?Missing value for --alpha-values}"
      shift 2
      ;;
    --views)
      VIEWS="${2:?Missing value for --views}"
      shift 2
      ;;
    --tasks)
      TASKS="${2:?Missing value for --tasks}"
      shift 2
      ;;
    --rank-by)
      RANK_BY="${2:?Missing value for --rank-by}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      printf "Unknown argument: %s\n\n" "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

args=(
  --generated-root "$GENERATED_ROOT"
  --eval-root "$EVAL_ROOT"
  --preds-root "$PREDS_ROOT"
  --metrics-root "$METRICS_ROOT"
  --datasets "$DATASET"
  --holdout-families $HOLDOUT_FAMILIES
  --models $MODELS
  --views $VIEWS
  --tasks $TASKS
  --train-samples "$TRAIN_SAMPLES"
  --n-values $N_VALUES
  --alpha-values $ALPHA_VALUES
  --rank-by "$RANK_BY"
  --seed "$SEED"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf "uv run python scripts/run_holdout_experiments.py"
  printf " %q" "${args[@]}"
  printf "\n"
  exit 0
fi

uv run python scripts/run_holdout_experiments.py "${args[@]}"
