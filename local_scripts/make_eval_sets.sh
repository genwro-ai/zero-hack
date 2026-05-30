#!/usr/bin/env bash
set -euo pipefail

# Create eval datasets locally under data/eval.
#
# This mirrors slurm/make_eval_sets.sbatch:
#   data/eval/<dataset>/holdout_mosfet/standard/id
#   data/eval/<dataset>/holdout_mosfet/standard/ood
#   data/eval/<dataset>/holdout_mosfet/diverse/id
#   data/eval/<dataset>/holdout_mosfet/diverse/ood
#
# Each holdout directory represents training on the other two families.
# The id view evaluates those two families; the ood view evaluates the holdout.

GENERATED_ROOT="${GENERATED_ROOT:-data/generated}"
EVAL_ROOT="${EVAL_ROOT:-data/eval}"
DATASETS="${DATASETS:-mixed_s005k_v40_a60 mixed_s010k_v40_a60 mixed_s020k_v40_a60 mixed_s100k_v40_a60 mixed_s500k_v40_a60 mixed_s1000k_v40_a60}"
EVAL_SOURCES="${EVAL_SOURCES:-test_standard test_diverse}"
HOLDOUT_FAMILIES="${HOLDOUT_FAMILIES:-mosfet igbt ic}"
N_VALID="${N_VALID:-100}"
N_ANOMALY_VALID="${N_ANOMALY_VALID:-200}"
N_ANOMALY_INVALID="${N_ANOMALY_INVALID:-129}"
EVAL_SEED="${EVAL_SEED:-1729}"
LIMIT_PER_FAMILY="${LIMIT_PER_FAMILY:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage: local_scripts/make_eval_sets.sh [options]

Options:
  --datasets "..."          space-separated dataset labels
  --eval-sources "..."      test/test_standard/test_diverse sources
  --holdout-families "..."  space-separated holdout families
  --generated-root PATH     generated-data root
  --eval-root PATH          eval-data output root
  --n-valid INT             sequences/family for Tasks 1&2
  --n-anomaly-valid INT     valid sequences/family for Task 3
  --n-anomaly-invalid INT   invalid sequences/family for Task 3
  --seed INT                eval-set seed
  --limit-per-family INT    optional smoke-test limit
  --dry-run                 print commands without running them
  -h, --help                show this help

Environment variables with the same names are also supported:
  GENERATED_ROOT=data/generated
  EVAL_ROOT=data/eval
  DATASETS="mixed_s005k_v40_a60 mixed_s010k_v40_a60"
  EVAL_SOURCES="test_standard test_diverse"
  HOLDOUT_FAMILIES="mosfet igbt ic"
  N_VALID=100
  N_ANOMALY_VALID=200
  N_ANOMALY_INVALID=129
  EVAL_SEED=1729
  LIMIT_PER_FAMILY=2
  DRY_RUN=1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      DATASETS="${2:?Missing value for --datasets}"
      shift 2
      ;;
    --holdout-families)
      HOLDOUT_FAMILIES="${2:?Missing value for --holdout-families}"
      shift 2
      ;;
    --eval-sources)
      EVAL_SOURCES="${2:?Missing value for --eval-sources}"
      shift 2
      ;;
    --generated-root)
      GENERATED_ROOT="${2:?Missing value for --generated-root}"
      shift 2
      ;;
    --eval-root)
      EVAL_ROOT="${2:?Missing value for --eval-root}"
      shift 2
      ;;
    --n-valid)
      N_VALID="${2:?Missing value for --n-valid}"
      shift 2
      ;;
    --n-anomaly-valid)
      N_ANOMALY_VALID="${2:?Missing value for --n-anomaly-valid}"
      shift 2
      ;;
    --n-anomaly-invalid)
      N_ANOMALY_INVALID="${2:?Missing value for --n-anomaly-invalid}"
      shift 2
      ;;
    --seed)
      EVAL_SEED="${2:?Missing value for --seed}"
      shift 2
      ;;
    --limit-per-family)
      LIMIT_PER_FAMILY="${2:?Missing value for --limit-per-family}"
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
  --out-root "$EVAL_ROOT"
  --datasets $DATASETS
  --eval-sources $EVAL_SOURCES
  --holdout-families $HOLDOUT_FAMILIES
  --n-valid "$N_VALID"
  --n-anomaly-valid "$N_ANOMALY_VALID"
  --n-anomaly-invalid "$N_ANOMALY_INVALID"
  --seed "$EVAL_SEED"
)

if [[ -n "$LIMIT_PER_FAMILY" ]]; then
  args+=(--limit-per-family "$LIMIT_PER_FAMILY")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

uv run python scripts/make_all_eval_sets.py "${args[@]}"
