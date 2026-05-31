#!/usr/bin/env bash
# Transformer before/after-GRPO completion comparison (decisive cut: E1+E2+E3).
#
# Per holdout family, runs five evals into outputs/metrics/<dataset>/transformer_holdout_<F>/:
#   sampled_base        base   T=1.0  (E1/E3 "before")
#   sampled_grpo        grpo   T=1.0  (E1/E3 "after")
#   sampled_base_masked base   T=1.0 + --enforce-rules   (E2 "free validity" control)
#   greedy_base         base   greedy (deployment regime, before)
#   greedy_grpo         grpo   greedy (deployment regime, after)
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-valid_s005k}
MODEL_ROOT=outputs/models/$DATASET
METRICS_ROOT=outputs/metrics/$DATASET
MAXEX=${MAXEX:-100}
FRACTIONS=${FRACTIONS:-0.6 0.8}
TEMP=${TEMP:-1.0}
SAMPLES=${SAMPLES:-4}

run() {  # ckpt holdout outdir extra...
  local ckpt=$1 holdout=$2 out=$3; shift 3
  echo "### $out"
  uv run python scripts/eval_completion.py \
    --checkpoint "$ckpt" --holdout-family "$holdout" \
    --fractions $FRACTIONS --max-examples-per-family "$MAXEX" \
    --out "$out" "$@" >/dev/null 2>>outputs/metrics/.matrix.log
}

for F in ic igbt mosfet; do
  D=$MODEL_ROOT/transformer_holdout_$F
  M=$METRICS_ROOT/transformer_holdout_$F
  run "$D/best.pt"      "$F" "$M/sampled_base"        --temperature "$TEMP" --samples "$SAMPLES"
  run "$D/best_grpo.pt" "$F" "$M/sampled_grpo"        --temperature "$TEMP" --samples "$SAMPLES"
  run "$D/best.pt"      "$F" "$M/sampled_base_masked" --temperature "$TEMP" --samples "$SAMPLES" --enforce-rules
  run "$D/best.pt"      "$F" "$M/greedy_base"
  run "$D/best_grpo.pt" "$F" "$M/greedy_grpo"
done
echo "DONE"
