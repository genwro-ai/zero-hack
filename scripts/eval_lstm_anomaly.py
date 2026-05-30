"""Evaluate a trained LSTM on anomaly detection (Task 3) via sequence likelihood.

The LSTM is trained only on *valid* sequences, so a rule-violating sequence
should receive a lower average per-step log-probability. We score every example
in an anomaly eval set with ``LSTMInference.score_sequence`` and ask whether
that likelihood signal separates valid from invalid sequences.

Because the binary accuracy/precision/recall depend on an arbitrary log-prob
threshold, the headline metric is **ROC-AUC** (threshold-free). We additionally
sweep the threshold to report the best-F1 operating point, and run the
rule-validator as an oracle baseline for comparison (learned vs. rule-based).

Inputs come from ``scripts/make_eval_set.py`` (``eval_input_anomaly.csv`` +
``anomaly_truth.csv``).

Usage:
    uv run python scripts/eval_lstm_anomaly.py \
        --checkpoint outputs/models/valid_s005k/lstm_scheduled_sampling/best.pt \
        --eval-dir data/eval/valid_s005k \
        --out outputs/metrics/valid_s005k/lstm_scheduled_sampling/anomaly
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from zero_hack.eval.anomaly import score_anomaly
from zero_hack.eval.io import read_anomaly_truth, read_eval_input_anomaly
from zero_hack.eval.validator import first_violated_rule, validate_sequence
from zero_hack.models.lstm.inference import load_lstm_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the LSTM checkpoint (.pt).")
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Dir with eval_input_anomaly.csv and anomaly_truth.csv (from make_eval_set.py).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed avg-logprob threshold (valid if >=). Omit to sweep for best F1.",
    )
    parser.add_argument(
        "--sweep-points",
        type=int,
        default=100,
        help="Number of candidate thresholds when sweeping.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True, help="Output dir for results.{json,md}.")
    return parser.parse_args()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _predictions_at(
    threshold: float,
    avg_logprobs: dict[str, float],
    sequences: dict[str, list[str]],
) -> dict[str, dict]:
    """Likelihood-method predictions at a given threshold.

    score = P(valid) = sigmoid(avg_logprob - threshold); is_valid if avg >= thr.
    A flagged anomaly's rule is attributed by the validator (the model only
    supplies the binary/score signal).
    """
    preds: dict[str, dict] = {}
    for example_id, avg in avg_logprobs.items():
        valid = avg >= threshold
        preds[example_id] = {
            "is_valid": int(valid),
            "score": _sigmoid(avg - threshold),
            "predicted_rule": None
            if valid
            else (first_violated_rule(sequences[example_id]) or "RULE_DEP_NO_CLEAN"),
        }
    return preds


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    model = load_lstm_checkpoint(args.checkpoint, device=args.device or "cpu")
    print(f"loaded {args.checkpoint} (meta: {model.meta})")

    examples = read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
    truth = read_anomaly_truth(eval_dir / "anomaly_truth.csv")
    families = {ex["example_id"]: ex["family"] for ex in examples}
    sequences = {ex["example_id"]: ex["sequence"] for ex in examples}

    # One forward pass per sequence; cache the average per-step log-prob.
    avg_logprobs: dict[str, float] = {}
    for ex in examples:
        seq = ex["sequence"]
        if not seq:
            continue
        avg_logprobs[ex["example_id"]] = model.score_sequence(ex["family"], seq) / len(seq)
    print(f"scored {len(avg_logprobs)} sequences")

    ids = [eid for eid in avg_logprobs if eid in truth]
    if not ids:
        raise SystemExit("No overlapping example ids between input and truth.")

    # ROC-AUC is threshold-free; compute it from predictions at any threshold.
    auc_preds = _predictions_at(0.0, avg_logprobs, sequences)
    roc_auc = score_anomaly(truth, auc_preds, families)["all"]["roc_auc"]

    # Threshold sweep for the best-F1 operating point.
    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        lo, hi = min(avg_logprobs.values()), max(avg_logprobs.values())
        if hi <= lo:
            thresholds = [lo]
        else:
            step = (hi - lo) / max(1, args.sweep_points - 1)
            thresholds = [lo + i * step for i in range(args.sweep_points)]

    best = None
    for thr in thresholds:
        metrics = score_anomaly(truth, _predictions_at(thr, avg_logprobs, sequences), families)
        f1 = metrics["all"]["f1"]
        if best is None or f1 > best["threshold_f1"]:
            best = {"threshold": thr, "threshold_f1": f1, "metrics": metrics}

    # Oracle baseline: the rule validator decides validity directly.
    validator_preds = {
        eid: {
            "is_valid": int(not validate_sequence(sequences[eid])),
            "score": None,
            "predicted_rule": (
                None
                if not validate_sequence(sequences[eid])
                else first_violated_rule(sequences[eid])
            ),
        }
        for eid in ids
    }
    validator_metrics = score_anomaly(truth, validator_preds, families)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "meta": model.meta,
        "eval_dir": str(eval_dir),
        "n": len(ids),
        "roc_auc": roc_auc,
        "best_threshold": best["threshold"],
        "likelihood_metrics_at_best_threshold": best["metrics"],
        "validator_baseline_metrics": validator_metrics,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = _render_md(args.checkpoint, roc_auc, best, validator_metrics, len(ids))
    (out_dir / "results.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"wrote {out_dir / 'results.json'}")
    print(f"wrote {out_dir / 'results.md'}")


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.4f}"


def _row(name: str, m: dict) -> str:
    return (
        f"| {name} | {m['n']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
        f"{m['recall']:.4f} | {m['f1']:.4f} | {_fmt(m['roc_auc'])} |"
    )


def _render_md(checkpoint: str, roc_auc, best: dict, validator: dict, n: int) -> str:
    likelihood = best["metrics"]["all"]
    lines = [
        "# LSTM - anomaly detection (likelihood)",
        "",
        f"Checkpoint: `{checkpoint}`",
        f"Examples: {n} - **ROC-AUC (threshold-free): {_fmt(roc_auc)}**",
        f"Best-F1 threshold (avg logprob): {best['threshold']:.4f}",
        "",
        "| Method | n | accuracy | precision | recall | f1 | roc_auc |",
        "|---|---|---|---|---|---|---|",
        _row("LSTM likelihood @ best thr", likelihood),
        _row("validator (oracle baseline)", validator["all"]),
        "",
        "## LSTM likelihood, per family (at best threshold)",
        "",
        "| Family | n | accuracy | precision | recall | f1 | roc_auc |",
        "|---|---|---|---|---|---|---|",
    ]
    for group, m in best["metrics"].items():
        if group != "all":
            lines.append(_row(group, m))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
