"""Likelihood-ratio anomaly detection (Task 3) in the holdout ("handoff") setting.

Raw sequence log-likelihood is a biased anomaly signal: it is dominated by the
population-level *background* frequency of the individual steps rather than by
whether their *ordering* is legal (Ren et al., NeurIPS 2019). A sequence of
common-but-misordered steps still scores high; a sequence of rare-but-legal
steps scores low. The fix is a likelihood *ratio* against a background model that
captures only the nuisance statistics:

    score(x) = logp_model(x) - lambda * logp_background(x)

We use a *low-order* n-gram as the background (it sees local co-occurrence but
not long-range structure), so the residual isolates exactly the long-range /
compositional structure the neural model learned. Anomalies that violate that
structure while looking locally plausible make ``logp_model`` crash while
``logp_background`` stays high, so the ratio drops sharply.

This script runs the comparison so we can see if the ratio actually helps:

    model_likelihood    raw avg log-prob (current eval_lstm_anomaly.py baseline)
    ngram_likelihood    background n-gram avg log-prob (reference)
    likelihood_ratio    logp_model - lambda * logp_ngram (the hypothesis)
    likelihood_ratio_z  z(logp_model) - z(logp_ngram)   (scale-free; no lambda)
    validator           rule oracle (upper bound, not a learned signal)

It is designed for the holdout/LOFO setting: the neural checkpoint and the
background n-gram are *both* trained with ``--holdout-family`` removed, and the
eval set is one view (``id`` = the two trained families, ``ood`` = the held-out
family). The headline metric is **ROC-AUC** (threshold-free); the best-F1
operating point is swept per detector and is comparable across detectors but
optimistic in absolute terms (it peeks at the eval labels).

Usage:
    uv run python scripts/eval_lstm_anomaly_lr.py \
        --checkpoint outputs/models/valid_s005k/lstm_holdout_ic/best.pt \
        --splits-dir data/generated/valid_s005k/splits \
        --holdout-family ic \
        --eval-dir data/eval/valid_s005k/holdout_ic/ood \
        --out outputs/metrics/valid_s005k/lstm_holdout_ic/anomaly_lr/ood
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from zero_hack.eval.anomaly import _roc_auc, score_anomaly
from zero_hack.eval.io import read_anomaly_truth, read_eval_input_anomaly
from zero_hack.eval.validator import first_violated_rule, validate_sequence
from zero_hack.models.common import load_split_records
from zero_hack.models.lstm.inference import load_lstm_checkpoint
from zero_hack.models.ngram.model import NGramModel

# Detectors are reported in this order; each maps an example id -> a scalar where
# HIGHER means MORE VALID (so anomalies sit at the low end).
_DETECTOR_ORDER = (
    "model_likelihood",
    "ngram_likelihood",
    "likelihood_ratio",
    "likelihood_ratio_z",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="LOFO-trained LSTM checkpoint (.pt).")
    parser.add_argument(
        "--splits-dir",
        default="data/generated/valid_s005k/splits",
        help="Training splits dir; the background n-gram is fit on these (minus holdout).",
    )
    parser.add_argument(
        "--holdout-family",
        choices=["mosfet", "igbt", "ic"],
        default=None,
        help="Family removed from the n-gram training set, to match the checkpoint's LOFO split.",
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Dir with eval_input_anomaly.csv and anomaly_truth.csv (one holdout view).",
    )
    parser.add_argument(
        "--bg-ngram-n",
        type=int,
        default=1,
        help=(
            "Background n-gram order. 1 (unigram) = pure token-frequency nuisance, "
            "the only order empirically found to help the ratio: any order >=2 also "
            "captures the local ordering signal and subtracting it hurts."
        ),
    )
    parser.add_argument("--bg-alpha", type=float, default=0.4, help="N-gram stupid-backoff weight.")
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Weight on the background log-prob in the (raw) likelihood ratio.",
    )
    parser.add_argument("--sweep-points", type=int, default=100, help="Thresholds in the F1 sweep.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True, help="Output dir for results.{json,md}.")
    return parser.parse_args()


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _zscore(values: dict[str, float]) -> dict[str, float]:
    """Standardise scores to zero mean / unit std (no labels used).

    Makes ``logp_model`` and ``logp_ngram`` directly subtractable despite living
    on different scales, so the ratio does not depend on an arbitrary ``lambda``.
    Falls back to centring-only when the spread is degenerate.
    """
    vals = list(values.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    if std <= 0.0:
        return {k: 0.0 for k in values}
    return {k: (v - mean) / std for k, v in values.items()}


def _predictions_at(
    threshold: float,
    scores: dict[str, float],
    sequences: dict[str, list[str]],
) -> dict[str, dict]:
    """Binary predictions for one detector at ``threshold`` (valid if score >= thr).

    Rule attribution on a flagged anomaly still comes from the validator: the
    detector supplies only the binary/continuous validity signal, exactly like
    the existing likelihood eval.
    """
    preds: dict[str, dict] = {}
    for example_id, score in scores.items():
        valid = score >= threshold
        preds[example_id] = {
            "is_valid": int(valid),
            "score": _sigmoid(score - threshold),
            "predicted_rule": None
            if valid
            else (first_violated_rule(sequences[example_id]) or "RULE_DEP_NO_CLEAN"),
        }
    return preds


def _roc_per_group(
    scores: dict[str, float],
    truth: dict[str, dict],
    families: dict[str, str],
    ids: list[str],
) -> dict[str, float | None]:
    """ROC-AUC of one detector, overall and per family.

    Computed straight from the raw scores (anomaly score = ``-validity_score``)
    so it is threshold-free and immune to sigmoid saturation. Positive class is
    anomaly (IS_VALID == 0).
    """
    groups: dict[str, list[str]] = {"all": list(ids)}
    for example_id in ids:
        groups.setdefault(families.get(example_id, "unknown"), []).append(example_id)

    out: dict[str, float | None] = {}
    for group, group_ids in groups.items():
        anomaly_scores = [-scores[i] for i in group_ids]
        labels = [1 if truth[i]["is_valid"] == 0 else 0 for i in group_ids]
        out[group] = _roc_auc(anomaly_scores, labels)
    return out


def _best_f1(
    scores: dict[str, float],
    truth: dict[str, dict],
    sequences: dict[str, list[str]],
    families: dict[str, str],
    sweep_points: int,
) -> dict:
    """Sweep the threshold and keep the best-F1 operating point for one detector."""
    lo, hi = min(scores.values()), max(scores.values())
    if hi <= lo:
        thresholds = [lo]
    else:
        step = (hi - lo) / max(1, sweep_points - 1)
        thresholds = [lo + i * step for i in range(sweep_points)]

    best: dict | None = None
    for thr in thresholds:
        metrics = score_anomaly(truth, _predictions_at(thr, scores, sequences), families)
        f1 = metrics["all"]["f1"]
        if best is None or f1 > best["f1"]:
            best = {"threshold": thr, "f1": f1, "metrics": metrics}
    assert best is not None
    return best


def _evaluate_detector(
    scores: dict[str, float],
    truth: dict[str, dict],
    sequences: dict[str, list[str]],
    families: dict[str, str],
    ids: list[str],
    sweep_points: int,
) -> dict:
    roc = _roc_per_group(scores, truth, families, ids)
    best = _best_f1(scores, truth, sequences, families, sweep_points)
    return {
        "roc_auc": roc["all"],
        "roc_auc_per_family": {k: v for k, v in roc.items() if k != "all"},
        "best_threshold": best["threshold"],
        "metrics_at_best_f1": best["metrics"]["all"],
    }


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)

    model = load_lstm_checkpoint(args.checkpoint, device=args.device or "cpu")
    print(f"loaded {args.checkpoint} (meta: {model.meta})")

    # Background n-gram on the SAME training data the model saw (minus holdout).
    bundle = load_split_records(args.splits_dir, holdout_family=args.holdout_family)
    train_records = bundle.records["train"]
    background = NGramModel(n=args.bg_ngram_n, backoff_alpha=args.bg_alpha).fit(train_records)
    print(
        f"fit background {args.bg_ngram_n}-gram on {len(train_records)} sequences "
        f"(holdout={args.holdout_family}, train_families={list(bundle.train_families)})"
    )

    examples = read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
    truth = read_anomaly_truth(eval_dir / "anomaly_truth.csv")
    families = {ex["example_id"]: ex["family"] for ex in examples}
    sequences = {ex["example_id"]: ex["sequence"] for ex in examples}

    # One model + one n-gram pass per sequence; cache average per-step log-probs.
    model_avg: dict[str, float] = {}
    ngram_avg: dict[str, float] = {}
    for ex in examples:
        seq = ex["sequence"]
        if not seq:
            continue
        eid = ex["example_id"]
        model_avg[eid] = model.score_sequence(ex["family"], seq) / len(seq)
        ngram_avg[eid] = background.score_sequence(ex["family"], seq) / len(seq)

    ids = [eid for eid in model_avg if eid in truth]
    if not ids:
        raise SystemExit("No overlapping example ids between input and truth.")
    print(f"scored {len(ids)} sequences")

    # Build the four detector score maps (higher = more valid).
    model_z = _zscore(model_avg)
    ngram_z = _zscore(ngram_avg)
    detectors: dict[str, dict[str, float]] = {
        "model_likelihood": model_avg,
        "ngram_likelihood": ngram_avg,
        "likelihood_ratio": {eid: model_avg[eid] - args.lam * ngram_avg[eid] for eid in model_avg},
        "likelihood_ratio_z": {eid: model_z[eid] - ngram_z[eid] for eid in model_avg},
    }

    results = {
        name: _evaluate_detector(
            detectors[name], truth, sequences, families, ids, args.sweep_points
        )
        for name in _DETECTOR_ORDER
    }

    # Validator oracle baseline (rule-based upper bound, not a learned signal).
    validator_preds = {
        eid: {
            "is_valid": int(not validate_sequence(sequences[eid])),
            "score": None,
            "predicted_rule": first_violated_rule(sequences[eid]),
        }
        for eid in ids
    }
    validator_metrics = score_anomaly(truth, validator_preds, families)["all"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "meta": model.meta,
        "eval_dir": str(eval_dir),
        "holdout_family": args.holdout_family,
        "background_ngram_n": args.bg_ngram_n,
        "lambda": args.lam,
        "n": len(ids),
        "detectors": results,
        "validator_baseline_metrics": validator_metrics,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = _render_md(args, results, validator_metrics, len(ids), sorted({families[i] for i in ids}))
    (out_dir / "results.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"wrote {out_dir / 'results.json'}")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _render_md(
    args: argparse.Namespace,
    results: dict,
    validator: dict,
    n: int,
    fams: list[str],
) -> str:
    lines = [
        "# LSTM — anomaly detection: likelihood ratio vs raw likelihood",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Eval: `{args.eval_dir}` · holdout=`{args.holdout_family}` · examples: {n}",
        f"Background: {args.bg_ngram_n}-gram · lambda={args.lam}",
        "",
        "Headline metric is **ROC-AUC** (threshold-free). best-F1 is swept per "
        "detector (comparable across rows, optimistic in absolute terms).",
        "",
        "| Detector | ROC-AUC | best-F1 | precision | recall | accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for name in _DETECTOR_ORDER:
        r = results[name]
        m = r["metrics_at_best_f1"]
        lines.append(
            f"| {name} | {_fmt(r['roc_auc'])} | {m['f1']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['accuracy']:.4f} |"
        )
    lines.append(
        f"| validator (oracle) | - | {validator['f1']:.4f} | "
        f"{validator['precision']:.4f} | {validator['recall']:.4f} | {validator['accuracy']:.4f} |"
    )

    if len(fams) > 1:
        lines += [
            "",
            "## ROC-AUC per family",
            "",
            "| Detector | " + " | ".join(fams) + " |",
            "|---|" + "|".join(["---"] * len(fams)) + "|",
        ]
        for name in _DETECTOR_ORDER:
            per_fam = results[name]["roc_auc_per_family"]
            cells = " | ".join(_fmt(per_fam.get(f)) for f in fams)
            lines.append(f"| {name} | {cells} |")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
