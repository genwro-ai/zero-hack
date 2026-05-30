#!/usr/bin/env python3
"""Generate Task 1/2/3 prediction files from baseline models."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.eval import io
from zero_hack.eval.validator import first_violated_rule, validate_sequence
from zero_hack.models.common import DEFAULT_RAW_DIR, load_record_splits
from zero_hack.models.most_frequent import MostFrequentModel
from zero_hack.models.ngram import NGramModel
from zero_hack.models.xgboost import XGBoostNextStep

MAX_COMPLETION = 400
TERMINATOR = "SHIP LOT"


def build_model(
    name: str, train_records: list, *, n: int, alpha: float, bucket: int, xgb_kwargs: dict
):
    if name == "ngram":
        return NGramModel(n=n, backoff_alpha=alpha).fit(train_records)
    if name == "most_frequent":
        return MostFrequentModel(position_bucket_size=bucket, backoff_alpha=alpha).fit(
            train_records
        )
    if name == "xgboost":
        return XGBoostNextStep(**xgb_kwargs).fit(train_records)
    raise ValueError(f"Unknown model {name!r}")


def complete_sequence(model, family: str, prefix: list[str]) -> list[str]:
    """Greedily extend ``prefix`` until SHIP LOT or the length cap; return suffix."""
    seq = list(prefix)
    produced: list[str] = []
    while len(seq) < MAX_COMPLETION:
        topk = model.predict_topk(family, seq, k=1)
        if not topk:
            break
        nxt = topk[0]
        seq.append(nxt)
        produced.append(nxt)
        if nxt == TERMINATOR:
            break
    return produced


def predict_anomaly(model, family: str, sequence: list[str], method: str, threshold: float) -> dict:
    if method == "validator":
        violations = validate_sequence(sequence)
        valid = not violations
        return {
            "is_valid": int(valid),
            "score": 1.0 if valid else 0.0,
            "predicted_rule": None if valid else first_violated_rule(sequence),
        }
    # likelihood: mean per-step log-prob -> validity score.
    n = max(1, len(sequence))
    avg_logprob = model.score_sequence(family, sequence) / n
    # Squash to (0,1): higher avg log-prob -> more likely valid.
    score = 1.0 / (1.0 + math.exp(-(avg_logprob - threshold)))
    valid = avg_logprob >= threshold
    return {
        "is_valid": int(valid),
        "score": round(score, 6),
        "predicted_rule": None if valid else (first_violated_rule(sequence) or "RULE_DEP_NO_CLEAN"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="ngram", choices=("ngram", "most_frequent", "xgboost"))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--industrial-dir", default=None)
    parser.add_argument("--no-include-industrial", action="store_true")
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    parser.add_argument("--eval-dir", default=str(PROJECT_ROOT / "outputs" / "eval"))
    parser.add_argument("--out-dir", default=None, help="Default: outputs/preds/<model>.")
    parser.add_argument("--tasks", nargs="+", default=["next_step", "completion", "anomaly"])
    parser.add_argument(
        "--anomaly-method", default="validator", choices=("validator", "ngram", "likelihood")
    )
    parser.add_argument("--anomaly-threshold", type=float, default=-1.0)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--bucket", type=int, default=5)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-depth", type=int, default=8)
    parser.add_argument("--xgb-lr", type=float, default=0.3)
    parser.add_argument("--xgb-lag", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = (
        Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "outputs" / "preds" / args.model
    )

    bundle = load_record_splits(
        args.raw_dir,
        industrial_dir=args.industrial_dir,
        include_industrial=not args.no_include_industrial,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")
    model = build_model(
        args.model,
        bundle.records["train"],
        n=args.n,
        alpha=args.alpha,
        bucket=args.bucket,
        xgb_kwargs={
            "n_estimators": args.xgb_estimators,
            "max_depth": args.xgb_depth,
            "learning_rate": args.xgb_lr,
            "lag": args.xgb_lag,
        },
    )

    if "next_step" in args.tasks or "completion" in args.tasks:
        valid_inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")

        if "next_step" in args.tasks:
            rows = [
                {
                    "example_id": r["example_id"],
                    "ranks": model.predict_topk(r["family"], r["partial_sequence"], k=5),
                }
                for r in valid_inputs
            ]
            io.write_next_step_predictions(out_dir / "nextstep.csv", rows)
            print(f"wrote {out_dir / 'nextstep.csv'} ({len(rows)} rows)")

        if "completion" in args.tasks:
            rows = [
                {
                    "example_id": r["example_id"],
                    "steps": complete_sequence(model, r["family"], r["partial_sequence"]),
                }
                for r in valid_inputs
            ]
            io.write_completion_predictions(out_dir / "completion.csv", rows)
            print(f"wrote {out_dir / 'completion.csv'} ({len(rows)} rows)")

    if "anomaly" in args.tasks:
        anomaly_inputs = io.read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
        rows = [
            {
                "example_id": r["example_id"],
                **predict_anomaly(
                    model, r["family"], r["sequence"], args.anomaly_method, args.anomaly_threshold
                ),
            }
            for r in anomaly_inputs
        ]
        io.write_anomaly_predictions(out_dir / "anomaly.csv", rows)
        print(f"wrote {out_dir / 'anomaly.csv'} ({len(rows)} rows, method={args.anomaly_method})")


if __name__ == "__main__":
    main()
