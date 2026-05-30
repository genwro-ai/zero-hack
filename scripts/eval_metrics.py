#!/usr/bin/env python3
"""Score Industrial AI submission files against ground truth.

Local, dependency-free stand-in for the organizer ``eval_metrics.py``, matching
the protocol and submission formats in ``data/industrial/generation_rules.md``
§5. When the official scorer is distributed, prefer it; this mirrors its CLI so
the two are interchangeable for self-evaluation.

Examples
--------
    uv run python scripts/eval_metrics.py --task next_step \\
        --ground-truth data/eval/valid_s005k/holdout_ic/ood/nextstep_truth.csv \\
        --predictions  outputs/preds/valid_s005k/holdout_ic/ood/ngram/nextstep.csv \\
        --eval-input   data/eval/valid_s005k/holdout_ic/ood/eval_input_valid.csv

    uv run python scripts/eval_metrics.py --task anomaly \\
        --ground-truth data/eval/valid_s005k/holdout_ic/ood/anomaly_truth.csv \\
        --predictions  outputs/preds/valid_s005k/holdout_ic/ood/ngram/anomaly.csv \\
        --eval-input   data/eval/valid_s005k/holdout_ic/ood/eval_input_anomaly.csv
"""

from __future__ import annotations

import argparse
import json

from zero_hack.eval.score import TASKS, score_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--ground-truth", required=True, help="Ground-truth CSV path.")
    parser.add_argument("--predictions", required=True, help="Submission CSV path.")
    parser.add_argument(
        "--eval-input",
        default=None,
        help="Eval input CSV, used for per-family breakdown.",
    )
    parser.add_argument(
        "--valid-supplement",
        default=None,
        help="[anomaly only] Valid examples to add when using organizer forbidden-only truth.",
    )
    parser.add_argument("--json", default=None, help="Optional path to also write metrics as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = score_task(
        args.task,
        ground_truth=args.ground_truth,
        predictions=args.predictions,
        eval_input=args.eval_input,
        valid_supplement=args.valid_supplement,
    )
    text = json.dumps(metrics, indent=2)
    print(f"== {args.task} ==")
    print(text)
    if args.json:
        from pathlib import Path

        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
