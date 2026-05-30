#!/usr/bin/env python3
"""Build local eval inputs and ground truth from held-out sequences."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.eval.anomaly_synth import corrupt_steps
from zero_hack.eval.io import join_steps
from zero_hack.models.common import DEFAULT_SPLITS_DIR, FAMILIES, load_split_records


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    parser.add_argument(
        "--eval-families",
        nargs="+",
        choices=FAMILIES,
        default=None,
        help="Family test splits to include. Defaults to all families.",
    )
    parser.add_argument("--n-valid", type=int, default=100, help="Sequences/family for Tasks 1&2.")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.6, 0.8])
    parser.add_argument(
        "--n-anomaly-valid",
        type=int,
        default=200,
        help="Valid sequences/family for Task 3.",
    )
    parser.add_argument(
        "--n-anomaly-invalid",
        type=int,
        default=129,
        help="Invalid sequences/family for Task 3.",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "eval" / "default"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    bundle = load_split_records(
        args.splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    eval_families = tuple(args.eval_families or FAMILIES)
    if args.eval_families:
        split = "family_tests"
        records = []
        for family in eval_families:
            records.extend(bundle.records[f"test_{family}"])
    else:
        split = args.split
        records = bundle.records[split]
    by_family: dict[str, list] = {}
    for rec in records:
        by_family.setdefault(rec.family, []).append(rec)
    for fam in by_family:
        rng.shuffle(by_family[fam])

    valid_rows: list[list] = []
    nextstep_truth: list[list] = []
    completion_truth: list[list] = []
    anomaly_rows: list[list] = []
    anomaly_truth: list[list] = []

    for family, recs in sorted(by_family.items()):
        for rec in recs[: args.n_valid]:
            steps = list(rec.steps)
            for frac in args.fractions:
                cut = int(len(steps) * frac)
                cut = max(1, min(cut, len(steps) - 1))
                example_id = f"{family}_{rec.sequence_id}_f{int(frac * 100)}"
                valid_rows.append([example_id, family, frac, join_steps(steps[:cut])])
                nextstep_truth.append([example_id, steps[cut]])
                completion_truth.append([example_id, join_steps(steps[cut:])])

    for family, recs in sorted(by_family.items()):
        kept_valid = kept_invalid = 0
        for rec in recs:
            if kept_valid >= args.n_anomaly_valid and kept_invalid >= args.n_anomaly_invalid:
                break
            steps = list(rec.steps)
            if kept_invalid < args.n_anomaly_invalid:
                corrupted = corrupt_steps(steps, rng)
                if corrupted is None:
                    continue
                seq, rule = corrupted
                example_id = f"{family}_{rec.sequence_id}_bad"
                anomaly_rows.append([example_id, family, join_steps(seq)])
                anomaly_truth.append([example_id, 0, rule])
                kept_invalid += 1
            elif kept_valid < args.n_anomaly_valid:
                example_id = f"{family}_{rec.sequence_id}_ok"
                anomaly_rows.append([example_id, family, join_steps(steps)])
                anomaly_truth.append([example_id, 1, ""])
                kept_valid += 1

    order = list(range(len(anomaly_rows)))
    rng.shuffle(order)
    anomaly_rows = [anomaly_rows[i] for i in order]
    anomaly_truth = [anomaly_truth[i] for i in order]

    _write(
        out_dir / "eval_input_valid.csv",
        ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"],
        valid_rows,
    )
    _write(out_dir / "nextstep_truth.csv", ["EXAMPLE_ID", "NEXT_STEP"], nextstep_truth)
    _write(out_dir / "completion_truth.csv", ["EXAMPLE_ID", "TRUE_SEQUENCE"], completion_truth)
    _write(out_dir / "eval_input_anomaly.csv", ["EXAMPLE_ID", "FAMILY", "SEQUENCE"], anomaly_rows)
    _write(out_dir / "anomaly_truth.csv", ["EXAMPLE_ID", "IS_VALID", "RULE"], anomaly_truth)
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "splits_dir": str(Path(args.splits_dir)),
                "split": split,
                "holdout_family": args.holdout_family,
                "train_families": list(bundle.train_families),
                "eval_families": list(eval_families),
                "evaluated_families": sorted(by_family),
                "n_valid_per_family": args.n_valid,
                "completion_fractions": args.fractions,
                "n_anomaly_valid_per_family": args.n_anomaly_valid,
                "n_anomaly_invalid_per_family": args.n_anomaly_invalid,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    n_bad = sum(1 for r in anomaly_truth if r[1] == 0)
    print(f"counts: {bundle.counts()}")
    print(f"wrote {len(valid_rows)} valid rows (Tasks 1&2) to {out_dir}")
    print(
        f"wrote {len(anomaly_rows)} anomaly rows (Task 3): {n_bad} invalid / "
        f"{len(anomaly_rows) - n_bad} valid"
    )


if __name__ == "__main__":
    main()
