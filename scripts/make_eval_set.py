#!/usr/bin/env python3
"""Build local eval inputs and ground truth from held-out sequences."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.eval.io import join_steps
from zero_hack.eval.validator import first_violated_rule, is_valid
from zero_hack.models.common import DEFAULT_SPLITS_DIR, family_test_split, load_split_records

_CLEAN_HINTS = ("CLEAN", "RCA", "HF DIP", "RINSE", "DRY WAFER")


def _corrupt(
    steps: list[str], rng: random.Random, max_tries: int = 12
) -> tuple[list[str], str] | None:
    """Return (corrupted_steps, first_rule) that the validator flags, or None."""
    n = len(steps)
    for _ in range(max_tries):
        op = rng.choice(("drop_clean", "drop_develop", "ship_early", "test_early", "swap"))
        seq = list(steps)

        if op == "drop_clean":
            idxs = [i for i, s in enumerate(seq) if any(h in s for h in _CLEAN_HINTS)]
            if idxs:
                del seq[rng.choice(idxs)]
        elif op == "drop_develop":
            idxs = [i for i, s in enumerate(seq) if s.startswith("DEVELOP")]
            if idxs:
                del seq[rng.choice(idxs)]
        elif op == "ship_early" and "SHIP LOT" in seq:
            seq.remove("SHIP LOT")
            seq.insert(rng.randint(0, max(0, len(seq) // 3)), "SHIP LOT")
        elif op == "test_early":
            idxs = [i for i, s in enumerate(seq) if s.endswith("TEST") and "WAFER SORT" not in s]
            if idxs:
                step = seq.pop(rng.choice(idxs))
                seq.insert(rng.randint(0, max(0, len(seq) // 4)), step)
        elif op == "swap" and n > 6:
            i = rng.randint(0, n - 2)
            j = rng.randint(0, n - 2)
            seq[i], seq[j] = seq[j], seq[i]

        if seq != steps and not is_valid(seq):
            return seq, first_violated_rule(seq) or "UNKNOWN"
    return None


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
    parser.add_argument("--n-valid", type=int, default=100, help="Sequences/family for Tasks 1&2.")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.6, 0.8])
    parser.add_argument("--n-anomaly", type=int, default=100, help="Sequences/family for Task 3.")
    parser.add_argument("--invalid-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "eval"))
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
    split = family_test_split(args.holdout_family) if args.holdout_family else args.split
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

    n_invalid_target = int(args.n_anomaly * args.invalid_frac)
    for family, recs in sorted(by_family.items()):
        kept_valid = kept_invalid = 0
        for rec in recs:
            if kept_valid + kept_invalid >= args.n_anomaly:
                break
            steps = list(rec.steps)
            want_invalid = kept_invalid < n_invalid_target
            if want_invalid:
                corrupted = _corrupt(steps, rng)
                if corrupted is None:
                    continue
                seq, rule = corrupted
                example_id = f"{family}_{rec.sequence_id}_bad"
                anomaly_rows.append([example_id, family, join_steps(seq)])
                anomaly_truth.append([example_id, 0, rule])
                kept_invalid += 1
            else:
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

    n_bad = sum(1 for r in anomaly_truth if r[1] == 0)
    print(f"counts: {bundle.counts()}")
    print(f"wrote {len(valid_rows)} valid rows (Tasks 1&2) to {out_dir}")
    print(
        f"wrote {len(anomaly_rows)} anomaly rows (Task 3): {n_bad} invalid / "
        f"{len(anomaly_rows) - n_bad} valid"
    )


if __name__ == "__main__":
    main()
