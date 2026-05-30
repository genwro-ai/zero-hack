from __future__ import annotations

import argparse

from zero_hack.models.common import DEFAULT_SPLITS_DIR, load_split_records
from zero_hack.models.vomm.model import VOMMModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Variable-order Markov (PPM-C) next-step baseline."
    )
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    parser.add_argument("--max-order", type=int, default=8)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    bundle = load_split_records(
        splits_dir=args.splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")

    model = VOMMModel(max_order=args.max_order).fit(bundle.records["train"])

    for split in bundle.test_split_names:
        summary = model.evaluate(bundle.records[split], bundle.vocabulary, k=args.k)
        label = split.removeprefix("test_")
        role = "ood" if label == bundle.holdout_family else "id"
        print(f"{split} ({role}) summary: {summary}")


if __name__ == "__main__":
    main()
