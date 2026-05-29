"""CLI to fit and evaluate the most-frequent next-step baseline."""

from __future__ import annotations

import argparse

from zero_hack.models.common import DEFAULT_RAW_DIR, load_record_splits
from zero_hack.models.most_frequent.model import MostFrequentModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Most-frequent next-step baseline.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--position-bucket-size", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    bundle = load_record_splits(
        raw_dir=args.raw_dir,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")

    model = MostFrequentModel(
        position_bucket_size=args.position_bucket_size,
        backoff_alpha=args.alpha,
    ).fit(bundle.records["train"])

    summary = model.evaluate(bundle.records["test"], bundle.vocabulary, k=args.k)
    print(f"test summary: {summary}")


if __name__ == "__main__":
    main()
