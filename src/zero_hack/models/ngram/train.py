"""CLI to fit and evaluate the symbolic n-gram next-step baseline."""

from __future__ import annotations

import argparse

from zero_hack.models.common import DEFAULT_RAW_DIR, load_record_splits
from zero_hack.models.ngram.model import NGramModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Symbolic n-gram next-step baseline.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    bundle = load_record_splits(
        raw_dir=args.raw_dir,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")

    model = NGramModel(n=args.n, backoff_alpha=args.alpha).fit(bundle.records["train"])

    summary = model.evaluate(bundle.records["test"], bundle.vocabulary, k=args.k)
    print(f"test summary: {summary}")


if __name__ == "__main__":
    main()
