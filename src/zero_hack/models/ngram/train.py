import argparse

from zero_hack.models.common import DEFAULT_SPLITS_DIR, load_split_records, split_role
from zero_hack.models.ngram.model import NGramModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Classic n-gram next-step baseline.")
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    bundle = load_split_records(
        splits_dir=args.splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")

    model = NGramModel(n=args.n, backoff_alpha=args.alpha).fit(bundle.records["train"])

    for split in bundle.test_split_names:
        summary = model.evaluate(bundle.records[split], bundle.vocabulary, k=args.k)
        print(f"{split} ({split_role(split, bundle)}) summary: {summary}")


if __name__ == "__main__":
    main()
