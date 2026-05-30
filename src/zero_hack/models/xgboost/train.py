import argparse

from zero_hack.models.common import DEFAULT_RAW_DIR, load_record_splits

from zero_hack.models.xgboost.model import XGBoostNextStep


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost next-step baseline.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--lag", type=int, default=8)
    args = parser.parse_args()

    bundle = load_record_splits(
        raw_dir=args.raw_dir,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")

    model = XGBoostNextStep(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        lag=args.lag,
    ).fit(bundle.records["train"])

    summary = model.evaluate(bundle.records["test"], bundle.vocabulary, k=args.k)
    print(f"test summary: {summary}")


if __name__ == "__main__":
    main()
