import argparse

from zero_hack.models.common import DEFAULT_RAW_DIR, load_record_splits
from zero_hack.models.xgboost.model import XGBoostNextStep


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost next-step baseline.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--industrial-dir", default=None)
    parser.add_argument("--no-include-industrial", action="store_true")
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--lag", type=int, default=8)
    args = parser.parse_args()

    bundle = load_record_splits(
        raw_dir=args.raw_dir,
        industrial_dir=args.industrial_dir or None,
        include_industrial=not args.no_include_industrial,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")

    model = XGBoostNextStep(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        lag=args.lag,
    ).fit(bundle.records["train"])

    for split in bundle.test_split_names:
        summary = model.evaluate(bundle.records[split], bundle.vocabulary, k=args.k)
        label = split.removeprefix("test_")
        role = "ood" if label == bundle.holdout_family else "id"
        print(f"{split} ({role}) summary: {summary}")


if __name__ == "__main__":
    main()
