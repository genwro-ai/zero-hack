import argparse

from zero_hack.models.common import (
    DEFAULT_METRICS_DIR,
    DEFAULT_SPLITS_DIR,
    load_split_records,
    report_splits,
    split_role,
    write_eval_report,
)
from zero_hack.models.xgboost.model import XGBoostNextStep


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost next-step baseline.")
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--report-dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--lag", type=int, default=8)
    args = parser.parse_args()

    bundle = load_split_records(
        splits_dir=args.splits_dir,
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

    results: dict[str, dict[str, dict[str, float]]] = {}
    for split in report_splits(bundle):
        summary = model.evaluate(bundle.records[split], bundle.vocabulary, k=args.k)
        results[split] = summary
        print(f"{split} ({split_role(split, bundle)}) summary: {summary}")
    write_eval_report("xgboost", bundle, results, k=args.k, report_dir=args.report_dir)


if __name__ == "__main__":
    main()
