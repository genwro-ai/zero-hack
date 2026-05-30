#!/usr/bin/env python3
import argparse
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.data.industrial_eval import (
    FAMILIES,
    load_industrial_variant_records,
    records_by_family,
    write_industrial_eval_set,
    write_threshold_calibration_set,
)

_CALIBRATION_VIEW = "calibration"
_CALIBRATION_HEALTHY = 650
_CALIBRATION_UNHEALTHY = 350


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--industrial-dir",
        default=str(PROJECT_ROOT / "data" / "industrial"),
        help="Directory containing *_variants.csv.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Output dataset labels to materialize under data/eval.",
    )
    parser.add_argument("--out-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.6, 0.8])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--holdout-families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    industrial_dir = Path(args.industrial_dir)

    variants = load_industrial_variant_records(industrial_dir)
    variant_by_family = records_by_family(variants)

    for dataset in args.datasets:
        for holdout_family in args.holdout_families:
            train_families = tuple(family for family in FAMILIES if family != holdout_family)
            views = {
                "id": train_families,
                "ood": (holdout_family,),
            }

            for view, families in views.items():
                out_dir = out_root / dataset / f"holdout_{holdout_family}" / view
                valid_records = [
                    record for family in families for record in variant_by_family.get(family, [])
                ]
                print(
                    f"{out_dir}: source=industrial_variants "
                    f"anomaly=generated_rule_cases families={','.join(families)} "
                    f"valid={len(valid_records)}"
                )
                if args.dry_run:
                    continue
                write_industrial_eval_set(
                    out_dir,
                    valid_records=valid_records,
                    fractions=tuple(args.fractions),
                    seed=args.seed,
                    metadata={
                        "dataset_label": dataset,
                        "holdout_family": holdout_family,
                        "train_families": list(train_families),
                        "view": view,
                    },
                )

            calibration_dir = out_root / dataset / f"holdout_{holdout_family}" / _CALIBRATION_VIEW
            calibration_valid = [
                record for family in train_families for record in variant_by_family.get(family, [])
            ]
            print(
                f"{calibration_dir}: source=industrial_variants "
                f"anomaly=generated_rule_cases "
                f"families={','.join(train_families)} "
                f"threshold_calibration={_CALIBRATION_HEALTHY}+{_CALIBRATION_UNHEALTHY}"
            )
            if args.dry_run:
                continue
            write_threshold_calibration_set(
                calibration_dir,
                valid_records=calibration_valid,
                n_valid=_CALIBRATION_HEALTHY,
                n_invalid=_CALIBRATION_UNHEALTHY,
                seed=args.seed,
                metadata={
                    "dataset_label": dataset,
                    "holdout_family": holdout_family,
                    "train_families": list(train_families),
                    "view": _CALIBRATION_VIEW,
                },
            )


if __name__ == "__main__":
    main()
