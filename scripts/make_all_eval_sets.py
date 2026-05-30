#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

from zero_hack import PROJECT_ROOT

_DATASET_SIZE = re.compile(r"_s(\d+)k$")


def _dataset_sort_key(name: str) -> tuple[int, str]:
    match = _DATASET_SIZE.search(name)
    if match:
        return int(match.group(1)), name
    return sys.maxsize, name


def _discover_datasets(generated_root: Path) -> list[str]:
    datasets = []
    for path in generated_root.iterdir():
        if not path.is_dir():
            continue
        splits_dir = path / "splits"
        if (splits_dir / "test.csv").exists():
            datasets.append(path.name)
    return sorted(datasets, key=_dataset_sort_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create eval sets for each generated dataset and each two-family-train "
            "holdout combination."
        )
    )
    parser.add_argument(
        "--generated-root",
        default=str(PROJECT_ROOT / "data" / "generated"),
        help="Root containing dataset directories such as valid_s005k.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset labels to process. Defaults to all discovered datasets.",
    )
    parser.add_argument(
        "--out-root",
        default=str(PROJECT_ROOT / "data" / "eval"),
        help=(
            "Parent output directory. Eval sets are written as <dataset>/holdout_<family>/{id,ood}."
        ),
    )
    parser.add_argument("--n-valid", type=int, default=100)
    parser.add_argument("--n-anomaly-valid", type=int, default=200)
    parser.add_argument("--n-anomaly-invalid", type=int, default=129)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.6, 0.8])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument(
        "--holdout-families",
        nargs="+",
        choices=("mosfet", "igbt", "ic"),
        default=["mosfet", "igbt", "ic"],
        help="Holdout combinations to build. Default: all three families.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_root = Path(args.generated_root)
    out_root = Path(args.out_root)
    datasets = args.datasets or _discover_datasets(generated_root)

    if not datasets:
        raise SystemExit(f"No datasets found under {generated_root}")

    script = PROJECT_ROOT / "scripts" / "make_eval_set.py"
    for dataset in datasets:
        splits_dir = generated_root / dataset / "splits"
        if not splits_dir.exists():
            raise SystemExit(f"Missing split directory for {dataset}: {splits_dir}")

        for holdout_family in args.holdout_families:
            eval_views = {
                "id": [family for family in ("mosfet", "igbt", "ic") if family != holdout_family],
                "ood": [holdout_family],
            }
            for view_name, eval_families in eval_views.items():
                cmd = [
                    sys.executable,
                    str(script),
                    "--splits-dir",
                    str(splits_dir),
                    "--out-dir",
                    str(out_root / dataset / f"holdout_{holdout_family}" / view_name),
                    "--holdout-family",
                    holdout_family,
                    "--eval-families",
                    *eval_families,
                    "--n-valid",
                    str(args.n_valid),
                    "--n-anomaly-valid",
                    str(args.n_anomaly_valid),
                    "--n-anomaly-invalid",
                    str(args.n_anomaly_invalid),
                    "--seed",
                    str(args.seed),
                    "--split",
                    args.split,
                    "--fractions",
                    *[str(value) for value in args.fractions],
                ]
                if args.limit_per_family is not None:
                    cmd.extend(["--limit-per-family", str(args.limit_per_family)])

                print(" ".join(cmd), flush=True)
                if not args.dry_run:
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
