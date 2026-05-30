#!/usr/bin/env python3
import argparse
from pathlib import Path

from zero_hack.data.datasets import (
    FAMILY_FILE_NAMES,
    dedupe_records,
    load_industrial_family_records,
    load_raw_family_records,
    namespace_sequence_ids,
    split_records,
    write_sequence_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="valid_s005k",
        help="Dataset label under data/generated, e.g. valid_s005k.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Raw input directory. Defaults to data/generated/<dataset>/raw.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Split output directory. Defaults to data/generated/<dataset>/splits.",
    )
    parser.add_argument(
        "--industrial-dir",
        default="data/industrial",
        help="Directory containing the provided Industrial track variant CSVs.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["mosfet", "igbt", "ic"],
        choices=sorted(FAMILY_FILE_NAMES),
        help="Families to include.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--no-include-industrial",
        action="store_true",
        help="Use only generated raw CSVs, without the provided industrial variants.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing split CSV files.",
    )
    return parser.parse_args()


def assert_can_write(output_dir: Path, force: bool) -> None:
    outputs = [
        output_dir / "train.csv",
        output_dir / "valid.csv",
        output_dir / "test.csv",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        paths = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing splits. Use --force.\n{paths}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir or f"data/generated/{args.dataset}/raw")
    output_dir = Path(args.output_dir or f"data/generated/{args.dataset}/splits")

    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    assert_can_write(output_dir, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_splits = {"train": [], "valid": [], "test": []}
    for family in args.families:
        generated = namespace_sequence_ids(
            load_raw_family_records(input_dir, family),
            "generated",
        )
        industrial = []
        if not args.no_include_industrial:
            industrial = namespace_sequence_ids(
                load_industrial_family_records(args.industrial_dir, family),
                "industrial",
            )

        records = industrial + generated
        deduped = dedupe_records(records)
        splits = split_records(
            deduped,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )

        raw_count = len(generated)
        industrial_count = len(industrial)
        duplicate_count = len(records) - len(deduped)
        print(
            f"{family}: generated={raw_count} industrial={industrial_count} "
            f"deduped={len(deduped)} "
            f"duplicates={duplicate_count} "
            f"train={len(splits['train'])} valid={len(splits['valid'])} "
            f"test={len(splits['test'])}"
        )

        family_name = FAMILY_FILE_NAMES[family].removesuffix(".csv")
        for split_name, split_records_for_family in splits.items():
            write_sequence_records(
                output_dir / f"{family_name}_{split_name}.csv",
                split_records_for_family,
            )
            combined_splits[split_name].extend(split_records_for_family)

    for split_name, records in combined_splits.items():
        write_sequence_records(output_dir / f"{split_name}.csv", records)

    print(f"wrote splits to {output_dir}")


if __name__ == "__main__":
    main()
