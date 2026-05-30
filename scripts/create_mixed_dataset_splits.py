#!/usr/bin/env python3
"""Create mixed vanilla/augmented splits with separate standard and diverse tests."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from zero_hack.data.datasets import (
    FAMILY_FILE_NAMES,
    SequenceRecord,
    dedupe_records,
    load_raw_family_records,
    namespace_sequence_ids,
    split_records,
    write_sequence_records,
)


def _dataset_label(count: int, vanilla_ratio: float, augmented_ratio: float) -> str:
    size = f"s{count // 1000:03d}k" if count % 1000 == 0 else f"s{count}"
    vanilla = int(round(vanilla_ratio * 100))
    augmented = int(round(augmented_ratio * 100))
    return f"mixed_{size}_v{vanilla:02d}_a{augmented:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000, help="Total records/family.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--generated-root", default="data/generated")
    parser.add_argument("--vanilla-dataset", default=None)
    parser.add_argument("--augmented-dataset", default=None)
    parser.add_argument("--vanilla-ratio", type=float, default=0.4)
    parser.add_argument("--augmented-ratio", type=float, default=0.6)
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(FAMILY_FILE_NAMES),
        default=["mosfet", "igbt", "ic"],
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _assert_ratios(args: argparse.Namespace) -> None:
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.vanilla_ratio <= 0 or args.augmented_ratio <= 0:
        raise SystemExit("source ratios must be positive")
    total = args.vanilla_ratio + args.augmented_ratio
    if abs(total - 1.0) > 1e-9:
        raise SystemExit("--vanilla-ratio + --augmented-ratio must equal 1.0")
    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1")
    if not 0 <= args.valid_ratio < 1:
        raise SystemExit("--valid-ratio must be between 0 and 1")
    if args.train_ratio + args.valid_ratio >= 1:
        raise SystemExit("--train-ratio + --valid-ratio must be below 1")


def _sample(
    records: list[SequenceRecord],
    count: int,
    *,
    rng: random.Random,
    label: str,
) -> list[SequenceRecord]:
    if count > len(records):
        raise SystemExit(f"Not enough records for {label}: need {count}, have {len(records)}")
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[:count]


def _mixed_counts(total: int, vanilla_ratio: float) -> tuple[int, int]:
    vanilla_count = int(round(total * vanilla_ratio))
    return vanilla_count, total - vanilla_count


def _write_family_splits(
    *,
    output_dir: Path,
    family: str,
    splits: dict[str, list[SequenceRecord]],
) -> None:
    family_name = FAMILY_FILE_NAMES[family].removesuffix(".csv")
    for split_name, records in splits.items():
        write_sequence_records(output_dir / f"{family_name}_{split_name}.csv", records)


def assert_can_write(output_dir: Path, force: bool) -> None:
    outputs = [
        output_dir / "train.csv",
        output_dir / "valid.csv",
        output_dir / "test.csv",
        output_dir / "test_standard.csv",
        output_dir / "test_diverse.csv",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        paths = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing mixed splits. Use --force.\n{paths}")


def main() -> None:
    args = parse_args()
    _assert_ratios(args)

    generated_root = Path(args.generated_root)
    vanilla_dataset = args.vanilla_dataset or (
        f"valid_s{args.count // 1000:03d}k" if args.count % 1000 == 0 else f"valid_s{args.count}"
    )
    augmented_dataset = args.augmented_dataset or (
        f"augmented_s{args.count // 1000:03d}k"
        if args.count % 1000 == 0
        else f"augmented_s{args.count}"
    )
    dataset = args.dataset or _dataset_label(args.count, args.vanilla_ratio, args.augmented_ratio)

    vanilla_raw_dir = generated_root / vanilla_dataset / "raw"
    augmented_raw_dir = generated_root / augmented_dataset / "raw"
    output_dir = generated_root / dataset / "splits"
    if not vanilla_raw_dir.exists():
        raise SystemExit(f"Missing vanilla raw directory: {vanilla_raw_dir}")
    if not augmented_raw_dir.exists():
        raise SystemExit(f"Missing augmented raw directory: {augmented_raw_dir}")

    assert_can_write(output_dir, args.force)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    target_counts = {
        "train": int(args.count * args.train_ratio),
        "valid": int(args.count * args.valid_ratio),
    }
    target_counts["test"] = args.count - target_counts["train"] - target_counts["valid"]

    combined: dict[str, list[SequenceRecord]] = {
        "train": [],
        "valid": [],
        "test": [],
        "test_standard": [],
        "test_diverse": [],
    }

    for family in args.families:
        vanilla = dedupe_records(
            namespace_sequence_ids(load_raw_family_records(vanilla_raw_dir, family), "vanilla")
        )
        augmented = dedupe_records(
            namespace_sequence_ids(load_raw_family_records(augmented_raw_dir, family), "augmented")
        )
        vanilla_splits = split_records(vanilla, args.train_ratio, args.valid_ratio, args.seed)
        augmented_splits = split_records(augmented, args.train_ratio, args.valid_ratio, args.seed)

        family_splits: dict[str, list[SequenceRecord]] = {}
        for split_name in ("train", "valid", "test"):
            vanilla_count, augmented_count = _mixed_counts(
                target_counts[split_name], args.vanilla_ratio
            )
            mixed = [
                *_sample(
                    vanilla_splits[split_name],
                    vanilla_count,
                    rng=rng,
                    label=f"{family}/{split_name}/vanilla",
                ),
                *_sample(
                    augmented_splits[split_name],
                    augmented_count,
                    rng=rng,
                    label=f"{family}/{split_name}/augmented",
                ),
            ]
            rng.shuffle(mixed)
            family_splits[split_name] = mixed

        family_splits["test_standard"] = vanilla_splits["test"]
        family_splits["test_diverse"] = augmented_splits["test"]
        _write_family_splits(output_dir=output_dir, family=family, splits=family_splits)

        for split_name, records in family_splits.items():
            combined[split_name].extend(records)

        print(
            f"{family}: train={len(family_splits['train'])} "
            f"valid={len(family_splits['valid'])} test={len(family_splits['test'])} "
            f"standard_test={len(family_splits['test_standard'])} "
            f"diverse_test={len(family_splits['test_diverse'])}"
        )

    for split_name, records in combined.items():
        write_sequence_records(output_dir / f"{split_name}.csv", records)

    print(
        f"wrote mixed splits to {output_dir} "
        f"(vanilla={args.vanilla_ratio:.2f}, augmented={args.augmented_ratio:.2f})"
    )


if __name__ == "__main__":
    main()
