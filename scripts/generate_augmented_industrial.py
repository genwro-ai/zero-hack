#!/usr/bin/env python3
"""Generate augmented Industrial sequence datasets using all documented variation axes."""

from __future__ import annotations

import argparse
from pathlib import Path

from zero_hack.data.augmented_generator import (
    AugmentationOptions,
    family_output_name,
    generate_augmented_dataset,
    write_augmented_csv,
)
from zero_hack.data.datasets import FAMILY_FILE_NAMES


def _optional_bool(value: str) -> bool | None:
    lowered = value.lower()
    if lowered in {"random", "none"}:
        return None
    if lowered in {"1", "true", "yes", "on", "present"}:
        return True
    if lowered in {"0", "false", "no", "off", "absent"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false/random, got {value!r}")


def _dataset_label(count: int) -> str:
    if count % 1000 == 0:
        return f"augmented_s{count // 1000:03d}k"
    return f"augmented_s{count}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-root", default="data/generated")
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(FAMILY_FILE_NAMES),
        default=["mosfet", "igbt", "ic"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-validate", action="store_true")

    parser.add_argument("--litho-cycles", type=int, choices=range(3, 7), default=None)
    parser.add_argument("--post-expose-bake", type=_optional_bool, default=None)
    parser.add_argument("--hard-bake", type=_optional_bool, default=None)
    parser.add_argument("--intermediate-clean", type=_optional_bool, default=None)
    parser.add_argument("--extra-measurements", type=_optional_bool, default=None)
    parser.add_argument("--dry-wafer", type=_optional_bool, default=None)
    parser.add_argument("--epitaxial-rework-check", type=_optional_bool, default=None)
    parser.add_argument("--pre-anneal-check", type=_optional_bool, default=None)
    parser.add_argument("--second-metal-layer", type=_optional_bool, default=False)
    parser.add_argument("--cmp-after-via-fill", type=_optional_bool, default=None)
    parser.add_argument(
        "--synonym-style",
        choices=("random", "canonical", "alternate"),
        default="random",
    )
    return parser.parse_args()


def _family_seed(base_seed: int, family: str) -> int:
    return base_seed + {"mosfet": 1, "igbt": 2, "ic": 3}[family]


def main() -> None:
    args = parse_args()
    dataset = args.dataset or _dataset_label(args.count)
    raw_dir = Path(args.output_root) / dataset / "raw"
    options = AugmentationOptions(
        litho_cycles=args.litho_cycles,
        post_expose_bake=args.post_expose_bake,
        hard_bake=args.hard_bake,
        intermediate_clean=args.intermediate_clean,
        extra_measurements=args.extra_measurements,
        dry_wafer=args.dry_wafer,
        epitaxial_rework_check=args.epitaxial_rework_check,
        pre_anneal_check=args.pre_anneal_check,
        second_metal_layer=args.second_metal_layer,
        cmp_after_via_fill=args.cmp_after_via_fill,
        synonym_style=args.synonym_style,
    )

    for family in args.families:
        out_path = raw_dir / family_output_name(family)
        if out_path.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite {out_path}. Use --force.")

        seed = _family_seed(args.seed, family)
        print(f"generating family={family} count={args.count} seed={seed} -> {out_path}")
        sequences = generate_augmented_dataset(
            family,
            args.count,
            seed=seed,
            options=options,
            validate=not args.no_validate,
        )
        write_augmented_csv(out_path, sequences)
        print(f"wrote {len(sequences)} sequences / {sum(len(seq) for seq in sequences):,} rows")


if __name__ == "__main__":
    main()
