"""Turn novel-family raw.csv files into train/valid/test splits the existing
family-keyed training pipeline can consume directly.

Novel profiles are FAMILY-LESS: unlike pseudo profiles (which are per-block
recombinations of real family grammars and inherit a clear base family), novel
profiles are cross-family atomic compositions that belong to no official family.
Their meta.json has NO ``base_family`` field.

To run these splits through the existing family-conditioned training pipeline
(which conditions on ``<FAMILY_*>`` tokens and supports ``--holdout-family``),
each profile is assigned an *arbitrary* conditioning-family token via an explicit
mapping. This assignment is a purely mechanical eval construct — it lets
leave-one-family-out work syntactically — but has NO semantic meaning: the flows
were generated without any family grammar and the assigned token is not ground
truth. A leave-one-(assigned)-family-out run over {mosfet,igbt,ic} is therefore
closer to in-distribution than a true OOD experiment, because all three novel
profiles are density variants of the same cross-family distribution. The cleanest
OOD experiment would be to merge the novel splits INTO a real dataset's splits and
hold out a real family; that augmentation/merge mode is a natural ``--augment``
future extension.

Default mapping (override with ``--family-map``):
    novel_mixed  → mosfet
    novel_sparse → igbt
    novel_dense  → ic

Usage:
    uv run python -m scripts.build_novel_splits
    uv run python scripts/build_novel_splits.py --train 0.8 --valid 0.1
    uv run python scripts/build_novel_splits.py \\
        --family-map novel_mixed=mosfet novel_sparse=igbt novel_dense=ic
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.data.datasets import FAMILY_FILE_NAMES

DEFAULT_IN_ROOT = PROJECT_ROOT / "data" / "eval" / "novel_families"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "generated" / "novel" / "splits"

# Default profile → conditioning-family mapping.
# This is an ARBITRARY eval construct (see module docstring).
DEFAULT_FAMILY_MAP: dict[str, str] = {
    "novel_mixed": "mosfet",
    "novel_sparse": "igbt",
    "novel_dense": "ic",
}


def read_sequences(path: Path) -> list[tuple[str, list[str]]]:
    seqs: OrderedDict[str, list[str]] = OrderedDict()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = row["SEQUENCE_ID"].strip()
            seqs.setdefault(sid, []).append(row["STEP"].strip())
    return list(seqs.items())


def write_split(path: Path, assigned_family: str, items: list[tuple[str, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FAMILY", "SEQUENCE_ID", "STEP"])
        for sid, steps in items:
            for step in steps:
                writer.writerow([assigned_family, sid, step])


def parse_family_map(pairs: list[str]) -> dict[str, str]:
    """Parse ``profile=family`` pairs from CLI into a dict."""
    result: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(
                f"Invalid --family-map entry {pair!r}; expected profile=family "
                "(e.g. novel_mixed=mosfet)"
            )
        profile, family = pair.split("=", 1)
        if family not in FAMILY_FILE_NAMES:
            allowed = ", ".join(sorted(FAMILY_FILE_NAMES))
            raise SystemExit(
                f"Unknown family {family!r} in --family-map. Expected one of: {allowed}"
            )
        result[profile.strip()] = family.strip()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-root", default=str(DEFAULT_IN_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction.")
    parser.add_argument("--valid", type=float, default=0.1, help="Valid fraction (rest -> test).")
    parser.add_argument(
        "--family-map",
        nargs="+",
        default=[],
        metavar="PROFILE=FAMILY",
        help=(
            "Override the default profile→family assignment, e.g. "
            "novel_mixed=mosfet novel_sparse=igbt novel_dense=ic. "
            "Each assignment is ARBITRARY (see module docstring)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_root, out_root = Path(args.in_root), Path(args.out_root)
    if not in_root.exists():
        raise SystemExit(f"novel-family input root not found: {in_root}")

    family_map = dict(DEFAULT_FAMILY_MAP)
    if args.family_map:
        family_map.update(parse_family_map(args.family_map))

    # Guard: two profiles must not map to the same family — leave-one-out needs
    # distinct assigned families.
    seen_families: dict[str, str] = {}
    for prof_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        raw = prof_dir / "raw.csv"
        if not raw.exists():
            continue
        profile_name = prof_dir.name
        if profile_name not in family_map:
            print(f"skip {profile_name}: no family assignment in --family-map (add it to use)")
            continue
        assigned_family = family_map[profile_name]
        if assigned_family in seen_families:
            raise SystemExit(
                f"family collision: {profile_name} and {seen_families[assigned_family]} "
                f"both map to {assigned_family!r}; leave-one-out needs distinct assigned families."
            )
        seen_families[assigned_family] = profile_name

        seqs = read_sequences(raw)
        n = len(seqs)
        n_train = int(n * args.train)
        n_valid = int(n * args.valid)
        splits = {
            "train": seqs[:n_train],
            "valid": seqs[n_train : n_train + n_valid],
            "test": seqs[n_train + n_valid :],
        }
        stem = FAMILY_FILE_NAMES[assigned_family].removesuffix(".csv")
        for split, items in splits.items():
            out_path = out_root / f"{stem}_{split}.csv"
            write_split(out_path, assigned_family, items)
            print(
                f"{profile_name:22s} -> {assigned_family:6s} {split:5s}: "
                f"{len(items):5d} seqs -> {out_path}"
            )

    print(f"\nwrote splits for {len(seen_families)} novel-family profiles to {out_root}")
    print(
        "NOTE: assigned family tokens are ARBITRARY eval constructs; "
        "novel flows are family-less.\n"
        "      Leave-one-(assigned)-family-out is closer to in-distribution than true OOD.\n"
        "      For a stronger OOD experiment, merge these splits into a real dataset and hold\n"
        "      out a real family (future --augment/merge mode)."
    )


if __name__ == "__main__":
    main()
