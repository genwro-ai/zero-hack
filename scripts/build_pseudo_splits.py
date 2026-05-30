"""Turn pseudo-family raw.csv files into train/valid/test splits the existing
family-keyed training pipeline can consume directly.

Each pseudo profile maps 1:1 to a distinct base family (read from its meta.json):
    pseudo_low_power      -> mosfet
    pseudo_power_vertical -> igbt
    pseudo_logic_dense    -> ic
We relabel each profile's FAMILY column to its base family and write
``<BASEFAMILY>_{train,valid,test}.csv`` (FAMILY,SEQUENCE_ID,STEP) under the output
splits dir. The model then trains conditioned on the real, trained base-family
token, and a leave-one-family-out run over {mosfet,igbt,ic} becomes a
leave-one-pseudo-GRAMMAR-out OOD experiment — no changes to load_split_records /
decoder_training needed.

Usage:
    uv run python -m scripts.build_pseudo_splits
    uv run python scripts/build_pseudo_splits.py --train 0.8 --valid 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.data.datasets import FAMILY_FILE_NAMES

DEFAULT_IN_ROOT = PROJECT_ROOT / "data" / "eval" / "pseudo_families"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "generated" / "pseudo" / "splits"


def read_sequences(path: Path) -> list[tuple[str, list[str]]]:
    seqs: OrderedDict[str, list[str]] = OrderedDict()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sid = row["SEQUENCE_ID"].strip()
            seqs.setdefault(sid, []).append(row["STEP"].strip())
    return list(seqs.items())


def write_split(path: Path, base_family: str, items: list[tuple[str, list[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FAMILY", "SEQUENCE_ID", "STEP"])
        for sid, steps in items:
            for step in steps:
                writer.writerow([base_family, sid, step])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-root", default=str(DEFAULT_IN_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--train", type=float, default=0.8, help="Train fraction.")
    parser.add_argument("--valid", type=float, default=0.1, help="Valid fraction (rest -> test).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_root, out_root = Path(args.in_root), Path(args.out_root)
    if not in_root.exists():
        raise SystemExit(f"pseudo-family input root not found: {in_root}")

    seen_families: dict[str, str] = {}
    for prof_dir in sorted(p for p in in_root.iterdir() if p.is_dir()):
        raw, meta = prof_dir / "raw.csv", prof_dir / "meta.json"
        if not (raw.exists() and meta.exists()):
            print(f"skip {prof_dir.name}: missing raw.csv/meta.json")
            continue
        base_family = json.loads(meta.read_text())["profile"]["base_family"]
        if base_family in seen_families:
            raise SystemExit(
                f"base_family collision: {prof_dir.name} and {seen_families[base_family]} "
                f"both map to {base_family!r}; leave-one-out needs distinct base families."
            )
        seen_families[base_family] = prof_dir.name

        seqs = read_sequences(raw)
        n = len(seqs)
        n_train = int(n * args.train)
        n_valid = int(n * args.valid)
        splits = {
            "train": seqs[:n_train],
            "valid": seqs[n_train : n_train + n_valid],
            "test": seqs[n_train + n_valid :],
        }
        stem = FAMILY_FILE_NAMES[base_family].removesuffix(".csv")
        for split, items in splits.items():
            out_path = out_root / f"{stem}_{split}.csv"
            write_split(out_path, base_family, items)
            print(
                f"{prof_dir.name:22s} -> {base_family:6s} {split:5s}: "
                f"{len(items):5d} seqs -> {out_path}"
            )

    print(f"\nwrote splits for {len(seen_families)} pseudo-families to {out_root}")


if __name__ == "__main__":
    main()
