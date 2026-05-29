"""Small helpers for loading the Industrial AI sequence data."""

from __future__ import annotations

import csv
from pathlib import Path

from zero_hack import INDUSTRIAL_DATA_DIR

FAMILY_VARIANT_FILES = {
    "mosfet": "MOSFET_variants.csv",
    "igbt": "IGBT_variants.csv",
    "ic": "IC_variants.csv",
}


def industrial_data_path(filename: str) -> Path:
    """Return an absolute path inside data/industrial."""
    return INDUSTRIAL_DATA_DIR / filename


def load_long_sequences(path: str | Path) -> dict[str, list[str]]:
    """Load a long-format sequence CSV with SEQUENCE_ID and STEP columns."""
    sequences: dict[str, list[str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sequences.setdefault(row["SEQUENCE_ID"], []).append(row["STEP"])
    return sequences


def load_family_sequences(family: str) -> dict[str, list[str]]:
    """Load one of the pre-generated Industrial AI family datasets."""
    key = family.lower()
    try:
        filename = FAMILY_VARIANT_FILES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(FAMILY_VARIANT_FILES))
        raise ValueError(f"Unknown family {family!r}. Expected one of: {allowed}") from exc
    return load_long_sequences(industrial_data_path(filename))
