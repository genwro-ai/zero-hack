"""Load raw generated CSVs into deduped, split JSONL records and back.

A processed dataset directory contains::

    vocab.json
    train.jsonl
    valid.jsonl
    test.jsonl

Each JSONL line is ``{"sequence_id", "family", "steps": [...]}``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from zero_hack.data import read_csv_sequences
from zero_hack.splits import SPLIT_NAMES, split_for
from zero_hack.vocab import FAMILIES


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    family: str
    steps: list[str]


def load_raw_family(raw_dir: str | Path, family: str) -> list[SequenceRecord]:
    """Read one family's raw CSV (``<FAMILY>.csv``) into records."""
    path = Path(raw_dir) / f"{family.upper()}.csv"
    sequences = read_csv_sequences(path)
    return [
        SequenceRecord(sequence_id=f"{family}-{sid}", family=family, steps=steps)
        for sid, steps in sequences.items()
    ]


def dedupe(records: list[SequenceRecord]) -> tuple[list[SequenceRecord], int]:
    """Drop exact duplicate (family, steps) sequences. Returns (kept, n_removed)."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    kept: list[SequenceRecord] = []
    for rec in records:
        key = (rec.family, tuple(rec.steps))
        if key in seen:
            continue
        seen.add(key)
        kept.append(rec)
    return kept, len(records) - len(kept)


def split_records(records: list[SequenceRecord]) -> dict[str, list[SequenceRecord]]:
    """Partition records into train/valid/test using the deterministic hash split."""
    out: dict[str, list[SequenceRecord]] = {name: [] for name in SPLIT_NAMES}
    for rec in records:
        out[split_for(rec.family, rec.steps)].append(rec)
    return out


def write_jsonl(path: str | Path, records: list[SequenceRecord]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(
                json.dumps(
                    {"sequence_id": rec.sequence_id, "family": rec.family, "steps": rec.steps}
                )
                + "\n"
            )


def read_jsonl(path: str | Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                SequenceRecord(
                    sequence_id=obj["sequence_id"], family=obj["family"], steps=obj["steps"]
                )
            )
    return records


def iter_all_families(raw_dir: str | Path) -> Iterator[SequenceRecord]:
    for family in FAMILIES:
        yield from load_raw_family(raw_dir, family)
