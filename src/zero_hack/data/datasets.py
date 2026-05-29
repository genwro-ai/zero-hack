import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

FAMILY_FILE_NAMES = {
    "mosfet": "MOSFET.csv",
    "igbt": "IGBT.csv",
    "ic": "IC.csv",
}

SPECIAL_TOKENS = ("<PAD>", "<BOS>", "<EOS>", "<UNK_STEP>")
FAMILY_TOKENS = {
    "mosfet": "<FAMILY_MOSFET>",
    "igbt": "<FAMILY_IGBT>",
    "ic": "<FAMILY_IC>",
    "unknown": "<FAMILY_UNKNOWN>",
}


@dataclass(frozen=True)
class SequenceRecord:
    """One process sequence."""

    family: str
    sequence_id: str
    steps: tuple[str, ...]


@dataclass(frozen=True)
class Vocabulary:
    """Token vocabulary for step-sequence models."""

    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<BOS>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<UNK_STEP>"]

    def encode(self, tokens: list[str] | tuple[str, ...]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]


def normalize_family(family: str) -> str:
    key = family.lower()
    if key not in FAMILY_FILE_NAMES:
        allowed = ", ".join(sorted(FAMILY_FILE_NAMES))
        raise ValueError(f"Unknown family {family!r}. Expected one of: {allowed}")
    return key


def load_sequence_records(path: str | Path, family: str | None = None) -> list[SequenceRecord]:
    """Load long-format sequence records.

    Accepted CSV schemas:
    - SEQUENCE_ID, STEP
    - FAMILY, SEQUENCE_ID, STEP
    """
    sequences: dict[tuple[str, str], list[str]] = {}
    path = Path(path)
    default_family = normalize_family(family) if family else None

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        if "SEQUENCE_ID" not in headers or "STEP" not in headers:
            raise ValueError(f"{path} must contain SEQUENCE_ID and STEP columns")

        for row in reader:
            row_family = row.get("FAMILY") or default_family
            if row_family is None:
                raise ValueError(f"{path} has no FAMILY column; pass family=...")
            family_key = normalize_family(row_family)
            step = row["STEP"].strip()
            if not step:
                continue
            key = (family_key, row["SEQUENCE_ID"].strip())
            sequences.setdefault(key, []).append(step)

    return [
        SequenceRecord(family=family_key, sequence_id=sequence_id, steps=tuple(steps))
        for (family_key, sequence_id), steps in sequences.items()
    ]


def load_raw_family_records(raw_dir: str | Path, family: str) -> list[SequenceRecord]:
    family_key = normalize_family(family)
    path = Path(raw_dir) / FAMILY_FILE_NAMES[family_key]
    return load_sequence_records(path, family=family_key)


def write_sequence_records(path: str | Path, records: list[SequenceRecord]) -> None:
    """Write long-format records with FAMILY, SEQUENCE_ID, STEP columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FAMILY", "SEQUENCE_ID", "STEP"])
        for record in records:
            for step in record.steps:
                writer.writerow([record.family, record.sequence_id, step])


def dedupe_records(records: list[SequenceRecord]) -> list[SequenceRecord]:
    """Remove exact duplicate full sequences within each family."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[SequenceRecord] = []
    for record in records:
        key = (record.family, record.steps)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def split_records(
    records: list[SequenceRecord],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 1729,
) -> dict[str, list[SequenceRecord]]:
    """Return deterministic train/valid/test splits."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be below 1")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    valid_end = train_end + int(len(shuffled) * valid_ratio)
    return {
        "train": shuffled[:train_end],
        "valid": shuffled[train_end:valid_end],
        "test": shuffled[valid_end:],
    }


def build_vocabulary(records: list[SequenceRecord], min_count: int = 1) -> Vocabulary:
    """Build a token vocabulary from records."""
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.steps)

    tokens = list(SPECIAL_TOKENS) + list(FAMILY_TOKENS.values())
    tokens.extend(sorted(token for token, count in counts.items() if count >= min_count))
    deduped_tokens = list(dict.fromkeys(tokens))
    token_to_id = {token: idx for idx, token in enumerate(deduped_tokens)}
    return Vocabulary(token_to_id=token_to_id, id_to_token=tuple(deduped_tokens))


class NextStepDataset:
    """Next-token prediction dataset over process sequences.

    This class intentionally does not require Torch. Its items are dictionaries
    of Python lists/ints. Use `collate_next_step_batch` or `make_torch_dataloader`
    when Torch is available.
    """

    def __init__(
        self,
        records: list[SequenceRecord],
        vocabulary: Vocabulary,
        max_context: int = 192,
        family_dropout: float = 0.0,
        seed: int = 1729,
    ) -> None:
        self.records = records
        self.vocabulary = vocabulary
        self.max_context = max_context
        self.family_dropout = family_dropout
        self.rng = random.Random(seed)
        self.index: list[tuple[int, int]] = []
        for record_idx, record in enumerate(records):
            for position in range(len(record.steps)):
                self.index.append((record_idx, position))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record_idx, position = self.index[idx]
        record = self.records[record_idx]
        family_token = FAMILY_TOKENS[record.family]
        if self.family_dropout and self.rng.random() < self.family_dropout:
            family_token = FAMILY_TOKENS["unknown"]

        prefix = list(record.steps[:position])
        tokens = ["<BOS>", family_token] + prefix
        tokens = tokens[-self.max_context :]
        input_ids = self.vocabulary.encode(tokens)
        target_id = self.vocabulary.token_to_id.get(record.steps[position], self.vocabulary.unk_id)

        return {
            "input_ids": input_ids,
            "target_id": target_id,
            "family": record.family,
            "sequence_id": record.sequence_id,
            "position": position,
        }


def collate_next_step_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    """Collate next-step examples into Torch tensors."""
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append([pad_id] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "target_id": torch.tensor([item["target_id"] for item in batch], dtype=torch.long),
        "family": [item["family"] for item in batch],
        "sequence_id": [item["sequence_id"] for item in batch],
        "position": torch.tensor([item["position"] for item in batch], dtype=torch.long),
    }


def make_torch_dataloader(
    dataset: NextStepDataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a Torch DataLoader for `NextStepDataset`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_next_step_batch(batch, dataset.vocabulary.pad_id),
    )
