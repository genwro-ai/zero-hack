from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from zero_hack.data.datasets import (
    FAMILY_TOKENS,
    Vocabulary,
    collate_next_step_batch,
)
from zero_hack.eval import io


@dataclass(frozen=True)
class EvalNextStepRecord:
    """One Task 1 eval example: partial sequence plus the true next step."""

    example_id: str
    family: str
    completion_fraction: float | None
    partial_sequence: tuple[str, ...]
    next_step: str


class EvalNextStepDataset:
    """Dataset for fixed next-step eval examples from data/eval/.../{id,ood}."""

    def __init__(
        self,
        records: list[EvalNextStepRecord],
        vocabulary: Vocabulary,
        max_context: int = 192,
    ) -> None:
        self.records = records
        self.vocabulary = vocabulary
        self.max_context = max_context

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        family_token = FAMILY_TOKENS.get(record.family, FAMILY_TOKENS["unknown"])
        tokens = ["<BOS>", family_token, *record.partial_sequence]
        tokens = tokens[-self.max_context :]

        return {
            "input_ids": self.vocabulary.encode(tokens),
            "target_id": self.vocabulary.token_to_id.get(
                record.next_step,
                self.vocabulary.unk_id,
            ),
            "family": record.family,
            "sequence_id": record.example_id,
            "position": len(record.partial_sequence),
            "completion_fraction": record.completion_fraction,
        }


def load_eval_next_step_records(eval_dir: str | Path) -> list[EvalNextStepRecord]:
    """Load Task 1 eval examples from one eval directory."""
    eval_dir = Path(eval_dir)
    inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
    truth = io.read_next_step_truth(eval_dir / "nextstep_truth.csv")

    records = []
    for row in inputs:
        example_id = row["example_id"]
        if example_id not in truth:
            raise ValueError(f"{eval_dir}: missing next-step truth for {example_id!r}")
        records.append(
            EvalNextStepRecord(
                example_id=example_id,
                family=row["family"],
                completion_fraction=row["completion_fraction"],
                partial_sequence=tuple(row["partial_sequence"]),
                next_step=truth[example_id],
            )
        )
    return records


def make_eval_next_step_dataset(
    eval_dir: str | Path,
    vocabulary: Vocabulary,
    *,
    max_context: int = 192,
) -> EvalNextStepDataset:
    """Create a fixed-example Task 1 dataset from one eval directory."""
    return EvalNextStepDataset(
        records=load_eval_next_step_records(eval_dir),
        vocabulary=vocabulary,
        max_context=max_context,
    )


def make_eval_next_step_loader(
    eval_dir: str | Path,
    vocabulary: Vocabulary,
    *,
    batch_size: int = 128,
    max_context: int = 192,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for one Task 1 eval directory."""
    dataset = make_eval_next_step_dataset(
        eval_dir,
        vocabulary,
        max_context=max_context,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_next_step_batch(batch, vocabulary.pad_id),
    )


def make_holdout_eval_next_step_loaders(
    eval_root: str | Path,
    dataset: str,
    holdout_family: str,
    vocabulary: Vocabulary,
    *,
    views: tuple[str, ...] = ("id", "ood"),
    batch_size: int = 128,
    max_context: int = 192,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create Task 1 eval loaders for data/eval/<dataset>/holdout_<family>/{id,ood}."""
    base_dir = Path(eval_root) / dataset / f"holdout_{holdout_family}"
    return {
        view: make_eval_next_step_loader(
            base_dir / view,
            vocabulary,
            batch_size=batch_size,
            max_context=max_context,
            num_workers=num_workers,
        )
        for view in views
    }
