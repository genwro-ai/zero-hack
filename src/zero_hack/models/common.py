from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from zero_hack import PROJECT_ROOT
from zero_hack.data import (
    FAMILY_FILE_NAMES,
    NextStepDataset,
    SequenceRecord,
    Vocabulary,
    build_vocabulary,
    dedupe_records,
    load_industrial_family_records,
    load_raw_family_records,
    make_torch_dataloader,
    namespace_sequence_ids,
)
from zero_hack.models.topk import TopKAccumulator
from zero_hack.splits import SPLIT_NAMES, split_for

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "generated" / "valid_s005k" / "raw"
DEFAULT_INDUSTRIAL_DIR = PROJECT_ROOT / "data" / "industrial"
FAMILIES = tuple(FAMILY_FILE_NAMES)
TEST_SPLIT_PREFIX = "test_"


def family_test_split(family: str) -> str:
    return f"{TEST_SPLIT_PREFIX}{family.lower()}"


@dataclass(frozen=True)
class DataBundle:
    vocabulary: Vocabulary
    records: dict[str, list[SequenceRecord]]
    train_families: tuple[str, ...]
    holdout_family: str | None = None

    def counts(self) -> dict[str, int]:
        return {name: len(records) for name, records in self.records.items()}

    @property
    def test_split_names(self) -> tuple[str, ...]:
        return tuple(
            split_name
            for family in FAMILIES
            if (split_name := family_test_split(family)) in self.records
        )


def load_record_splits(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    *,
    industrial_dir: str | Path | None = DEFAULT_INDUSTRIAL_DIR,
    include_industrial: bool = True,
    families: tuple[str, ...] = FAMILIES,
    holdout_family: str | None = None,
    limit_per_family: int | None = None,
) -> DataBundle:
    """Load raw CSVs, dedupe, and split by sequence hash."""
    raw_dir = Path(raw_dir)
    industrial_dir = DEFAULT_INDUSTRIAL_DIR if industrial_dir is None else Path(industrial_dir)
    families = tuple(family.lower() for family in families)
    if holdout_family is not None:
        holdout_family = holdout_family.lower()
        if holdout_family not in families:
            raise ValueError(f"holdout_family={holdout_family!r} is not in {families}")
    train_families = tuple(family for family in families if family != holdout_family)
    if not train_families:
        raise ValueError("At least one training family is required")

    records: list[SequenceRecord] = []
    for family in families:
        fam_records = namespace_sequence_ids(load_raw_family_records(raw_dir, family), "generated")
        if include_industrial:
            fam_records = (
                namespace_sequence_ids(
                    load_industrial_family_records(industrial_dir, family),
                    "industrial",
                )
                + fam_records
            )
        if limit_per_family is not None:
            fam_records = fam_records[:limit_per_family]
        records.extend(fam_records)

    records = dedupe_records(records)

    by_split: dict[str, list[SequenceRecord]] = {name: [] for name in SPLIT_NAMES}
    for family in families:
        by_split[family_test_split(family)] = []

    for record in records:
        split = split_for(record.family, list(record.steps))
        if split == "test":
            by_split[family_test_split(record.family)].append(record)
            if record.family != holdout_family:
                by_split["test"].append(record)
        elif record.family != holdout_family:
            by_split[split].append(record)

    vocabulary = build_vocabulary(by_split["train"])
    return DataBundle(
        vocabulary=vocabulary,
        records=by_split,
        train_families=train_families,
        holdout_family=holdout_family,
    )


def make_dataset(
    bundle: DataBundle,
    split: str,
    *,
    max_context: int = 192,
) -> NextStepDataset:
    return NextStepDataset(
        records=bundle.records[split],
        vocabulary=bundle.vocabulary,
        max_context=max_context,
    )


def make_loaders(
    bundle: DataBundle,
    *,
    batch_size: int = 128,
    max_context: int = 192,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for split in bundle.records:
        dataset = make_dataset(bundle, split, max_context=max_context)
        loaders[split] = make_torch_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


NeuralModel = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 3e-3
    weight_decay: float = 0.0
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    log_every: int = 50
    k: int = 3


def train_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    *,
    config: TrainConfig,
    device: torch.device,
    pad_id: int = 0,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        seen = 0
        for step, batch in enumerate(loaders["train"]):
            if config.max_train_batches is not None and step >= config.max_train_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target_id"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item()
            seen += 1
            if config.log_every and (step + 1) % config.log_every == 0:
                print(f"epoch {epoch + 1} step {step + 1} loss {running / seen:.4f}")

        avg = running / max(seen, 1)
        print(f"epoch {epoch + 1} done | train loss {avg:.4f}")
        if "valid" in loaders:
            summary = evaluate_model(
                model,
                loaders["valid"],
                device=device,
                k=config.k,
                max_batches=config.max_eval_batches,
            )
            print(f"epoch {epoch + 1} valid {summary['all']}")
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    k: int = 3,
    max_batches: int | None = None,
) -> dict[str, dict[str, float]]:
    model.eval()
    acc = TopKAccumulator(k=k)
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target_id"]
        families = batch["family"]

        logits = model(input_ids, attention_mask)
        topk = torch.topk(logits, k=min(k, logits.size(-1)), dim=-1).indices.cpu()
        for i in range(len(targets)):
            acc.update(int(targets[i]), topk[i].tolist(), group=families[i])
    return acc.summary()
