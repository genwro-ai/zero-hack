"""Shared plumbing for the next-step models (neural + symbolic).

This module owns everything the individual model packages should *not*
re-implement: loading raw family CSVs into deterministic, leakage-free
train/valid/test splits, building the Torch dataloaders, and the generic
neural train / evaluate / top-k loops that operate on any ``nn.Module``
returning next-step logits of shape ``[batch, vocab_size]``.

The four model packages (``transformer``, ``lstm``, ``gru``, ``ngram``) each
own only their architecture plus a thin CLI; they delegate data and training
here so every baseline is compared on identical splits and metrics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from zero_hack import PROJECT_ROOT
from zero_hack.data import (
    NextStepDataset,
    SequenceRecord,
    Vocabulary,
    build_vocabulary,
    dedupe_records,
    load_raw_family_records,
    make_torch_dataloader,
)
from zero_hack.metrics import TopKAccumulator
from zero_hack.splits import SPLIT_NAMES, split_for
from zero_hack.vocab import FAMILIES

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "generated" / "valid_s005k" / "raw"


# --------------------------------------------------------------------------- #
# Data: deterministic, dedup-aware splits shared by every model               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataBundle:
    """Everything a model needs: the vocab plus split -> records."""

    vocabulary: Vocabulary
    records: dict[str, list[SequenceRecord]]

    def counts(self) -> dict[str, int]:
        return {name: len(self.records[name]) for name in SPLIT_NAMES}


def load_record_splits(
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    *,
    families: tuple[str, ...] = FAMILIES,
    limit_per_family: int | None = None,
) -> DataBundle:
    """Load raw family CSVs, dedupe, and split by content hash (no leakage).

    The vocabulary is built from the *train* split only. ``limit_per_family``
    truncates each family before splitting to keep smoke runs fast.
    """
    raw_dir = Path(raw_dir)
    records: list[SequenceRecord] = []
    for family in families:
        fam_records = load_raw_family_records(raw_dir, family)
        if limit_per_family is not None:
            fam_records = fam_records[:limit_per_family]
        records.extend(fam_records)

    records = dedupe_records(records)

    by_split: dict[str, list[SequenceRecord]] = {name: [] for name in SPLIT_NAMES}
    for record in records:
        by_split[split_for(record.family, list(record.steps))].append(record)

    vocabulary = build_vocabulary(by_split["train"])
    return DataBundle(vocabulary=vocabulary, records=by_split)


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
    """Build train/valid/test dataloaders from a :class:`DataBundle`."""
    loaders: dict[str, DataLoader] = {}
    for split in SPLIT_NAMES:
        dataset = make_dataset(bundle, split, max_context=max_context)
        loaders[split] = make_torch_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders


# --------------------------------------------------------------------------- #
# Devices                                                                      #
# --------------------------------------------------------------------------- #
def pick_device(prefer: str | None = None) -> torch.device:
    """Return the best available device (cuda > mps > cpu), honoring ``prefer``."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Generic neural training / evaluation                                        #
# --------------------------------------------------------------------------- #
# A model forward takes (input_ids[B,T], attention_mask[B,T] bool) and returns
# next-step logits of shape [B, vocab_size].
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
    """Minimal next-step training loop. Returns the trained model (in place)."""
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
    """Top-1 / top-k next-step accuracy, broken down by family."""
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
