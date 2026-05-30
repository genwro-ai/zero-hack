"""Scheduled-sampling (free-running) training for the next-step neural baselines.

The default trainer (`zero_hack.models.common.train_model`) uses pure teacher
forcing: every prediction is conditioned on the *ground-truth* prefix. That
never exposes the model to its own mistakes, so at generation time a single
wrong step pushes the prefix off-distribution and errors compound.

Scheduled sampling (Bengio et al., 2015) closes that train/inference gap. We
roll each sequence out step by step and, with a probability that ramps up over
training, feed the model's *own* predicted step back in as the next input
instead of the gold step - while still scoring against the gold target at every
position. The model thus learns to recover after an error rather than assuming
a perfect history.

This needs full sequences (not the flattened (prefix, target) pairs the default
loader produces), so this module ships its own ``SequenceDataset`` / loader.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from zero_hack.data import FAMILY_TOKENS, SequenceRecord, Vocabulary
from zero_hack.models.common import DataBundle, TrainConfig, evaluate_model


class SequenceDataset:
    """Yields whole sequences as encoded step ids plus the family token id."""

    def __init__(
        self,
        records: list[SequenceRecord],
        vocabulary: Vocabulary,
        max_context: int = 192,
        family_dropout: float = 0.0,
        seed: int = 1729,
    ) -> None:
        self.vocabulary = vocabulary
        self.max_context = max_context
        self.family_dropout = family_dropout
        self.rng = random.Random(seed)
        unknown_token = FAMILY_TOKENS["unknown"]
        self.unknown_id = vocabulary.token_to_id.get(unknown_token, vocabulary.unk_id)
        self.items: list[tuple[list[int], int, str]] = []
        for record in records:
            if not record.steps:
                continue
            step_ids = vocabulary.encode(record.steps)
            family_token = FAMILY_TOKENS.get(record.family, unknown_token)
            family_id = vocabulary.token_to_id.get(family_token, vocabulary.unk_id)
            self.items.append((step_ids, family_id, record.family))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        step_ids, family_id, family = self.items[idx]
        # Family-token dropout: with probability ``family_dropout`` condition on
        # <FAMILY_UNKNOWN> instead of the true family token. The ``family``
        # metadata is left unchanged so per-family eval grouping is unaffected.
        # When p==0.0 the RNG is never touched (NO-OP for the default).
        if self.family_dropout and self.rng.random() < self.family_dropout:
            family_id = self.unknown_id
        return {"step_ids": step_ids, "family_id": family_id, "family": family}


def collate_sequence_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(len(item["step_ids"]) for item in batch)
    step_ids: list[list[int]] = []
    lengths: list[int] = []
    for item in batch:
        ids = item["step_ids"]
        lengths.append(len(ids))
        step_ids.append(ids + [pad_id] * (max_len - len(ids)))
    return {
        "step_ids": torch.tensor(step_ids, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "family_id": torch.tensor([item["family_id"] for item in batch], dtype=torch.long),
        "family": [item["family"] for item in batch],
    }


def make_sequence_loader(
    bundle: DataBundle,
    split: str,
    *,
    batch_size: int = 64,
    max_context: int = 192,
    shuffle: bool | None = None,
    num_workers: int = 0,
    family_dropout: float = 0.0,
) -> DataLoader:
    # Family-token dropout is training-only: never apply it to valid/test/
    # per-family splits (they always condition on the true family token).
    dropout = family_dropout if split == "train" else 0.0
    dataset = SequenceDataset(
        bundle.records[split],
        bundle.vocabulary,
        max_context=max_context,
        family_dropout=dropout,
    )
    pad_id = bundle.vocabulary.pad_id
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda items: collate_sequence_batch(items, pad_id),
    )


def scheduled_sampling_prob(
    epoch: int,
    total_epochs: int,
    max_prob: float,
    schedule: str = "linear",
) -> float:
    """Probability of feeding the model's own prediction at a given epoch.

    ``linear`` ramps 0 -> ``max_prob`` across epochs (epoch 0 is pure teacher
    forcing, the final epoch reaches ``max_prob``). ``constant`` uses
    ``max_prob`` from the first epoch.
    """
    if max_prob <= 0.0:
        return 0.0
    max_prob = min(max_prob, 1.0)
    if schedule == "constant":
        return max_prob
    if schedule == "linear":
        if total_epochs <= 1:
            return max_prob
        return max_prob * (epoch / (total_epochs - 1))
    raise ValueError(f"Unknown scheduled-sampling schedule: {schedule!r}")


def _scheduled_sampling_batch(
    model: nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
    ss_prob: float,
    bos_id: int,
    max_context: int,
) -> tuple[float, int]:
    """Roll one batch of sequences out left-to-right and accumulate gradients.

    At step ``t`` every still-active sequence has an identical context length
    (``2 + t``: ``<BOS>``, family token, then ``t`` already-chosen steps), so a
    single batched forward pass covers them. Each step's loss is backpropagated
    immediately; the chosen-token ids fed forward are detached, so the per-step
    graphs are independent and only one is alive at a time.
    """
    step_ids = batch["step_ids"].to(device)  # [B, L]
    lengths = batch["lengths"].to(device)  # [B]
    family_id = batch["family_id"].to(device)  # [B]
    batch_size = step_ids.size(0)
    total_tokens = int(lengths.sum().item())
    if total_tokens == 0:
        return 0.0, 0

    bos_col = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    head = torch.cat([bos_col, family_id.unsqueeze(1)], dim=1)  # [B, 2]
    # `chosen` starts as ground truth; positions are overwritten as we step so
    # that future contexts see whatever was actually fed forward.
    chosen = step_ids.clone()

    optimizer.zero_grad()
    loss_total = 0.0
    max_len = int(lengths.max().item())
    for t in range(max_len):
        active = lengths > t
        if not bool(active.any()):
            break

        context = torch.cat([head, chosen[:, :t]], dim=1)  # [B, 2 + t]
        if context.size(1) > max_context:
            context = context[:, -max_context:]
        context = context[active]
        attention_mask = torch.ones_like(context, dtype=torch.bool)

        logits = model(context, attention_mask)  # [active, vocab]
        target = step_ids[active, t]
        loss = criterion(logits, target)
        (loss / total_tokens).backward()
        loss_total += float(loss.item())

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            if ss_prob > 0.0:
                use_pred = torch.rand(pred.size(0), device=device) < ss_prob
                next_tok = torch.where(use_pred, pred, target)
            else:
                next_tok = target
            chosen[active, t] = next_tok

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss_total, total_tokens


def train_model_scheduled_sampling(
    model: nn.Module,
    train_loader: DataLoader,
    vocabulary: Vocabulary,
    *,
    config: TrainConfig,
    device: torch.device,
    eval_loader: DataLoader | None = None,
    max_context: int = 192,
) -> nn.Module:
    """Train ``model`` with scheduled sampling over full sequences.

    ``eval_loader`` (if given) is a standard next-step loader used for the
    teacher-forced per-epoch validation print, matching the default trainer.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(reduction="sum")
    bos_id = vocabulary.bos_id

    for epoch in range(config.epochs):
        ss_prob = scheduled_sampling_prob(
            epoch, config.epochs, config.ss_max_prob, config.ss_schedule
        )
        model.train()
        running = 0.0
        tokens_seen = 0
        for step, batch in enumerate(_limited(train_loader, config.max_train_batches)):
            loss_sum, n_tokens = _scheduled_sampling_batch(
                model,
                batch,
                optimizer,
                criterion,
                device=device,
                ss_prob=ss_prob,
                bos_id=bos_id,
                max_context=max_context,
            )
            running += loss_sum
            tokens_seen += n_tokens
            if config.log_every and (step + 1) % config.log_every == 0:
                avg = running / max(tokens_seen, 1)
                print(f"epoch {epoch + 1} step {step + 1} ss_p {ss_prob:.3f} loss {avg:.4f}")

        avg = running / max(tokens_seen, 1)
        print(f"epoch {epoch + 1} done | ss_p {ss_prob:.3f} | train loss {avg:.4f}")
        if eval_loader is not None:
            summary = evaluate_model(
                model,
                eval_loader,
                device=device,
                k=config.k,
                max_batches=config.max_eval_batches,
            )
            print(f"epoch {epoch + 1} valid {summary['all']}")
    return model


def _limited(loader: Iterable[Any], max_batches: int | None) -> Iterable[Any]:
    if max_batches is None:
        yield from loader
        return
    for step, batch in enumerate(loader):
        if step >= max_batches:
            break
        yield batch
