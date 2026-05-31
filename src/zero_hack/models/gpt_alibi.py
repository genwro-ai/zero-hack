"""GPT-ALiBi decoder for the leave-one-family-out holdout runner.

This is the transformer the report selects in section 3 ("GPT-ALiBi"). It is kept
separate from :mod:`zero_hack.models.gpt.model` (which uses learned absolute
position embeddings and backs the DPO pipeline) so the two architectures stay
independent. The model uses ALiBi attention bias instead of position embeddings,
which keeps generation stable across prefix lengths and on the held-out family.

The public surface is small on purpose:

* :class:`AlibiGPTConfig` / :class:`AlibiGPTModel` — the network.
* :func:`train_gpt_alibi_adapter` — fit on a :class:`DataBundle` and return an
  adapter.
* :class:`GPTAlibiAdapter` — exposes ``predict_topk(family, prefix, k)`` and
  ``score_sequence(family, steps)``, the same contract as the classic baselines,
  so ``scripts/run_holdout_experiments.py`` can score it through the shared
  prediction/threshold/scoring machinery.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from zero_hack.data import (
    FAMILY_TOKENS,
    SPECIAL_TOKENS,
    SequenceRecord,
    Vocabulary,
)
from zero_hack.models.common import DataBundle, count_parameters, pick_device

SEQUENCE_TERMINATOR = "SHIP LOT"
CAUSAL_IGNORE = -100


@dataclass
class AlibiGPTConfig:
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_context: int = 256
    tie_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _alibi_slopes(nhead: int, device: torch.device) -> torch.Tensor:
    start = 2.0 ** (-8.0 / nhead)
    return torch.tensor([start ** (i + 1) for i in range(nhead)], device=device)


class _AlibiAttention(nn.Module):
    def __init__(self, config: AlibiGPTConfig) -> None:
        super().__init__()
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, seq_len, 3, self.nhead, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.proj(attended)


class _GPTBlock(nn.Module):
    def __init__(self, config: AlibiGPTConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.attn = _AlibiAttention(config)
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.ln_attn(hidden), attn_mask)
        hidden = hidden + self.mlp(self.ln_mlp(hidden))
        return hidden


class AlibiGPTModel(nn.Module):
    """Decoder-only model with ALiBi attention for next-step prediction."""

    def __init__(self, vocab_size: int, config: AlibiGPTConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(_GPTBlock(config) for _ in range(config.num_layers))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.token_embedding.weight

    def _hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_context:
            input_ids = input_ids[:, -self.config.max_context :]
            attention_mask = attention_mask[:, -self.config.max_context :]
            seq_len = input_ids.size(1)

        device = input_ids.device
        hidden = self.dropout(self.token_embedding(input_ids))

        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        keep = causal[None, None] & attention_mask[:, None, None, :]
        positions = torch.arange(seq_len, device=device)
        dist = (positions[:, None] - positions[None, :]).float()
        bias = (-_alibi_slopes(self.config.nhead, device)[:, None, None] * dist).unsqueeze(0)
        blocked = torch.zeros_like(keep, dtype=torch.float).masked_fill(~keep, -torch.inf)
        attn_mask = bias + blocked

        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        return self.ln_f(hidden)

    def forward_all(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for every input position."""
        return self.head(self._hidden(input_ids, attention_mask))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward_all(input_ids, attention_mask)
        if input_ids.shape[1] > self.config.max_context:
            attention_mask = attention_mask[:, -self.config.max_context :]
        last_valid = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        return logits[batch_idx, last_valid, :]


def _invalid_prediction_ids(vocabulary: Vocabulary) -> list[int]:
    invalid = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
    return [vocabulary.token_to_id[token] for token in invalid if token in vocabulary.token_to_id]


def _encode_prefix(
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str] | tuple[str, ...],
    *,
    max_context: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    family_token = FAMILY_TOKENS.get(family.lower(), FAMILY_TOKENS["unknown"])
    tokens = ["<BOS>", family_token, *prefix][-max_context:]
    input_ids = torch.tensor([vocabulary.encode(tokens)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    return input_ids, attention_mask


class _SequenceDataset(Dataset):
    """Full-sequence rows ``[<BOS>, <FAMILY>, *steps, <EOS>]`` for causal LM training."""

    def __init__(
        self,
        records: list[SequenceRecord],
        vocabulary: Vocabulary,
        max_len: int,
    ) -> None:
        self.rows: list[list[int]] = []
        for record in records:
            tokens = [
                "<BOS>",
                FAMILY_TOKENS.get(record.family, FAMILY_TOKENS["unknown"]),
                *record.steps,
                "<EOS>",
            ]
            ids = vocabulary.encode(tokens)[:max_len]
            if len(ids) >= 3:
                self.rows.append(ids)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> list[int]:
        return self.rows[idx]


def _collate_sequences(batch: list[list[int]], pad_id: int) -> tuple[torch.Tensor, ...]:
    width = max(len(ids) for ids in batch) - 1
    inputs, labels, mask = [], [], []
    for ids in batch:
        prefix = ids[:-1]
        pad = width - len(prefix)
        inputs.append(prefix + [pad_id] * pad)
        # Drop the BOS->FAMILY transition from the loss so we only learn step tokens.
        labels.append([CAUSAL_IGNORE] + ids[2:] + [CAUSAL_IGNORE] * pad)
        mask.append([1] * len(prefix) + [0] * pad)
    return (
        torch.tensor(inputs),
        torch.tensor(labels),
        torch.tensor(mask, dtype=torch.bool),
    )


def _lr_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def _causal_loss(model: AlibiGPTModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    use_cuda = device.type == "cuda"
    total, seen = 0.0, 0
    for inputs, labels, mask in loader:
        inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
            logits = model.forward_all(inputs, mask)
            loss = F.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), ignore_index=CAUSAL_IGNORE
            )
        total += float(loss.item())
        seen += 1
    return total / max(1, seen)


def fit_gpt_alibi(
    model: AlibiGPTModel,
    train_records: list[SequenceRecord],
    valid_records: list[SequenceRecord],
    vocabulary: Vocabulary,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    num_workers: int = 0,
    label_smoothing: float = 0.02,
) -> AlibiGPTModel:
    model = model.to(device)
    train_data = _SequenceDataset(train_records, vocabulary, model.config.max_context)
    if len(train_data) == 0:
        return model
    use_cuda = device.type == "cuda"
    collate = partial(_collate_sequences, pad_id=vocabulary.pad_id)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    valid_data = _SequenceDataset(valid_records, vocabulary, model.config.max_context)
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    total = max(1, len(train_loader) * epochs)
    warmup = max(1, int(0.05 * total))
    scheduler = LambdaLR(optimizer, lambda s: _lr_lambda(s, warmup_steps=warmup, total_steps=total))
    best_loss, best_state, stale = math.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for inputs, labels, mask in train_loader:
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                logits = model.forward_all(inputs, mask)
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    labels.flatten(),
                    ignore_index=CAUSAL_IGNORE,
                    label_smoothing=label_smoothing,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
        train_loss = running / max(1, len(train_loader))
        valid_loss = _causal_loss(model, valid_loader, device) if len(valid_data) else train_loss
        print(f"  gpt-alibi epoch {epoch}/{epochs} train={train_loss:.4f} valid={valid_loss:.4f}")
        if valid_loss < best_loss - 1e-3:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch} best_valid={best_loss:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class GPTAlibiAdapter:
    """Wrap a trained :class:`AlibiGPTModel` in the classic-baseline model contract."""

    def __init__(self, model: AlibiGPTModel, vocabulary: Vocabulary, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.vocabulary = vocabulary
        self.device = device
        self.invalid_ids = _invalid_prediction_ids(vocabulary)

    @torch.no_grad()
    def predict_topk(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        k: int = 3,
    ) -> list[str]:
        input_ids, mask = _encode_prefix(
            self.vocabulary,
            family,
            list(prefix_steps),
            max_context=self.model.config.max_context,
            device=self.device,
        )
        logits = self.model(input_ids, mask).squeeze(0)
        if self.invalid_ids:
            logits[torch.tensor(self.invalid_ids, device=self.device)] = -torch.inf
        top = torch.topk(logits, min(k, logits.numel())).indices.tolist()
        return [self.vocabulary.id_to_token[i] for i in top]

    @torch.no_grad()
    def score_sequence(self, family: str, steps: list[str] | tuple[str, ...]) -> float:
        steps = list(steps)
        if not steps:
            return 0.0
        input_ids, mask = _encode_prefix(
            self.vocabulary,
            family,
            steps,
            max_context=self.model.config.max_context,
            device=self.device,
        )
        log_probs = F.log_softmax(self.model.forward_all(input_ids, mask)[0, :-1], dim=-1)
        chosen = log_probs.gather(1, input_ids[0, 1:, None]).squeeze(1)
        # chosen[0] is P(<FAMILY> | <BOS>); chosen[1:] are the step log-probs.
        return float(chosen[1:].sum())


def train_gpt_alibi_adapter(
    bundle: DataBundle,
    train_records: list[SequenceRecord],
    *,
    config: AlibiGPTConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> GPTAlibiAdapter:
    device = device or pick_device(None)
    model = AlibiGPTModel(
        vocab_size=len(bundle.vocabulary.id_to_token),
        config=config,
        pad_id=bundle.vocabulary.pad_id,
    )
    print(f"  gpt-alibi parameters={count_parameters(model)} device={device}")
    model = fit_gpt_alibi(
        model,
        train_records,
        bundle.records["valid"],
        bundle.vocabulary,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        patience=patience,
        num_workers=num_workers,
    )
    return GPTAlibiAdapter(model, bundle.vocabulary, device)
