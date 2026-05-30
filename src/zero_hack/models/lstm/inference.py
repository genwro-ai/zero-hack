"""Checkpoint save/load and autoregressive inference for the LSTM baseline.

The trained ``LSTMModel`` only exposes ``forward`` (next-step logits). To run
the sequence-completion task (Task 2) we wrap it in ``LSTMInference``, which
implements the same ``predict_topk`` / ``score_sequence`` interface as the
classic baselines (see ``zero_hack.models.classic_baselines``). That lets us
reuse ``complete_sequence`` and the existing completion metrics unchanged.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, Vocabulary
from zero_hack.models.lstm.model import LSTMConfig, LSTMModel
from zero_hack.models.violation_mask import ViolationMask


def save_lstm_checkpoint(
    path: str | Path,
    model: LSTMModel,
    vocabulary: Vocabulary,
    *,
    max_context: int,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Persist model weights + vocab + config to ``path`` (a ``.pt`` file).

    A sibling ``vocab.json`` is written next to it for transparency/debugging.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "lstm_config": dataclasses.asdict(model.config),
        "token_to_id": dict(vocabulary.token_to_id),
        "id_to_token": list(vocabulary.id_to_token),
        "max_context": max_context,
        "meta": meta or {},
    }
    torch.save(payload, path)
    vocab_path = path.parent / "vocab.json"
    vocab_path.write_text(
        json.dumps({"id_to_token": list(vocabulary.id_to_token)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_lstm_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    enforce_rules: bool = False,
) -> LSTMInference:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    vocabulary = Vocabulary(
        token_to_id=dict(payload["token_to_id"]),
        id_to_token=tuple(payload["id_to_token"]),
    )
    config = LSTMConfig(**payload["lstm_config"])
    model = LSTMModel(
        vocab_size=len(vocabulary.id_to_token),
        config=config,
        pad_id=vocabulary.pad_id,
    )
    model.load_state_dict(payload["model_state"])
    return LSTMInference(
        model,
        vocabulary,
        device=device,
        max_context=int(payload.get("max_context", 192)),
        meta=payload.get("meta", {}),
        enforce_rules=enforce_rules,
    )


class LSTMInference:
    """Autoregressive inference wrapper around a trained ``LSTMModel``.

    Mirrors ``zero_hack.models.classic_baselines.ClassicBaselineModel`` so the
    shared ``complete_sequence`` rollout helper works on it directly. Special
    and family tokens are masked out of predictions so generation only ever
    emits real process steps.
    """

    def __init__(
        self,
        model: LSTMModel,
        vocabulary: Vocabulary,
        *,
        device: torch.device | str = "cpu",
        max_context: int = 192,
        meta: dict[str, Any] | None = None,
        enforce_rules: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.vocabulary = vocabulary
        self.max_context = max_context
        self.meta = meta or {}

        blocked = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
        self._blocked_ids = [
            vocabulary.token_to_id[token] for token in blocked if token in vocabulary.token_to_id
        ]
        self.violation_mask = ViolationMask(vocabulary.id_to_token) if enforce_rules else None

    def _family_token(self, family: str) -> str:
        return FAMILY_TOKENS.get(family.lower(), FAMILY_TOKENS["unknown"])

    def _context_ids(self, family: str, prefix_steps: list[str] | tuple[str, ...]) -> list[int]:
        tokens = ["<BOS>", self._family_token(family), *prefix_steps]
        tokens = tokens[-self.max_context :]
        return self.vocabulary.encode(tokens)

    @torch.no_grad()
    def _logits(self, family: str, prefix_steps: list[str] | tuple[str, ...]) -> torch.Tensor:
        ids = self._context_ids(family, prefix_steps)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        logits = self.model(input_ids, attention_mask)[0]  # [vocab]
        if self._blocked_ids:
            logits[self._blocked_ids] = float("-inf")
        if self.violation_mask is not None:
            logits = self.violation_mask(logits, list(prefix_steps))
        return logits

    def predict_topk(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        k: int = 3,
    ) -> list[str]:
        logits = self._logits(family, prefix_steps)
        k = min(k, logits.size(-1))
        top_ids = torch.topk(logits, k=k).indices.tolist()
        return [self.vocabulary.id_to_token[i] for i in top_ids]

    @torch.no_grad()
    def score_sequence(
        self,
        family: str,
        steps: list[str] | tuple[str, ...],
    ) -> float:
        """Total log-probability of ``steps`` under teacher forcing.

        Special/family tokens are *not* masked here — this scores the actual
        next-token distribution (used for likelihood-based anomaly detection).
        """
        total = 0.0
        prefix: list[str] = []
        for step in steps:
            ids = self._context_ids(family, prefix)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            log_probs = F.log_softmax(self.model(input_ids, attention_mask)[0], dim=-1)
            step_id = self.vocabulary.token_to_id.get(step, self.vocabulary.unk_id)
            total += float(log_probs[step_id].item())
            prefix.append(step)
        return total
