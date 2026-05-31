"""N-gram-normalised ("residual") training for the next-step neural baselines.

The plain trainer fits cross-entropy on the raw model logits, so the network
spends capacity re-learning the easy local structure a count-based n-gram already
captures. Subtracting ``log p_ngram`` *after* training is a no-op for the model
(it is constant w.r.t. the weights). To make the normalisation actually change
what is learned, the n-gram must sit **inside the softmax as a fixed prior**:

    combined_logits = model_logits + log p_ngram(. | context)
    loss            = cross_entropy(combined_logits, target)

This is a product-of-experts / residual model (a.k.a. neural boosting over an
n-gram base): the n-gram explains the local co-occurrence, and the network is
trained to model only the *residual* structure the n-gram misses. The combined
likelihood is what we read off at inference, so the normalisation is baked in and
the whole thing is a drop-in: :class:`ResidualModel` is an ``nn.Module`` whose
``forward(input_ids, attention_mask)`` returns the combined logits, so it slots
straight into ``train_model`` / ``evaluate_model`` / ``LSTMInference`` unchanged.

The n-gram prior is parameter-free and frozen; gradients flow only through the
wrapped network. For an unknown / held-out family the n-gram backs off to its
global counts, exactly as in :class:`~zero_hack.models.ngram.model.NGramModel`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, SequenceRecord, Vocabulary
from zero_hack.models.lstm.inference import (
    LSTMInference,
    load_lstm_checkpoint,
    save_lstm_checkpoint,
)
from zero_hack.models.ngram.model import SCORE_FLOOR, NGramModel

_LOG_FLOOR = math.log(SCORE_FLOOR)


class NGramBias:
    """Builds a frozen ``log p_ngram(. | context)`` bias vector per batch row.

    The context is reconstructed straight from ``input_ids`` (the real step
    tokens, ignoring ``<BOS>`` / specials), and the family from whichever family
    token appears in the row — so this is robust to context truncation and needs
    nothing beyond ``input_ids``. Missing family => the n-gram's global backoff.
    """

    def __init__(self, ngram: NGramModel, vocabulary: Vocabulary) -> None:
        self.ngram = ngram
        self.vocabulary = vocabulary
        self.vocab_size = len(vocabulary.id_to_token)

        special_ids = {
            vocabulary.token_to_id[t] for t in SPECIAL_TOKENS if t in vocabulary.token_to_id
        }
        self._family_id_to_name = {
            vocabulary.token_to_id[token]: name
            for name, token in FAMILY_TOKENS.items()
            if token in vocabulary.token_to_id
        }
        self._non_step_ids = special_ids | set(self._family_id_to_name)

    def _row_bias(self, ids: list[int]) -> list[float]:
        family: str | None = None
        steps: list[str] = []
        for token_id in ids:
            if token_id in self._family_id_to_name:
                family = self._family_id_to_name[token_id]
            elif token_id not in self._non_step_ids:
                steps.append(self.vocabulary.id_to_token[token_id])

        scores = self.ngram._scores(family or "", steps)
        bias = [_LOG_FLOOR] * self.vocab_size
        for token, prob in scores.items():
            idx = self.vocabulary.token_to_id.get(token)
            if idx is not None:
                bias[idx] = math.log(max(prob, SCORE_FLOOR))
        return bias

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        rows = input_ids.detach().cpu().tolist()
        bias = [self._row_bias(row) for row in rows]
        return torch.tensor(bias, dtype=torch.float32, device=input_ids.device)


class ResidualModel(nn.Module):
    """Wrap a next-step network so its logits are normalised by an n-gram prior.

    ``forward`` returns ``network_logits + log p_ngram(. | context)``. The bias is
    frozen (no parameters, computed under ``no_grad``), so this is a transparent
    drop-in for any code that calls ``model(input_ids, attention_mask)``.
    """

    def __init__(self, network: nn.Module, ngram_bias: NGramBias) -> None:
        super().__init__()
        self.network = network
        self.ngram_bias = ngram_bias

    @property
    def config(self) -> Any:
        return self.network.config

    @property
    def pad_id(self) -> int:
        return self.network.pad_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.network(input_ids, attention_mask)
        return logits + self.ngram_bias(input_ids)


def build_ngram(
    records: list[SequenceRecord],
    *,
    n: int,
    alpha: float,
) -> NGramModel:
    """Fit the frozen n-gram base on the training records."""
    return NGramModel(n=n, backoff_alpha=alpha).fit(records)


def wrap_residual(
    network: nn.Module,
    bundle_train_records: list[SequenceRecord],
    vocabulary: Vocabulary,
    *,
    n: int,
    alpha: float,
) -> ResidualModel:
    """Build the n-gram from ``bundle_train_records`` and wrap ``network``."""
    ngram = build_ngram(bundle_train_records, n=n, alpha=alpha)
    return ResidualModel(network, NGramBias(ngram, vocabulary))


_RESIDUAL_META_KEY = "ngram_residual"


def save_residual_checkpoint(
    path: str | Path,
    model: ResidualModel,
    vocabulary: Vocabulary,
    *,
    max_context: int,
    n: int,
    alpha: float,
    splits_dir: str | Path,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Persist the inner network + the n-gram recipe needed to rebuild the prior.

    Only the network weights are saved; the n-gram prior is deterministic given
    ``splits_dir`` + (``n``, ``alpha``), so it is rebuilt at load time. The
    payload is otherwise a normal LSTM checkpoint, so ``detect_checkpoint_kind``
    and friends keep working.
    """
    full_meta = dict(meta or {})
    full_meta[_RESIDUAL_META_KEY] = {
        "n": n,
        "alpha": alpha,
        "splits_dir": str(splits_dir),
    }
    return save_lstm_checkpoint(
        path,
        model.network,  # type: ignore[arg-type]
        vocabulary,
        max_context=max_context,
        meta=full_meta,
    )


def load_residual_inference(
    checkpoint: str | Path,
    *,
    splits_dir: str | Path | None = None,
    device: torch.device | str = "cpu",
    enforce_rules: bool = False,
) -> LSTMInference:
    """Load a residual checkpoint as an :class:`LSTMInference` with the n-gram prior.

    The returned object's ``score_sequence`` / ``predict_topk`` therefore report
    the *combined* (n-gram-normalised) distribution, so every existing eval script
    measures the residual model with no changes.
    """
    inference = load_lstm_checkpoint(checkpoint, device=device, enforce_rules=enforce_rules)
    residual_meta = inference.meta.get(_RESIDUAL_META_KEY)
    if residual_meta is None:
        raise ValueError(
            f"{checkpoint} is not an n-gram-residual checkpoint "
            f"(missing '{_RESIDUAL_META_KEY}' in meta)."
        )

    from zero_hack.models.common import load_split_records

    resolved_splits = splits_dir or residual_meta["splits_dir"]
    bundle = load_split_records(
        resolved_splits, holdout_family=inference.meta.get("holdout_family")
    )
    ngram = build_ngram(bundle.records["train"], n=residual_meta["n"], alpha=residual_meta["alpha"])
    bias = NGramBias(ngram, inference.vocabulary)
    # Splice the prior in front of the loaded network so all of LSTMInference's
    # logit paths (predict_topk, score_sequence) see the combined distribution.
    inference.model = ResidualModel(inference.model, bias).to(inference.device).eval()
    return inference


__all__ = [
    "NGramBias",
    "ResidualModel",
    "build_ngram",
    "load_residual_inference",
    "save_residual_checkpoint",
    "wrap_residual",
]
