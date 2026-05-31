"""Inference-time family inference via prefix-likelihood Bayesian model averaging.

The sequence models condition generation on a ``<FAMILY_*>`` token. At eval we
may face an *unknown / new* family for which no token was ever trained. Rather
than retrain, we treat the family as a latent variable and infer it from the
observed prefix, which is itself family-discriminative:

    P(f | prefix) ∝ exp( prefix_logprob(prefix | f) )            # over candidates
    p(next | prefix) = Σ_f P(f | prefix) · p(next | prefix, f)   # BMA mixture

This module provides a thin ``FamilyScorer`` protocol, an ``LSTMFamilyScorer``
adapter over :class:`~zero_hack.models.lstm.inference.LSTMInference`, an optional
transformer adapter, and a :class:`FamilyRouter` that computes the posterior over
families and the Bayesian-model-averaged next-step distribution.

Usage by task:

* **Task 1 (next-step):** use the BMA distribution directly. Call
  :meth:`FamilyRouter.bma_next_logprobs` / :meth:`FamilyRouter.predict_topk`,
  which marginalise over the family posterior at the given prefix.
* **Task 2 (completion):** call :meth:`FamilyRouter.route` once on the given
  prefix to pick the single most-likely family token, then roll out the
  completion under that *fixed* family (e.g. via ``LSTMInference.predict_topk``
  or the shared ``complete_sequence`` helper). Re-routing every step is both
  slower and less stable; the prefix already determines the family.
* **Task 3 (full sequence):** treat the whole sequence as the prefix for routing
  (likelihood-based family identification / anomaly scoring).

The prefix is always a list of *real step strings* — no ``<BOS>`` and no family
token. Those are added inside the scorer/inference layer.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from zero_hack.data import Vocabulary
from zero_hack.models.lstm.inference import LSTMInference

DEFAULT_FAMILIES: tuple[str, ...] = ("mosfet", "igbt", "ic")


@runtime_checkable
class FamilyScorer(Protocol):
    """Minimal interface a model must expose to be routed over families."""

    vocabulary: Vocabulary

    def prefix_logprob(self, family: str, prefix: list[str]) -> float:
        """Total (unmasked) teacher-forcing log-prob of ``prefix`` under ``family``."""
        ...

    def next_logits(self, family: str, prefix: list[str]) -> torch.Tensor:
        """Next-step logits over the vocab with non-step tokens masked to ``-inf``.

        Matches ``LSTMInference._logits`` semantics: special and family tokens
        are already masked out so only real process steps carry mass.
        """
        ...


class LSTMFamilyScorer:
    """Adapt an :class:`LSTMInference` to the :class:`FamilyScorer` protocol."""

    def __init__(self, inference: LSTMInference) -> None:
        self.inference = inference
        self.vocabulary = inference.vocabulary

    def prefix_logprob(self, family: str, prefix: list[str]) -> float:
        # score_sequence already does teacher-forced sum of log-probs (unmasked).
        return self.inference.score_sequence(family, prefix)

    def next_logits(self, family: str, prefix: list[str]) -> torch.Tensor:
        return self.inference._logits(family, prefix)


class TransformerFamilyScorer:
    """Optional adapter for a decoder/transformer wrapper.

    Expects an object exposing the same ``score_sequence(family, steps) -> float``
    and ``_logits(family, prefix) -> Tensor[vocab]`` (masked) surface as
    :class:`LSTMInference`. Kept here so a transformer can be routed identically;
    the LSTM adapter is the primary, tested path.
    """

    def __init__(self, inference: object) -> None:
        if not (hasattr(inference, "score_sequence") and hasattr(inference, "_logits")):
            raise TypeError(
                "TransformerFamilyScorer expects an inference object with "
                "`score_sequence` and `_logits` methods."
            )
        self.inference = inference
        self.vocabulary = inference.vocabulary  # type: ignore[attr-defined]

    def prefix_logprob(self, family: str, prefix: list[str]) -> float:
        return float(self.inference.score_sequence(family, prefix))  # type: ignore[attr-defined]

    def next_logits(self, family: str, prefix: list[str]) -> torch.Tensor:
        return self.inference._logits(family, prefix)  # type: ignore[attr-defined]


class FamilyRouter:
    """Route an unknown-family prefix via prefix-likelihood BMA over families."""

    def __init__(
        self,
        scorer: FamilyScorer,
        families: tuple[str, ...] = DEFAULT_FAMILIES,
        *,
        prior: dict[str, float] | None = None,
        temperature: float = 1.0,
        include_unknown: bool = False,
        vocabulary: Vocabulary | None = None,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")

        candidates = list(families)
        if include_unknown and "unknown" not in candidates:
            candidates.append("unknown")
        if not candidates:
            raise ValueError("FamilyRouter needs at least one candidate family")
        self.families = tuple(candidates)
        self.temperature = float(temperature)

        self.scorer = scorer
        self.vocabulary = vocabulary or getattr(scorer, "vocabulary", None)
        if self.vocabulary is None:
            raise ValueError("No vocabulary available; pass `vocabulary=` explicitly.")

        # Normalised log prior over candidate families (uniform by default).
        if prior is None:
            log_p = -math.log(len(self.families))
            self._log_prior = {f: log_p for f in self.families}
        else:
            weights = torch.tensor(
                [max(float(prior.get(f, 0.0)), 0.0) for f in self.families],
                dtype=torch.float64,
            )
            if float(weights.sum()) <= 0.0:
                raise ValueError("prior must have positive mass on some candidate family")
            log_prior = torch.log(weights / weights.sum())
            self._log_prior = {
                f: float(lp) for f, lp in zip(self.families, log_prior.tolist(), strict=True)
            }

    def _posterior_logits(self, prefix: list[str]) -> dict[str, float]:
        return {
            family: self.scorer.prefix_logprob(family, prefix) / self.temperature
            + self._log_prior[family]
            for family in self.families
        }

    def posterior(self, prefix: list[str]) -> dict[str, float]:
        """Normalised P(family | prefix) over candidate families (sums to 1)."""
        logits = self._posterior_logits(prefix)
        vec = torch.tensor([logits[f] for f in self.families], dtype=torch.float64)
        probs = torch.softmax(vec, dim=-1)
        return {f: float(p) for f, p in zip(self.families, probs.tolist(), strict=True)}

    def route(self, prefix: list[str]) -> str:
        """Argmax family of the posterior — the single best conditioning family."""
        post = self.posterior(prefix)
        return max(post, key=post.__getitem__)

    def bma_next_logprobs(self, prefix: list[str]) -> torch.Tensor:
        """Log of the BMA next-step distribution over the vocab.

        ``log p(next | prefix) = logsumexp_f( log P(f|prefix) + log_softmax(logits_f) )``,
        computed in log space for numerical stability. The result is a valid
        log-prob vector over the vocab (``logsumexp == ~0``); masked tokens stay
        at ``-inf``.
        """
        post = self.posterior(prefix)
        components: list[torch.Tensor] = []
        for family in self.families:
            weight = post[family]
            if weight <= 0.0:
                continue
            log_probs = F.log_softmax(
                self.scorer.next_logits(family, prefix).to(torch.float64), dim=-1
            )
            components.append(math.log(weight) + log_probs)
        if not components:  # pragma: no cover - posterior always has mass
            raise RuntimeError("Empty family posterior; cannot form BMA distribution.")
        stacked = torch.stack(components, dim=0)  # [n_families, vocab]
        return torch.logsumexp(stacked, dim=0)

    def predict_topk(self, prefix: list[str], k: int = 5) -> list[str]:
        """Top-``k`` next steps under the BMA distribution, as step strings."""
        log_probs = self.bma_next_logprobs(prefix)
        k = min(k, log_probs.size(-1))
        top_ids = torch.topk(log_probs, k=k).indices.tolist()
        return [self.vocabulary.id_to_token[i] for i in top_ids]


def _smoke_test() -> None:
    """Tiny CPU smoke test exercising the real LSTM + checkpoint path.

    Builds an in-memory LSTM from ~40 real sequences (the legacy checkpoint at
    outputs/models/valid_s005k/lstm/best.pt lacks ``token_to_id`` and cannot be
    loaded), saves/reloads it through ``save_lstm_checkpoint`` to exercise the
    real path, then validates the router API. The model is randomly initialised
    and minimally trained, so any family preference is a sanity signal only.
    """
    import tempfile
    from pathlib import Path

    from zero_hack.data import build_vocabulary, load_sequence_records
    from zero_hack.models.lstm.inference import (
        load_lstm_checkpoint,
        save_lstm_checkpoint,
    )
    from zero_hack.models.lstm.model import LSTMConfig, LSTMModel

    splits = Path("data/generated/valid_s005k/splits")
    records = []
    for family, fname in (
        ("mosfet", "MOSFET_train.csv"),
        ("igbt", "IGBT_train.csv"),
        ("ic", "IC_train.csv"),
    ):
        recs = load_sequence_records(splits / fname, family=family)
        records.extend(recs[:14])  # ~40 sequences total
    assert records, "no records loaded for smoke test"

    vocab = build_vocabulary(records)
    torch.manual_seed(0)
    model = LSTMModel(
        vocab_size=len(vocab.id_to_token),
        config=LSTMConfig(embedding_dim=32, hidden_dim=32, num_layers=1, dropout=0.0),
        pad_id=vocab.pad_id,
    )

    # Save + reload through the real checkpoint path.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = save_lstm_checkpoint(
            Path(tmp) / "best.pt", model, vocab, max_context=64, meta={"smoke": True}
        )
        inference = load_lstm_checkpoint(ckpt, device="cpu")

    scorer = LSTMFamilyScorer(inference)
    router = FamilyRouter(scorer, families=("mosfet", "igbt", "ic"))

    mosfet_prefix = list(records[0].steps[:6])
    assert mosfet_prefix, "empty prefix"

    post = router.posterior(mosfet_prefix)
    assert set(post) == {"mosfet", "igbt", "ic"}, post
    assert abs(sum(post.values()) - 1.0) < 1e-6, post

    chosen = router.route(mosfet_prefix)
    assert chosen in router.families, chosen

    logp = router.bma_next_logprobs(mosfet_prefix)
    assert logp.shape == (len(vocab.id_to_token),), logp.shape
    total = float(torch.logsumexp(logp, dim=0))
    assert abs(total) < 1e-4, f"BMA logsumexp != 0: {total}"

    topk = router.predict_topk(mosfet_prefix, k=5)
    assert len(topk) == 5, topk
    assert all(isinstance(s, str) for s in topk), topk

    print("posterior:", {k: round(v, 4) for k, v in post.items()})
    print("route ->", chosen, "(MOSFET prefix; random tiny model, sanity only)")
    print("bma logsumexp:", round(total, 8))
    print("predict_topk:", topk)
    print("smoke test OK")


if __name__ == "__main__":
    _smoke_test()
