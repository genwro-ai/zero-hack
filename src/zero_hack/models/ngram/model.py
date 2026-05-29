"""Per-family symbolic n-gram model with stupid-backoff.

A counting-based next-step baseline. We keep, *per family*, one ``Counter`` of
next steps for every observed context of length ``0 .. n-1`` (length 0 being the
unigram distribution). Prediction uses stupid backoff: score a candidate with
the highest-order context that exists, discounting by ``alpha`` for every order
we have to back off.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from zero_hack.data import SequenceRecord
from zero_hack.metrics import TopKAccumulator

BOS_TOKEN = "<BOS>"

# Floor probability used by ``score_sequence`` to avoid ``log(0)`` for steps a
# family never produced during training.
_SCORE_FLOOR = 1e-9


class NGramModel:
    """Counting n-gram model (default ``n=5``) with stupid backoff.

    Counts are stored per family. For each context length ``order`` in
    ``0 .. n-1`` we map ``context_tuple -> Counter[next_step]``. The order-0
    context is the empty tuple, i.e. the unigram distribution.
    """

    def __init__(self, n: int = 5, backoff_alpha: float = 0.4) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.backoff_alpha = backoff_alpha
        # family -> order -> context_tuple -> Counter[next_step]
        self._contexts: dict[str, list[dict[tuple[str, ...], Counter]]] = {}
        # family -> Counter[next_step]  (unigram convenience handle = order 0)
        self._unigrams: dict[str, Counter] = {}
        # Global fallbacks (across all families) for robustness to unseen family.
        self._global_contexts: list[dict[tuple[str, ...], Counter]] = [
            defaultdict(Counter) for _ in range(self.n)
        ]
        self._global_unigram: Counter = Counter()

    # ------------------------------------------------------------------ #
    # Fitting                                                            #
    # ------------------------------------------------------------------ #
    def _padded(self, steps: tuple[str, ...] | list[str]) -> list[str]:
        """Left-pad a sequence with ``n-1`` BOS tokens."""
        return [BOS_TOKEN] * (self.n - 1) + list(steps)

    def fit(self, records: list[SequenceRecord]) -> NGramModel:
        contexts: dict[str, list[dict[tuple[str, ...], Counter]]] = {}
        unigrams: dict[str, Counter] = {}

        def family_tables(family: str) -> list[dict[tuple[str, ...], Counter]]:
            if family not in contexts:
                contexts[family] = [defaultdict(Counter) for _ in range(self.n)]
                unigrams[family] = contexts[family][0][()]
            return contexts[family]

        for record in records:
            tables = family_tables(record.family)
            padded = self._padded(record.steps)
            for pos in range(self.n - 1, len(padded)):
                nxt = padded[pos]
                for order in range(self.n):
                    ctx = tuple(padded[pos - order : pos])
                    tables[order][ctx][nxt] += 1
                    self._global_contexts[order][ctx][nxt] += 1

        self._contexts = contexts
        self._unigrams = unigrams
        # Order-0 global counter (empty context) doubles as global unigram.
        self._global_unigram = self._global_contexts[0][()]
        return self

    # ------------------------------------------------------------------ #
    # Internal scoring helpers                                           #
    # ------------------------------------------------------------------ #
    def _tables_for(self, family: str) -> list[dict[tuple[str, ...], Counter]] | None:
        return self._contexts.get(family)

    def _backoff_scores(self, family: str, context: tuple[str, ...]) -> dict[str, float]:
        """Stupid-backoff score per candidate next step for a given context.

        Walks from the highest available order down to the unigram, taking the
        first non-empty context counter and discounting by ``alpha`` per backoff.
        """
        tables = self._tables_for(family)
        if tables is None:
            tables = self._global_contexts

        # context already trimmed to <= n-1 tokens; try longest first.
        for backed_off in range(len(context) + 1):
            order = len(context) - backed_off
            ctx = context[backed_off:] if order > 0 else ()
            counter = tables[order].get(ctx)
            if counter:
                total = sum(counter.values())
                weight = self.backoff_alpha**backed_off
                return {step: weight * (cnt / total) for step, cnt in counter.items()}

        # Family had no data at all; fall back to global unigram.
        if self._global_unigram:
            total = sum(self._global_unigram.values())
            weight = self.backoff_alpha ** len(context)
            return {step: weight * (cnt / total) for step, cnt in self._global_unigram.items()}
        return {}

    def _trim_context(self, prefix_steps: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        """Take the last ``n-1`` steps, BOS-padding when the prefix is short."""
        want = self.n - 1
        steps = list(prefix_steps)
        if len(steps) < want:
            steps = [BOS_TOKEN] * (want - len(steps)) + steps
        return tuple(steps[-want:]) if want > 0 else ()

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def predict_topk(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        k: int = 3,
    ) -> list[str]:
        context = self._trim_context(prefix_steps)
        scores = self._backoff_scores(family, context)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [step for step, _ in ranked[:k]]

    def score_sequence(
        self,
        family: str,
        steps: list[str] | tuple[str, ...],
    ) -> float:
        """Total backoff log-probability of ``steps`` for ``family``."""
        padded = self._padded(steps)
        total_logprob = 0.0
        for pos in range(self.n - 1, len(padded)):
            context = tuple(padded[pos - (self.n - 1) : pos])
            nxt = padded[pos]
            scores = self._backoff_scores(family, context)
            prob = scores.get(nxt, 0.0)
            total_logprob += math.log(max(prob, _SCORE_FLOOR))
        return total_logprob

    def evaluate(
        self,
        records: list[SequenceRecord],
        vocabulary,
        k: int = 3,
    ) -> dict:
        acc = TopKAccumulator(k=k)
        for record in records:
            for pos in range(len(record.steps)):
                prefix = record.steps[:pos]
                gold_step = record.steps[pos]
                preds = self.predict_topk(record.family, prefix, k=k)
                gold_id = vocabulary.token_to_id.get(gold_step, vocabulary.unk_id)
                pred_ids = [vocabulary.token_to_id.get(step, vocabulary.unk_id) for step in preds]
                acc.update(gold_id, pred_ids, group=record.family)
        return acc.summary()
