"""Most-frequent next-step baseline (the sanity-check baseline).

Per the baseline plan this conditions on ``(family, position bucket, previous
step)`` and backs off to coarser contexts whenever the full context was never
seen during training. It is intentionally the simplest counting model: no
sequential context beyond the immediately previous step, just "what usually
comes next here?".

Counts are stored per family across four context tables, consulted from most to
least specific:

1. ``(position_bucket, prev_step)``  -- full context
2. ``(prev_step,)``                  -- drop the position bucket
3. ``(position_bucket,)``            -- drop the previous step
4. ``()``                            -- family unigram

A final global unigram (pooled across families) covers an unseen family. Each
backoff step discounts by ``backoff_alpha`` (stupid backoff), so
``score_sequence`` stays comparable across contexts for the anomaly setup.
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


class MostFrequentModel:
    """Counting baseline conditioned on family, position bucket and prev step.

    ``position_bucket_size`` controls how finely the step position is binned
    (``position // size``); larger buckets share statistics across nearby
    positions. ``backoff_alpha`` discounts each backoff level.
    """

    def __init__(self, position_bucket_size: int = 5, backoff_alpha: float = 0.4) -> None:
        if position_bucket_size < 1:
            raise ValueError("position_bucket_size must be >= 1")
        self.position_bucket_size = position_bucket_size
        self.backoff_alpha = backoff_alpha
        # family -> (pos_bucket, prev_step) -> Counter[next_step]
        self._pos_prev: dict[str, dict[tuple[int, str], Counter]] = {}
        # family -> prev_step -> Counter[next_step]
        self._prev: dict[str, dict[str, Counter]] = {}
        # family -> pos_bucket -> Counter[next_step]
        self._pos: dict[str, dict[int, Counter]] = {}
        # family -> Counter[next_step]  (unigram)
        self._unigram: dict[str, Counter] = {}
        # Pooled across families, for robustness to an unseen family.
        self._global_unigram: Counter = Counter()

    # ------------------------------------------------------------------ #
    # Fitting                                                            #
    # ------------------------------------------------------------------ #
    def _bucket(self, position: int) -> int:
        return position // self.position_bucket_size

    def fit(self, records: list[SequenceRecord]) -> MostFrequentModel:
        pos_prev: dict[str, dict[tuple[int, str], Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        prev: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        pos: dict[str, dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))
        unigram: dict[str, Counter] = defaultdict(Counter)
        global_unigram: Counter = Counter()

        for record in records:
            family = record.family
            for position, nxt in enumerate(record.steps):
                prev_step = record.steps[position - 1] if position > 0 else BOS_TOKEN
                bucket = self._bucket(position)
                pos_prev[family][(bucket, prev_step)][nxt] += 1
                prev[family][prev_step][nxt] += 1
                pos[family][bucket][nxt] += 1
                unigram[family][nxt] += 1
                global_unigram[nxt] += 1

        self._pos_prev = pos_prev
        self._prev = prev
        self._pos = pos
        self._unigram = unigram
        self._global_unigram = global_unigram
        return self

    # ------------------------------------------------------------------ #
    # Internal scoring helpers                                           #
    # ------------------------------------------------------------------ #
    def _backoff_scores(self, family: str, position: int, prev_step: str) -> dict[str, float]:
        """Discounted next-step scores from the most specific context available."""
        bucket = self._bucket(position)
        # (table, key, backoff_level) from most to least specific.
        candidates = (
            (self._pos_prev.get(family, {}), (bucket, prev_step), 0),
            (self._prev.get(family, {}), prev_step, 1),
            (self._pos.get(family, {}), bucket, 2),
            (self._unigram.get(family), (), 3),
        )
        for table, key, level in candidates:
            if table is None:
                continue
            counter = table if key == () else table.get(key)
            if counter:
                total = sum(counter.values())
                weight = self.backoff_alpha**level
                return {step: weight * (cnt / total) for step, cnt in counter.items()}

        # Family unseen at training time; fall back to the global unigram.
        if self._global_unigram:
            total = sum(self._global_unigram.values())
            weight = self.backoff_alpha**4
            return {step: weight * (cnt / total) for step, cnt in self._global_unigram.items()}
        return {}

    @staticmethod
    def _prev_of(prefix_steps: list[str] | tuple[str, ...]) -> str:
        return prefix_steps[-1] if prefix_steps else BOS_TOKEN

    # ------------------------------------------------------------------ #
    # Public API (mirrors NGramModel so it shares the same eval/setups)  #
    # ------------------------------------------------------------------ #
    def predict_topk(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        k: int = 3,
    ) -> list[str]:
        position = len(prefix_steps)
        scores = self._backoff_scores(family, position, self._prev_of(prefix_steps))
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
        total_logprob = 0.0
        for position, nxt in enumerate(steps):
            prev_step = steps[position - 1] if position > 0 else BOS_TOKEN
            scores = self._backoff_scores(family, position, prev_step)
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
