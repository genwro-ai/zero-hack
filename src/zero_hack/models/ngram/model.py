from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from zero_hack.data import SequenceRecord

BOS_TOKEN = "<BOS>"
SCORE_FLOOR = 1e-9

Context = tuple[str, ...]
ContextTable = dict[Context, Counter[str]]


class NGramModel:
    def __init__(self, n: int = 5, backoff_alpha: float = 0.4) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.backoff_alpha = backoff_alpha
        self.by_family: dict[str, ContextTable] = {}
        self.global_counts: ContextTable = defaultdict(Counter)

    def fit(self, records: list[SequenceRecord]) -> NGramModel:
        self.by_family = {}
        self.global_counts = defaultdict(Counter)

        for record in records:
            family_counts = self.by_family.setdefault(record.family, defaultdict(Counter))
            for position, next_step in enumerate(record.steps):
                prefix = record.steps[:position]
                for context in self._contexts(prefix):
                    family_counts[context][next_step] += 1
                    self.global_counts[context][next_step] += 1
        return self

    def predict_topk(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        k: int = 3,
    ) -> list[str]:
        scores = self._scores(family, prefix_steps)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [step for step, _ in ranked[:k]]

    def score_sequence(
        self,
        family: str,
        steps: list[str] | tuple[str, ...],
    ) -> float:
        logprob = 0.0
        for position, next_step in enumerate(steps):
            scores = self._scores(family, steps[:position])
            logprob += math.log(max(scores.get(next_step, 0.0), SCORE_FLOOR))
        return logprob

    def _scores(self, family: str, prefix_steps: Iterable[str]) -> dict[str, float]:
        table = self.by_family.get(family)
        if table is None:
            table = self.global_counts

        for backoff_level, context in enumerate(self._contexts(prefix_steps)):
            counter = table.get(context)
            if counter:
                return _normalised(counter, weight=self.backoff_alpha**backoff_level)

        unigram = self.global_counts.get(())
        if unigram:
            return _normalised(unigram, weight=self.backoff_alpha**self.n)
        return {}

    def _contexts(self, prefix_steps: Iterable[str]) -> list[Context]:
        max_context = self.n - 1
        if max_context == 0:
            return [()]

        context = list(prefix_steps)[-max_context:]
        missing = max_context - len(context)
        if missing:
            context = [BOS_TOKEN] * missing + context

        return [tuple(context[start:]) for start in range(len(context) + 1)]


def _normalised(counter: Counter[str], weight: float) -> dict[str, float]:
    total = sum(counter.values())
    return {step: weight * count / total for step, count in counter.items()}
