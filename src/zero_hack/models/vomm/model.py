from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from zero_hack.data import SequenceRecord
from zero_hack.models.topk import TopKAccumulator

SCORE_FLOOR = 1e-12

Context = tuple[str, ...]
ContextTable = dict[Context, Counter[str]]


class VOMMModel:
    def __init__(self, max_order: int = 8) -> None:
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        self.max_order = max_order
        self.by_family: dict[str, ContextTable] = {}
        self.global_counts: ContextTable = defaultdict(Counter)
        self.alphabet: set[str] = set()

    def fit(self, records: list[SequenceRecord]) -> VOMMModel:
        self.by_family = {}
        self.global_counts = defaultdict(Counter)
        self.alphabet = set()

        for record in records:
            family_counts = self.by_family.setdefault(record.family, defaultdict(Counter))
            steps = record.steps
            for position, next_step in enumerate(steps):
                self.alphabet.add(next_step)
                for order in range(min(self.max_order, position) + 1):
                    context = tuple(steps[position - order : position])
                    family_counts[context][next_step] += 1
                    self.global_counts[context][next_step] += 1
        return self

    def predict_topk(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        k: int = 3,
    ) -> list[str]:
        probs = self._distribution(family, prefix_steps)
        ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
        return [step for step, _ in ranked[:k]]

    def score_sequence(
        self,
        family: str,
        steps: list[str] | tuple[str, ...],
    ) -> float:
        steps = list(steps)
        logprob = 0.0
        for position, next_step in enumerate(steps):
            probs = self._distribution(family, steps[:position])
            logprob += math.log(max(probs.get(next_step, 0.0), SCORE_FLOOR))
        return logprob

    def evaluate(
        self,
        records: list[SequenceRecord],
        vocabulary,
        k: int = 3,
    ) -> dict:
        acc = TopKAccumulator(k=k)
        for record in records:
            for position, gold_step in enumerate(record.steps):
                preds = self.predict_topk(record.family, record.steps[:position], k=k)
                gold_id = vocabulary.token_to_id.get(gold_step, vocabulary.unk_id)
                pred_ids = [vocabulary.token_to_id.get(step, vocabulary.unk_id) for step in preds]
                acc.update(gold_id, pred_ids, group=record.family)
        return acc.summary()

    def _distribution(self, family: str, prefix_steps: Iterable[str]) -> dict[str, float]:
        table = self.by_family.get(family, self.global_counts)
        prefix = list(prefix_steps)
        length = len(prefix)
        probs: dict[str, float] = {}
        excluded: set[str] = set()
        remaining = 1.0

        for order in range(min(self.max_order, length), -1, -1):
            if remaining <= 0.0:
                break
            counter = table.get(tuple(prefix[length - order :]))
            if not counter:
                continue
            seen = {step: count for step, count in counter.items() if step not in excluded}
            if not seen:
                continue
            total = sum(seen.values())
            distinct = len(seen)
            denom = total + distinct
            for step, count in seen.items():
                probs[step] = probs.get(step, 0.0) + remaining * count / denom
            remaining *= distinct / denom
            excluded.update(seen)

        if remaining > 0.0:
            rest = [step for step in self.alphabet if step not in excluded]
            if rest:
                share = remaining / len(rest)
                for step in rest:
                    probs[step] = probs.get(step, 0.0) + share
        return probs
