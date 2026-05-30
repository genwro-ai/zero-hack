from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from zero_hack.data import SequenceRecord
from zero_hack.models.topk import TopKAccumulator

SCORE_FLOOR = 1e-9


@dataclass
class FamilyCounts:
    by_position: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    unigram: Counter[str] = field(default_factory=Counter)


class MostFrequentModel:
    def __init__(self, position_bucket_size: int = 5) -> None:
        if position_bucket_size < 1:
            raise ValueError("position_bucket_size must be >= 1")
        self.position_bucket_size = position_bucket_size
        self.by_family: dict[str, FamilyCounts] = {}
        self.global_by_position: dict[int, Counter[str]] = defaultdict(Counter)
        self.global_unigram: Counter[str] = Counter()

    def fit(self, records: list[SequenceRecord]) -> MostFrequentModel:
        self.by_family = {}
        self.global_by_position = defaultdict(Counter)
        self.global_unigram = Counter()

        for record in records:
            family_counts = self.by_family.setdefault(record.family, FamilyCounts())
            for position, next_step in enumerate(record.steps):
                position_bucket = self._bucket(position)

                family_counts.by_position[position_bucket][next_step] += 1
                family_counts.unigram[next_step] += 1
                self.global_by_position[position_bucket][next_step] += 1
                self.global_unigram[next_step] += 1
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

    def _scores(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
    ) -> dict[str, float]:
        position_bucket = self._bucket(len(prefix_steps))
        family_counts = self.by_family.get(family)

        if family_counts is not None:
            counter = family_counts.by_position.get(position_bucket)
            if counter:
                return _normalised(counter)
            if family_counts.unigram:
                return _normalised(family_counts.unigram)

        counter = self.global_by_position.get(position_bucket)
        if counter:
            return _normalised(counter)
        if self.global_unigram:
            return _normalised(self.global_unigram)
        return {}

    def _bucket(self, position: int) -> int:
        return position // self.position_bucket_size


def _normalised(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    return {step: count / total for step, count in counter.items()}
