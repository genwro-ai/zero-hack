from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from zero_hack.data import SequenceRecord
from zero_hack.models.topk import TopKAccumulator

BOS_TOKEN = "<BOS>"
SCORE_FLOOR = 1e-9


@dataclass
class FamilyCounts:
    by_position_and_previous: dict[tuple[int, str], Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    by_previous: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    by_position: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    unigram: Counter[str] = field(default_factory=Counter)


class MostFrequentModel:
    def __init__(self, position_bucket_size: int = 5, backoff_alpha: float = 0.4) -> None:
        if position_bucket_size < 1:
            raise ValueError("position_bucket_size must be >= 1")
        self.position_bucket_size = position_bucket_size
        self.backoff_alpha = backoff_alpha
        self.by_family: dict[str, FamilyCounts] = {}
        self.global_unigram: Counter[str] = Counter()

    def fit(self, records: list[SequenceRecord]) -> MostFrequentModel:
        self.by_family = {}
        self.global_unigram = Counter()

        for record in records:
            family_counts = self.by_family.setdefault(record.family, FamilyCounts())
            for position, next_step in enumerate(record.steps):
                previous_step = record.steps[position - 1] if position else BOS_TOKEN
                position_bucket = self._bucket(position)

                family_counts.by_position_and_previous[(position_bucket, previous_step)][
                    next_step
                ] += 1
                family_counts.by_previous[previous_step][next_step] += 1
                family_counts.by_position[position_bucket][next_step] += 1
                family_counts.unigram[next_step] += 1
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
        previous_step = prefix_steps[-1] if prefix_steps else BOS_TOKEN
        position_bucket = self._bucket(len(prefix_steps))
        family_counts = self.by_family.get(family)

        if family_counts is not None:
            counters = [
                family_counts.by_position_and_previous.get((position_bucket, previous_step)),
                family_counts.by_previous.get(previous_step),
                family_counts.by_position.get(position_bucket),
                family_counts.unigram,
            ]
            for backoff_level, counter in enumerate(counters):
                if counter:
                    return _normalised(counter, weight=self.backoff_alpha**backoff_level)

        if self.global_unigram:
            return _normalised(self.global_unigram, weight=self.backoff_alpha**4)
        return {}

    def _bucket(self, position: int) -> int:
        return position // self.position_bucket_size


def _normalised(counter: Counter[str], weight: float) -> dict[str, float]:
    total = sum(counter.values())
    return {step: weight * count / total for step, count in counter.items()}
