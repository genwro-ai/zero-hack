from __future__ import annotations

import math
from collections import Counter

import vlmc

from zero_hack.data import FAMILY_FILE_NAMES, SequenceRecord

FAMILY_PREFIX = "<FAMILY:"
UNK_STEP = "<UNK_STEP>"
SCORE_FLOOR = 1e-9


class VLMCModel:
    def __init__(self, max_depth: int = 5) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self.max_depth = max_depth
        self.tree = None
        self.token_to_id: dict[str, int] = {}
        self.id_to_step: dict[int, str] = {}
        self.step_token_ids: set[int] = set()
        self.unigram_counts: Counter[int] = Counter()

    def fit(self, records: list[SequenceRecord]) -> VLMCModel:
        self.token_to_id = {}
        self.id_to_step = {}
        self.step_token_ids = set()
        self.unigram_counts = Counter()

        family_tokens = [_family_token(family) for family in FAMILY_FILE_NAMES]
        for token in [*family_tokens, UNK_STEP]:
            self._token_id(token)

        encoded_records: list[list[int]] = []
        for record in records:
            encoded = [self._token_id(_family_token(record.family))]
            for step in record.steps:
                step_id = self._token_id(step)
                self.id_to_step[step_id] = step
                self.step_token_ids.add(step_id)
                self.unigram_counts[step_id] += 1
                encoded.append(step_id)
            encoded_records.append(encoded)

        if not encoded_records:
            raise ValueError("VLMCModel.fit requires at least one training record")

        self.tree = vlmc.VLMC(len(self.token_to_id), max_depth=self.max_depth)
        self.tree.fit(encoded_records)
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
        prefix: list[str] = []
        for next_step in steps:
            scores = self._scores(family, prefix)
            logprob += math.log(max(scores.get(next_step, 0.0), SCORE_FLOOR))
            prefix.append(next_step)
        return logprob

    def _scores(self, family: str, prefix_steps: list[str] | tuple[str, ...]) -> dict[str, float]:
        distribution = self._vlmc_distribution(family, prefix_steps)
        if distribution is None:
            return self._unigram_scores()

        step_counts = {
            self.id_to_step[token_id]: distribution[token_id]
            for token_id in self.step_token_ids
            if token_id < len(distribution) and distribution[token_id] > 0
        }
        total = sum(step_counts.values())
        if total <= 0:
            return self._unigram_scores()
        return {step: count / total for step, count in step_counts.items()}

    def _vlmc_distribution(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
    ) -> list[int] | None:
        if self.tree is None:
            raise RuntimeError("VLMCModel must be fit before scoring")

        context = self._encode_context(family, prefix_steps)
        try:
            suffix = self.tree.get_suffix(context)
            return list(self.tree.get_distribution(suffix))
        except KeyError:
            return None

    def _encode_context(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
    ) -> list[int]:
        unk_id = self.token_to_id[UNK_STEP]
        family_id = self.token_to_id.get(_family_token(family), unk_id)
        step_ids = [self.token_to_id.get(step, unk_id) for step in prefix_steps]
        return [family_id, *step_ids]

    def _unigram_scores(self) -> dict[str, float]:
        total = sum(self.unigram_counts.values())
        if total <= 0:
            return {}
        return {
            self.id_to_step[token_id]: count / total
            for token_id, count in self.unigram_counts.items()
        }

    def _token_id(self, token: str) -> int:
        token_id = self.token_to_id.get(token)
        if token_id is None:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
        return token_id


def _family_token(family: str) -> str:
    return f"{FAMILY_PREFIX}{family.lower()}>"
