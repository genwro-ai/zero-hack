from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from zero_hack.data import SequenceRecord
from zero_hack.models.topk import TopKAccumulator

SCORE_FLOOR = 1e-12


@dataclass
class _DiscreteHMM:
    start: np.ndarray
    transition: np.ndarray
    emission: np.ndarray


class HMMModel:
    """Family-conditioned categorical HMM trained with Baum-Welch.

    Each family gets an HMM over the shared step vocabulary, with a global HMM as
    fallback for unknown/held-out family labels. The hidden state count is kept
    small by default so it behaves like the other lightweight classic baselines.
    """

    def __init__(
        self,
        hidden_states: int = 5,
        iterations: int = 8,
        smoothing: float = 1e-2,
        seed: int = 1729,
    ) -> None:
        if hidden_states < 1:
            raise ValueError("hidden_states must be >= 1")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if smoothing <= 0.0:
            raise ValueError("smoothing must be > 0")
        self.hidden_states = hidden_states
        self.iterations = iterations
        self.smoothing = smoothing
        self.seed = seed
        self.vocabulary: tuple[str, ...] = ()
        self.token_to_id: dict[str, int] = {}
        self.by_family: dict[str, _DiscreteHMM] = {}
        self.global_model: _DiscreteHMM | None = None

    def fit(self, records: list[SequenceRecord]) -> HMMModel:
        self.vocabulary = tuple(sorted({step for record in records for step in record.steps}))
        self.token_to_id = {step: idx for idx, step in enumerate(self.vocabulary)}
        self.by_family = {}
        self.global_model = None

        if not self.vocabulary:
            return self

        grouped: dict[str, list[SequenceRecord]] = defaultdict(list)
        for record in records:
            grouped[record.family].append(record)

        rng = np.random.default_rng(self.seed)
        all_observations = self._records_to_observations(records)
        self.global_model = _fit_discrete_hmm(
            all_observations,
            hidden_states=self.hidden_states,
            vocab_size=len(self.vocabulary),
            iterations=self.iterations,
            smoothing=self.smoothing,
            rng=rng,
        )
        for family, family_records in grouped.items():
            observations = self._records_to_observations(family_records)
            self.by_family[family] = _fit_discrete_hmm(
                observations,
                hidden_states=self.hidden_states,
                vocab_size=len(self.vocabulary),
                iterations=self.iterations,
                smoothing=self.smoothing,
                rng=rng,
            )
        return self

    def predict_topk(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        k: int = 3,
    ) -> list[str]:
        model = self._model_for_family(family)
        if model is None:
            return []

        token_probs = self._next_token_distribution(model, prefix_steps)
        ranked_ids = np.argsort(-token_probs, kind="stable")[:k]
        ranked = sorted(
            ((self.vocabulary[int(idx)], float(token_probs[int(idx)])) for idx in ranked_ids),
            key=lambda item: (-item[1], item[0]),
        )
        return [step for step, _ in ranked]

    def score_sequence(
        self,
        family: str,
        steps: list[str] | tuple[str, ...],
    ) -> float:
        model = self._model_for_family(family)
        if model is None:
            return 0.0 if not steps else len(steps) * math.log(SCORE_FLOOR)

        alpha = model.start.copy()
        logprob = 0.0
        for position, step in enumerate(steps):
            if position:
                alpha = alpha @ model.transition
            alpha *= self._emission_vector(model, step)
            scale = float(alpha.sum())
            if scale <= 0.0:
                logprob += math.log(SCORE_FLOOR)
                alpha = np.full(self.hidden_states, 1.0 / self.hidden_states)
                continue
            logprob += math.log(max(scale, SCORE_FLOOR))
            alpha /= scale
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

    def _records_to_observations(self, records: list[SequenceRecord]) -> list[np.ndarray]:
        return [
            np.array([self.token_to_id[step] for step in record.steps], dtype=np.int64)
            for record in records
            if record.steps
        ]

    def _model_for_family(self, family: str) -> _DiscreteHMM | None:
        return self.by_family.get(family) or self.global_model

    def _next_token_distribution(
        self,
        model: _DiscreteHMM,
        prefix_steps: list[str] | tuple[str, ...],
    ) -> np.ndarray:
        if prefix_steps:
            alpha = self._filtered_state_distribution(model, prefix_steps)
            next_state = alpha @ model.transition
        else:
            next_state = model.start
        probs = next_state @ model.emission
        total = float(probs.sum())
        if total <= 0.0:
            return np.full(len(self.vocabulary), 1.0 / len(self.vocabulary))
        return probs / total

    def _filtered_state_distribution(
        self,
        model: _DiscreteHMM,
        steps: list[str] | tuple[str, ...],
    ) -> np.ndarray:
        alpha = model.start.copy()
        for position, step in enumerate(steps):
            if position:
                alpha = alpha @ model.transition
            alpha *= self._emission_vector(model, step)
            scale = float(alpha.sum())
            if scale <= 0.0:
                alpha = np.full(self.hidden_states, 1.0 / self.hidden_states)
            else:
                alpha /= scale
        return alpha

    def _emission_vector(self, model: _DiscreteHMM, step: str) -> np.ndarray:
        token_id = self.token_to_id.get(step)
        if token_id is None:
            return np.full(self.hidden_states, SCORE_FLOOR)
        return model.emission[:, token_id]


def _fit_discrete_hmm(
    observations: list[np.ndarray],
    *,
    hidden_states: int,
    vocab_size: int,
    iterations: int,
    smoothing: float,
    rng: np.random.Generator,
) -> _DiscreteHMM:
    if not observations:
        return _uniform_hmm(hidden_states, vocab_size)

    start, transition, emission = _initial_parameters(
        observations,
        hidden_states=hidden_states,
        vocab_size=vocab_size,
        smoothing=smoothing,
        rng=rng,
    )

    for _ in range(iterations):
        start_counts = np.full(hidden_states, smoothing)
        transition_counts = np.full((hidden_states, hidden_states), smoothing)
        emission_counts = np.full((hidden_states, vocab_size), smoothing)

        for sequence in observations:
            alpha, beta = _forward_backward(sequence, start, transition, emission)
            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), SCORE_FLOOR)

            start_counts += gamma[0]
            for token_id, weights in zip(sequence, gamma, strict=True):
                emission_counts[:, int(token_id)] += weights

            for position in range(len(sequence) - 1):
                next_emission = emission[:, int(sequence[position + 1])]
                xi = (
                    alpha[position, :, None]
                    * transition
                    * next_emission[None, :]
                    * beta[position + 1, None, :]
                )
                denom = float(xi.sum())
                if denom > 0.0:
                    transition_counts += xi / denom

        start = _normalise(start_counts)
        transition = _normalise_rows(transition_counts)
        emission = _normalise_rows(emission_counts)

    return _DiscreteHMM(start=start, transition=transition, emission=emission)


def _initial_parameters(
    observations: list[np.ndarray],
    *,
    hidden_states: int,
    vocab_size: int,
    smoothing: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_counts = np.full(hidden_states, smoothing)
    transition_counts = np.full((hidden_states, hidden_states), smoothing)
    emission_counts = np.full((hidden_states, vocab_size), smoothing)

    for sequence in observations:
        last_state = 0
        for position, token_id in enumerate(sequence):
            state = min(hidden_states - 1, int(position * hidden_states / max(1, len(sequence))))
            if position == 0:
                start_counts[state] += 1.0
            else:
                transition_counts[last_state, state] += 1.0
            emission_counts[state, int(token_id)] += 1.0
            last_state = state

    jitter = 1.0 + rng.uniform(-1e-3, 1e-3, size=emission_counts.shape)
    emission_counts *= jitter

    return (
        _normalise(start_counts),
        _normalise_rows(transition_counts),
        _normalise_rows(emission_counts),
    )


def _forward_backward(
    sequence: np.ndarray,
    start: np.ndarray,
    transition: np.ndarray,
    emission: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(sequence)
    hidden_states = len(start)
    alpha = np.zeros((length, hidden_states))
    beta = np.zeros((length, hidden_states))
    scales = np.zeros(length)

    alpha[0] = start * emission[:, int(sequence[0])]
    scales[0] = max(float(alpha[0].sum()), SCORE_FLOOR)
    alpha[0] /= scales[0]

    for position in range(1, length):
        alpha[position] = (alpha[position - 1] @ transition) * emission[:, int(sequence[position])]
        scales[position] = max(float(alpha[position].sum()), SCORE_FLOOR)
        alpha[position] /= scales[position]

    beta[-1] = 1.0
    for position in range(length - 2, -1, -1):
        beta[position] = (
            transition @ (emission[:, int(sequence[position + 1])] * beta[position + 1])
        ) / scales[position + 1]

    return alpha, beta


def _uniform_hmm(hidden_states: int, vocab_size: int) -> _DiscreteHMM:
    return _DiscreteHMM(
        start=np.full(hidden_states, 1.0 / hidden_states),
        transition=np.full((hidden_states, hidden_states), 1.0 / hidden_states),
        emission=np.full((hidden_states, vocab_size), 1.0 / vocab_size),
    )


def _normalise(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0.0:
        return np.full_like(values, 1.0 / len(values), dtype=float)
    return values / total


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    totals = values.sum(axis=1, keepdims=True)
    return values / np.maximum(totals, SCORE_FLOOR)
