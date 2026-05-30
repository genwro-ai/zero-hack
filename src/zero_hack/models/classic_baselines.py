from __future__ import annotations

import math
from typing import Any, Protocol

from zero_hack.data import SequenceRecord
from zero_hack.eval.validator import first_violated_rule, validate_sequence
from zero_hack.models.most_frequent import MostFrequentModel
from zero_hack.models.ngram import NGramModel

MAX_COMPLETION_STEPS = 400
SEQUENCE_TERMINATOR = "SHIP LOT"
CLASSIC_BASELINES = ("most_frequent", "ngram")


class ClassicBaselineModel(Protocol):
    def predict_topk(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        k: int = 3,
    ) -> list[str]: ...

    def score_sequence(
        self,
        family: str,
        steps: list[str] | tuple[str, ...],
    ) -> float: ...


def build_classic_baseline(
    name: str,
    train_records: list[SequenceRecord],
    *,
    n: int = 5,
    alpha: float = 0.4,
    bucket: int = 5,
) -> ClassicBaselineModel:
    if name == "ngram":
        return NGramModel(n=n, backoff_alpha=alpha).fit(train_records)
    if name == "most_frequent":
        return MostFrequentModel(position_bucket_size=bucket).fit(train_records)
    allowed = ", ".join(CLASSIC_BASELINES)
    raise ValueError(f"Unknown classic baseline {name!r}. Expected one of: {allowed}")


def complete_sequence(
    model: ClassicBaselineModel,
    family: str,
    prefix: list[str],
    *,
    max_steps: int = MAX_COMPLETION_STEPS,
) -> list[str]:
    seq = list(prefix)
    produced: list[str] = []
    while len(seq) < max_steps:
        topk = model.predict_topk(family, seq, k=1)
        if not topk:
            break
        next_step = topk[0]
        seq.append(next_step)
        produced.append(next_step)
        if next_step == SEQUENCE_TERMINATOR:
            break
    return produced


def predict_anomaly(
    model: ClassicBaselineModel,
    family: str,
    sequence: list[str],
    method: str,
    threshold: float,
) -> dict[str, Any]:
    if method == "validator":
        violations = validate_sequence(sequence)
        valid = not violations
        return {
            "is_valid": int(valid),
            "score": 1.0 if valid else 0.0,
            "predicted_rule": None if valid else first_violated_rule(sequence),
        }

    if method != "likelihood":
        raise ValueError("anomaly method must be one of: validator, likelihood")

    avg_logprob = model.score_sequence(family, sequence) / max(1, len(sequence))
    score = 1.0 / (1.0 + math.exp(-(avg_logprob - threshold)))
    valid = avg_logprob >= threshold
    return {
        "is_valid": int(valid),
        "score": round(score, 6),
        "predicted_rule": None if valid else (first_violated_rule(sequence) or "RULE_DEP_NO_CLEAN"),
    }
