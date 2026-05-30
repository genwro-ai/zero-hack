import math
from typing import Any, Protocol

from zero_hack.data import SequenceRecord
from zero_hack.eval.validator import first_violated_rule, validate_sequence
from zero_hack.models.hmm import HMMModel
from zero_hack.models.most_frequent import MostFrequentModel
from zero_hack.models.ngram import NGramModel
from zero_hack.models.vomm import VOMMModel

MAX_COMPLETION_STEPS = 400
SEQUENCE_TERMINATOR = "SHIP LOT"
CLASSIC_BASELINES = ("most_frequent", "ngram", "vomm", "hmm")


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
    hmm_states: int | None = None,
    hmm_iterations: int = 8,
    hmm_smoothing: float = 1e-2,
    seed: int = 1729,
) -> ClassicBaselineModel:
    if name == "ngram":
        return NGramModel(n=n, backoff_alpha=alpha).fit(train_records)
    if name == "most_frequent":
        return MostFrequentModel(position_bucket_size=bucket).fit(train_records)
    if name == "vomm":
        return VOMMModel(max_order=n).fit(train_records)
    if name == "hmm":
        return HMMModel(
            hidden_states=hmm_states or n,
            iterations=hmm_iterations,
            smoothing=hmm_smoothing,
            seed=seed,
        ).fit(train_records)
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


def sequence_avg_logprob(
    model: ClassicBaselineModel,
    family: str,
    steps: list[str] | tuple[str, ...],
) -> float:
    return model.score_sequence(family, steps) / max(1, len(steps))


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

    if method not in ("likelihood", "hybrid"):
        raise ValueError("anomaly method must be one of: validator, likelihood, hybrid")

    avg_logprob = sequence_avg_logprob(model, family, sequence)
    likelihood = 1.0 / (1.0 + math.exp(-(avg_logprob - threshold)))

    if method == "likelihood":
        valid = avg_logprob >= threshold
        rule = None if valid else (first_violated_rule(sequence) or "RULE_DEP_NO_CLEAN")
        return {"is_valid": int(valid), "score": round(likelihood, 6), "predicted_rule": rule}

    rule = first_violated_rule(sequence)
    if rule is not None:
        return {"is_valid": 0, "score": round(0.5 * likelihood, 6), "predicted_rule": rule}
    valid = avg_logprob >= threshold
    score = 0.5 + 0.5 * likelihood
    fallback = None if valid else "RULE_DEP_NO_CLEAN"
    return {"is_valid": int(valid), "score": round(score, 6), "predicted_rule": fallback}
