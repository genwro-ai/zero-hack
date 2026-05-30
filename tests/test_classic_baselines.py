import math

import pytest

from zero_hack.models.classic_baselines import (
    predict_anomaly,
    sequence_avg_logprob,
)


class FakeModel:
    def __init__(self, total_logprob: float) -> None:
        self._total = total_logprob

    def score_sequence(self, family: str, steps) -> float:
        return self._total

    def predict_topk(self, family: str, prefix, k: int = 3):
        return []


def test_sequence_avg_logprob_divides_total_by_length():
    model = FakeModel(-10.0)
    assert sequence_avg_logprob(model, "ic", ["a", "b", "c", "d", "e"]) == pytest.approx(-2.0)


def test_sequence_avg_logprob_empty_sequence_does_not_divide_by_zero():
    model = FakeModel(-3.0)
    assert sequence_avg_logprob(model, "ic", []) == pytest.approx(-3.0)


def test_predict_anomaly_valid_when_avg_at_or_above_threshold():
    model = FakeModel(-10.0)
    out = predict_anomaly(model, "ic", ["a"] * 5, "likelihood", threshold=-3.0)
    assert out["is_valid"] == 1
    assert out["predicted_rule"] is None
    assert out["score"] == pytest.approx(1.0 / (1.0 + math.exp(-(-2.0 - -3.0))), abs=1e-6)


def test_predict_anomaly_flags_anomaly_below_threshold():
    model = FakeModel(-10.0)
    out = predict_anomaly(model, "ic", ["a"] * 5, "likelihood", threshold=-1.0)
    assert out["is_valid"] == 0
    assert out["predicted_rule"] is not None
    assert out["score"] == pytest.approx(1.0 / (1.0 + math.exp(-(-2.0 - -1.0))), abs=1e-6)


def test_predict_anomaly_decision_matches_avg_logprob_threshold():
    model = FakeModel(-10.0)
    avg = sequence_avg_logprob(model, "ic", ["a"] * 5)
    for threshold in (-5.0, -2.0, -1.999, -1.0, 0.0):
        out = predict_anomaly(model, "ic", ["a"] * 5, "likelihood", threshold)
        assert out["is_valid"] == int(avg >= threshold)
