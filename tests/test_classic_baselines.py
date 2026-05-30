import math

import pytest

from zero_hack.data import SequenceRecord
from zero_hack.models.classic_baselines import (
    CLASSIC_BASELINES,
    build_classic_baseline,
    predict_anomaly,
    sequence_avg_logprob,
)
from zero_hack.models.hmm import HMMModel


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


def test_hmm_is_registered_as_classic_baseline():
    assert "hmm" in CLASSIC_BASELINES
    model = build_classic_baseline(
        "hmm",
        _hmm_records(),
        n=3,
        hmm_iterations=3,
        hmm_smoothing=1e-3,
    )
    assert isinstance(model, HMMModel)


def test_hmm_predicts_after_simple_prefix():
    model = HMMModel(hidden_states=3, iterations=5, smoothing=1e-3).fit(_hmm_records())
    assert model.predict_topk("ic", [], k=1) == ["START"]
    assert model.predict_topk("ic", ["START", "ETCH"], k=1) == ["SHIP LOT"]


def test_hmm_scores_seen_order_above_reversed_order():
    model = HMMModel(hidden_states=3, iterations=5, smoothing=1e-3).fit(_hmm_records())
    seen = model.score_sequence("ic", ["START", "ETCH", "SHIP LOT"])
    reversed_order = model.score_sequence("ic", ["SHIP LOT", "ETCH", "START"])
    assert seen > reversed_order


def test_hmm_unknown_family_uses_global_fallback():
    model = HMMModel(hidden_states=3, iterations=3, smoothing=1e-3).fit(_hmm_records())
    assert model.predict_topk("mosfet", [], k=1) == ["START"]


def _hmm_records() -> list[SequenceRecord]:
    return [
        SequenceRecord("ic", f"ic_{idx}", ("START", "ETCH", "SHIP LOT")) for idx in range(20)
    ] + [
        SequenceRecord("igbt", f"igbt_{idx}", ("START", "DEPOSIT", "SHIP LOT")) for idx in range(20)
    ]
