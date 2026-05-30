import math

import pytest

from zero_hack import PROJECT_ROOT
from zero_hack.data import load_sequence_records
from zero_hack.eval.anomaly_synth import build_validation_anomaly_set
from zero_hack.eval.validator import is_valid
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold
from zero_hack.models.classic_baselines import build_classic_baseline, predict_anomaly

SPLITS = PROJECT_ROOT / "data" / "generated" / "valid_s005k" / "splits"


def _records(name: str, n: int):
    return load_sequence_records(SPLITS / name)[:n]


IC = _records("IC_valid.csv", 40)
MOSFET = _records("MOSFET_valid.csv", 40)
RECORDS = IC + MOSFET


def test_build_per_family_counts_and_labels():
    examples = build_validation_anomaly_set(RECORDS, n_valid=5, n_invalid=3, seed=1729)
    for family in ("ic", "mosfet"):
        fam = [e for e in examples if e.family == family]
        assert sum(e.label == 0 for e in fam) == 5
        assert sum(e.label == 1 for e in fam) == 3


def test_build_invalid_examples_are_actually_invalid():
    examples = build_validation_anomaly_set(RECORDS, n_valid=5, n_invalid=5, seed=1729)
    for e in examples:
        if e.label == 1:
            assert not is_valid(e.steps)


def test_build_valid_examples_keep_an_original_sequence():
    examples = build_validation_anomaly_set(IC, n_valid=5, n_invalid=0, seed=1729)
    originals = {tuple(r.steps) for r in IC}
    assert examples
    for e in examples:
        assert e.label == 0
        assert tuple(e.steps) in originals


def test_build_is_deterministic_for_a_fixed_seed():
    a = build_validation_anomaly_set(RECORDS, n_valid=5, n_invalid=5, seed=7)
    b = build_validation_anomaly_set(RECORDS, n_valid=5, n_invalid=5, seed=7)
    assert a == b


def test_tune_returns_a_finite_threshold_and_valid_f1():
    model = build_classic_baseline("ngram", _records("IC_train.csv", 200))
    result = tune_anomaly_threshold(model, IC, n_valid=20, n_invalid=13, seed=1729)
    assert math.isfinite(result.threshold)
    assert 0.0 <= result.f1 <= 1.0


def test_tuned_threshold_roundtrips_through_predict_anomaly():
    model = build_classic_baseline("ngram", _records("IC_train.csv", 300))
    result = tune_anomaly_threshold(model, IC, n_valid=20, n_invalid=13, seed=1729)
    val = build_validation_anomaly_set(IC, n_valid=20, n_invalid=13, seed=1729)

    tp = fp = fn = 0
    for e in val:
        pred = predict_anomaly(model, e.family, list(e.steps), "likelihood", result.threshold)
        pred_anom = pred["is_valid"] == 0
        gold_anom = e.label == 1
        tp += pred_anom and gold_anom
        fp += pred_anom and not gold_anom
        fn += (not pred_anom) and gold_anom
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    assert f1 == pytest.approx(result.f1)
