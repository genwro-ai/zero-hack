import pytest

from zero_hack.models.anomaly_threshold import (
    anomaly_f1,
    candidate_thresholds,
    sweep_threshold,
)


def test_anomaly_f1_perfect_separation():
    scores = [-1.0, -2.0, -3.0, -4.0]
    labels = [0, 0, 1, 1]
    result = anomaly_f1(scores, labels, threshold=-2.5)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_anomaly_f1_known_confusion():
    scores = [-1.0, -2.0, -3.0, -4.0]
    labels = [0, 0, 1, 1]
    result = anomaly_f1(scores, labels, threshold=-1.5)
    assert result.precision == pytest.approx(2 / 3)  # tp=2, fp=1
    assert result.recall == pytest.approx(1.0)  # tp=2, fn=0
    assert result.f1 == pytest.approx(0.8)


def test_anomaly_f1_no_positive_predictions_is_zero():
    scores = [-1.0, -2.0]
    labels = [1, 1]
    result = anomaly_f1(scores, labels, threshold=-3.0)
    assert result.f1 == 0.0
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_candidate_thresholds_cover_every_partition():
    scores = [-1.0, -2.0, -3.0, -4.0]
    grid = candidate_thresholds(scores)
    assert grid == sorted(grid)
    assert grid == pytest.approx([-4.0, -3.5, -2.5, -1.5, 0.0])


def test_candidate_thresholds_single_unique_value():
    grid = candidate_thresholds([-2.0, -2.0, -2.0])
    assert grid == pytest.approx([-2.0, -1.0])


def test_sweep_finds_optimal_threshold_on_separable_data():
    scores = [-1.0, -2.0, -3.0, -4.0]
    labels = [0, 0, 1, 1]
    result = sweep_threshold(scores, labels)
    assert result.f1 == pytest.approx(1.0)
    # the only F1=1.0 boundary is between -3 and -2
    assert result.threshold == pytest.approx(-2.5)


def test_sweep_tiebreak_prefers_higher_recall_then_higher_threshold():
    scores = [-1.0, -2.0, -3.0, -4.0]
    labels = [1, 0, 0, 1]
    result = sweep_threshold(scores, labels)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(1.0)
    assert result.threshold == pytest.approx(0.0)


def test_sweep_respects_explicit_grid():
    scores = [-1.0, -2.0, -3.0, -4.0]
    labels = [0, 0, 1, 1]
    result = sweep_threshold(scores, labels, grid=[-1.5, 0.0])
    assert result.threshold == pytest.approx(-1.5)
    assert result.f1 == pytest.approx(0.8)


def test_sweep_empty_scores_raises():
    with pytest.raises(ValueError):
        sweep_threshold([], [])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        anomaly_f1([-1.0, -2.0], [1], threshold=-1.5)
