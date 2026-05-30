from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from zero_hack.data import SequenceRecord
from zero_hack.eval import io
from zero_hack.eval.anomaly_synth import ValExample, build_validation_anomaly_set
from zero_hack.models.classic_baselines import ClassicBaselineModel, sequence_avg_logprob

_GUARD_MARGIN = 1.0


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float


def anomaly_f1(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> ThresholdResult:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")

    tp = fp = fn = 0
    for score, label in zip(scores, labels, strict=True):
        pred_anomaly = score < threshold
        gold_anomaly = label == 1
        if pred_anomaly and gold_anomaly:
            tp += 1
        elif pred_anomaly and not gold_anomaly:
            fp += 1
        elif not pred_anomaly and gold_anomaly:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return ThresholdResult(threshold=threshold, f1=f1, precision=precision, recall=recall)


def candidate_thresholds(scores: Sequence[float]) -> list[float]:
    unique = sorted(set(scores))
    if not unique:
        return []
    midpoints = [(a + b) / 2.0 for a, b in zip(unique, unique[1:], strict=False)]
    return [unique[0], *midpoints, unique[-1] + _GUARD_MARGIN]


def sweep_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    grid: Sequence[float] | None = None,
) -> ThresholdResult:
    if len(scores) == 0:
        raise ValueError("cannot tune a threshold with no scores")
    candidates = list(grid) if grid is not None else candidate_thresholds(scores)
    results = (anomaly_f1(scores, labels, t) for t in candidates)
    return max(results, key=lambda r: (r.f1, r.recall, r.threshold))


def tune_anomaly_threshold(
    model: ClassicBaselineModel,
    records: list[SequenceRecord],
    *,
    n_valid: int = 200,
    n_invalid: int = 129,
    seed: int = 1729,
    grid: Sequence[float] | None = None,
) -> ThresholdResult:
    examples = build_validation_anomaly_set(
        records, n_valid=n_valid, n_invalid=n_invalid, seed=seed
    )
    if not examples:
        raise ValueError("validation anomaly set is empty; cannot tune threshold")
    scores = [sequence_avg_logprob(model, ex.family, ex.steps) for ex in examples]
    labels = [ex.label for ex in examples]
    return sweep_threshold(scores, labels, grid=grid)


def load_threshold_examples(eval_dir: str | Path) -> list[ValExample]:
    """Load fixed anomaly-threshold examples from an eval-set directory.

    ``ValExample.label`` follows the threshold-tuning convention:
    ``1`` means anomaly/invalid and ``0`` means valid.
    """

    eval_dir = Path(eval_dir)
    inputs = io.read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
    truth = io.read_anomaly_truth(eval_dir / "anomaly_truth.csv")
    examples: list[ValExample] = []
    for row in inputs:
        example_id = row["example_id"]
        if example_id not in truth:
            raise ValueError(f"{eval_dir}: missing anomaly truth for {example_id!r}")
        is_valid = int(truth[example_id]["is_valid"])
        examples.append(
            ValExample(
                family=row["family"],
                steps=list(row["sequence"]),
                label=0 if is_valid else 1,
            )
        )
    return examples


def tune_anomaly_threshold_from_examples(
    model: ClassicBaselineModel,
    examples: Sequence[ValExample],
    *,
    grid: Sequence[float] | None = None,
) -> ThresholdResult:
    if not examples:
        raise ValueError("validation anomaly set is empty; cannot tune threshold")
    scores = [sequence_avg_logprob(model, ex.family, ex.steps) for ex in examples]
    labels = [ex.label for ex in examples]
    return sweep_threshold(scores, labels, grid=grid)


def tune_anomaly_threshold_from_eval_dir(
    model: ClassicBaselineModel,
    eval_dir: str | Path,
    *,
    grid: Sequence[float] | None = None,
) -> ThresholdResult:
    return tune_anomaly_threshold_from_examples(
        model,
        load_threshold_examples(eval_dir),
        grid=grid,
    )
