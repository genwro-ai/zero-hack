from pathlib import Path

from zero_hack.eval import io
from zero_hack.eval.anomaly import score_anomaly
from zero_hack.eval.completion import score_completion
from zero_hack.eval.next_step import score_next_step

TASKS = ("next_step", "completion", "anomaly")


def _families_from_input(eval_input: str | Path | None) -> dict[str, str] | None:
    if eval_input is None:
        return None
    return {row["example_id"]: row["family"] for row in io.read_eval_input_valid(eval_input)}


def _families_from_anomaly_input(eval_input: str | Path | None) -> dict[str, str] | None:
    if eval_input is None:
        return None
    return {row["example_id"]: row["family"] for row in io.read_eval_input_anomaly(eval_input)}


def score_next_step_files(
    ground_truth: str | Path,
    predictions: str | Path,
    eval_input: str | Path | None = None,
) -> dict:
    return score_next_step(
        io.read_next_step_truth(ground_truth),
        io.read_next_step_predictions(predictions),
        families=_families_from_input(eval_input),
    )


def score_completion_files(
    ground_truth: str | Path,
    predictions: str | Path,
    eval_input: str | Path | None = None,
) -> dict:
    return score_completion(
        io.read_completion_truth(ground_truth),
        io.read_completion_predictions(predictions),
        families=_families_from_input(eval_input),
    )


def score_anomaly_files(
    ground_truth: str | Path,
    predictions: str | Path,
    eval_input: str | Path | None = None,
) -> dict:
    return score_anomaly(
        io.read_anomaly_truth(ground_truth),
        io.read_anomaly_predictions(predictions),
        families=_families_from_anomaly_input(eval_input),
    )


def score_task(
    task: str,
    ground_truth: str | Path,
    predictions: str | Path,
    eval_input: str | Path | None = None,
) -> dict:
    """Dispatch to the scorer for ``task`` (one of :data:`TASKS`)."""
    dispatch = {
        "next_step": score_next_step_files,
        "completion": score_completion_files,
        "anomaly": score_anomaly_files,
    }
    if task not in dispatch:
        raise ValueError(f"Unknown task {task!r}. Expected one of: {', '.join(TASKS)}")
    return dispatch[task](ground_truth, predictions, eval_input)
