#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any

from zero_hack import PROJECT_ROOT
from zero_hack.eval import io
from zero_hack.eval.score import TASKS, score_task
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold_from_eval_dir
from zero_hack.models.classic_baselines import (
    CLASSIC_BASELINES,
    build_classic_baseline,
    complete_sequence,
    predict_anomaly,
)
from zero_hack.models.common import FAMILIES, load_split_records

_DATASET_SIZE = re.compile(r"_s(\d+)k$")
_VIEWS = ("id", "ood")


def _dataset_sort_key(name: str) -> tuple[int, str]:
    match = _DATASET_SIZE.search(name)
    if match:
        return int(match.group(1)), name
    return 10**12, name


def _discover_datasets(generated_root: Path) -> list[str]:
    datasets = []
    for path in generated_root.iterdir():
        if path.is_dir() and (path / "splits").exists() and path.name.startswith("valid_"):
            datasets.append(path.name)
    return sorted(datasets, key=_dataset_sort_key)


def _truth_path(eval_dir: Path, task: str) -> Path:
    return (
        eval_dir
        / {
            "next_step": "nextstep_truth.csv",
            "completion": "completion_truth.csv",
            "anomaly": "anomaly_truth.csv",
        }[task]
    )


def _eval_input_path(eval_dir: Path, task: str) -> Path:
    if task == "anomaly":
        return eval_dir / "eval_input_anomaly.csv"
    return eval_dir / "eval_input_valid.csv"


def _pred_path(pred_dir: Path, task: str) -> Path:
    return (
        pred_dir
        / {
            "next_step": "nextstep.csv",
            "completion": "completion.csv",
            "anomaly": "anomaly.csv",
        }[task]
    )


def _write_eval_predictions(
    model: Any,
    *,
    eval_dir: Path,
    pred_dir: Path,
    tasks: tuple[str, ...],
    threshold: float,
    k: int,
    max_completion_steps: int,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    if "next_step" in tasks or "completion" in tasks:
        inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
        if "next_step" in tasks:
            rows = [
                {
                    "example_id": row["example_id"],
                    "ranks": model.predict_topk(row["family"], row["partial_sequence"], k=k),
                }
                for row in inputs
            ]
            io.write_next_step_predictions(pred_dir / "nextstep.csv", rows)

        if "completion" in tasks:
            rows = [
                {
                    "example_id": row["example_id"],
                    "steps": complete_sequence(
                        model,
                        row["family"],
                        list(row["partial_sequence"]),
                        max_steps=max_completion_steps,
                    ),
                }
                for row in inputs
            ]
            io.write_completion_predictions(pred_dir / "completion.csv", rows)

    if "anomaly" in tasks:
        rows = [
            {
                "example_id": row["example_id"],
                **predict_anomaly(
                    model, row["family"], list(row["sequence"]), "likelihood", threshold
                ),
            }
            for row in io.read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
        ]
        io.write_anomaly_predictions(pred_dir / "anomaly.csv", rows)


def _score_prediction_files(
    *,
    eval_dir: Path,
    pred_dir: Path,
    metrics_dir: Path,
    tasks: tuple[str, ...],
) -> dict[str, dict]:
    results = {}
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        metrics = score_task(
            task,
            ground_truth=_truth_path(eval_dir, task),
            predictions=_pred_path(pred_dir, task),
            eval_input=_eval_input_path(eval_dir, task),
        )
        results[task] = metrics
        (metrics_dir / f"{task}.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classic baselines and score fixed holdout eval CSVs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--generated-root", default=str(PROJECT_ROOT / "data" / "generated"))
    parser.add_argument("--eval-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--preds-root", default=str(PROJECT_ROOT / "outputs" / "preds"))
    parser.add_argument("--metrics-root", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--holdout-families",
        nargs="+",
        choices=FAMILIES,
        default=list(FAMILIES),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=CLASSIC_BASELINES,
        default=["ngram"],
    )
    parser.add_argument("--views", nargs="+", choices=_VIEWS, default=list(_VIEWS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--max-completion-steps", type=int, default=400)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def _resolve_anomaly_threshold(
    model: Any,
    bundle: Any,
    *,
    calibration_dir: Path,
    tasks: tuple[str, ...],
) -> tuple[float, dict | None]:
    if "anomaly" not in tasks:
        return -1.0, None

    if not calibration_dir.exists():
        raise SystemExit(
            f"Missing threshold calibration set: {calibration_dir}. "
            "Run scripts/make_all_eval_sets.py first."
        )
    result = tune_anomaly_threshold_from_eval_dir(model, calibration_dir)
    record = {
        "source": "threshold_calibration",
        "objective": "f1",
        "tuned_on": str(calibration_dir),
        "train_families": list(bundle.train_families),
        "threshold": result.threshold,
        "val_f1": result.f1,
        "val_precision": result.precision,
        "val_recall": result.recall,
    }
    return result.threshold, record


def _write_results(
    metrics_dir: Path,
    results: dict[str, dict],
    tuning: dict | None,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for task, metrics in results.items():
        (metrics_dir / f"{task}.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
    (metrics_dir / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    if tuning is not None:
        (metrics_dir / "anomaly_threshold.json").write_text(
            json.dumps(tuning, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    args = _parse_args()
    generated_root = Path(args.generated_root)
    eval_root = Path(args.eval_root)
    preds_root = Path(args.preds_root)
    metrics_root = Path(args.metrics_root)
    datasets = args.datasets or _discover_datasets(generated_root)
    tasks = tuple(args.tasks)

    if not datasets:
        raise SystemExit(f"No datasets found under {generated_root}")

    for dataset in datasets:
        splits_dir = generated_root / dataset / "splits"
        if not splits_dir.exists():
            raise SystemExit(f"Missing splits directory: {splits_dir}")

        for holdout_family in args.holdout_families:
            bundle = load_split_records(
                splits_dir,
                holdout_family=holdout_family,
                limit_per_family=args.limit_per_family,
            )
            print(
                f"\n== dataset={dataset} holdout={holdout_family} "
                f"train_families={bundle.train_families} counts={bundle.counts()}"
            )

            for model_name in args.models:
                print(f"--> fitting {model_name}")
                model = build_classic_baseline(
                    model_name,
                    bundle.records["train"],
                    n=args.n,
                    alpha=args.alpha,
                    seed=args.seed,
                )

                threshold, tuning = _resolve_anomaly_threshold(
                    model,
                    bundle,
                    calibration_dir=eval_root
                    / dataset
                    / f"holdout_{holdout_family}"
                    / "calibration",
                    tasks=tasks,
                )
                if tuning is not None:
                    print(
                        f"--> tuned anomaly threshold for {model_name}: {threshold:.4f} "
                        f"(val F1={tuning['val_f1']:.4f} "
                        f"P={tuning['val_precision']:.4f} R={tuning['val_recall']:.4f})"
                    )

                for view in args.views:
                    eval_dir = eval_root / dataset / f"holdout_{holdout_family}" / view
                    if not eval_dir.exists():
                        raise SystemExit(
                            f"Missing fixed eval set: {eval_dir}. "
                            "Run scripts/make_all_eval_sets.py first."
                        )
                    pred_dir = (
                        preds_root / dataset / f"holdout_{holdout_family}" / view / model_name
                    )
                    metrics_dir = (
                        metrics_root / dataset / f"holdout_{holdout_family}" / view / model_name
                    )
                    print(f"--> evaluating {model_name} on fixed {view} set: {eval_dir}")
                    _write_eval_predictions(
                        model,
                        eval_dir=eval_dir,
                        pred_dir=pred_dir,
                        tasks=tasks,
                        threshold=threshold,
                        k=5,
                        max_completion_steps=args.max_completion_steps,
                    )
                    results = _score_prediction_files(
                        eval_dir=eval_dir,
                        pred_dir=pred_dir,
                        metrics_dir=metrics_dir,
                        tasks=tasks,
                    )
                    _write_results(metrics_dir, results, tuning)
                    compact = {
                        task: metrics.get("all", metrics) for task, metrics in results.items()
                    }
                    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
