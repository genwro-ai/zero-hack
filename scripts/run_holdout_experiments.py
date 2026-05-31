#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from zero_hack import PROJECT_ROOT
from zero_hack.data import SequenceRecord
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
_VARIANT_EVAL_SOURCE = "industrial_variants_with_generated_rule_anomalies"


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


def _assert_variant_eval_set(eval_dir: Path, tasks: tuple[str, ...]) -> None:
    if not {"next_step", "completion"}.intersection(tasks):
        return

    metadata_path = eval_dir / "metadata.json"
    if not metadata_path.exists():
        raise SystemExit(
            f"Missing eval metadata for Task 1/2: {metadata_path}. "
            "Regenerate eval sets with scripts/make_all_eval_sets.py."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = metadata.get("source")
    if source != _VARIANT_EVAL_SOURCE:
        raise SystemExit(
            f"Task 1/2 eval set must come from Industrial *_variants.csv files; "
            f"{metadata_path} has source={source!r}."
        )


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
        default=["ngram", "vlmc"],
    )
    parser.add_argument("--views", nargs="+", choices=_VIEWS, default=list(_VIEWS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Total train records to keep after holdout filtering, stratified by train family.",
    )
    parser.add_argument("--max-completion-steps", type=int, default=400)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument(
        "--n-values",
        nargs="+",
        type=int,
        default=None,
        help="Grid values for n-gram order / VLMC max depth. Overrides --n.",
    )
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument(
        "--alpha-values",
        nargs="+",
        type=float,
        default=None,
        help="Grid values for n-gram backoff alpha. Overrides --alpha for ngram.",
    )
    parser.add_argument(
        "--rank-by",
        default="next_step_mrr",
        help="Metric column used to sort classic_search_summary.{csv,json}.",
    )
    parser.add_argument(
        "--rank-ascending",
        action="store_true",
        help="Sort the search summary in ascending order.",
    )
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
    config: dict[str, Any],
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
    (metrics_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


def _model_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    n_values = args.n_values or [args.n]
    alpha_values = args.alpha_values or [args.alpha]
    grid_mode = args.n_values is not None or args.alpha_values is not None

    configs = []
    for model_name in args.models:
        if model_name == "ngram":
            for n in n_values:
                for alpha in alpha_values:
                    label = f"ngram_n{n}_a{_format_float(alpha)}" if grid_mode else "ngram"
                    configs.append(
                        {
                            "model": model_name,
                            "label": label,
                            "n": n,
                            "alpha": alpha,
                        }
                    )
        elif model_name == "vlmc":
            for n in n_values:
                label = f"vlmc_d{n}" if grid_mode else "vlmc"
                configs.append(
                    {
                        "model": model_name,
                        "label": label,
                        "n": n,
                        "alpha": args.alpha,
                    }
                )
        else:
            raise ValueError(f"Unhandled model config: {model_name}")
    return configs


def _format_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def _limit_train_records(
    records: list[SequenceRecord],
    *,
    total: int | None,
    seed: int,
) -> list[SequenceRecord]:
    if total is None:
        return records
    if total <= 0:
        raise ValueError("--train-samples must be positive")
    if total >= len(records):
        return records

    by_family: dict[str, list[SequenceRecord]] = {}
    for record in records:
        by_family.setdefault(record.family, []).append(record)

    families = sorted(by_family)
    shuffled = {}
    for index, family in enumerate(families):
        family_records = list(by_family[family])
        random.Random(seed + index).shuffle(family_records)
        shuffled[family] = family_records

    targets = {family: total // len(families) for family in families}
    for family in families[: total % len(families)]:
        targets[family] += 1

    selected_counts = {family: min(targets[family], len(shuffled[family])) for family in families}
    remaining = total - sum(selected_counts.values())
    while remaining > 0:
        progressed = False
        for family in families:
            if selected_counts[family] < len(shuffled[family]):
                selected_counts[family] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    limited = [
        record for family in families for record in shuffled[family][: selected_counts[family]]
    ]
    random.Random(seed).shuffle(limited)
    return limited


def _compact_metric_row(
    *,
    dataset: str,
    holdout_family: str,
    view: str,
    config: dict[str, Any],
    train_samples: int,
    results: dict[str, dict],
    tuning: dict | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": dataset,
        "holdout_family": holdout_family,
        "view": view,
        "method": config["label"],
        "model": config["model"],
        "n": config["n"],
        "alpha": config["alpha"],
        "train_samples": train_samples,
    }
    if tuning is not None:
        row["threshold"] = tuning["threshold"]
        row["threshold_val_f1"] = tuning["val_f1"]

    for task, metrics in results.items():
        values = metrics.get("all", metrics)
        for key, value in values.items():
            if isinstance(value, str | int | float | bool) or value is None:
                row[f"{task}_{key}"] = value
    return row


def _write_search_summary(
    path_base: Path,
    rows: list[dict[str, Any]],
    *,
    rank_by: str,
    rank_ascending: bool,
) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = path_base.with_suffix(".json")
    csv_path = path_base.with_suffix(".csv")
    ranked_rows = _rank_rows(rows, rank_by=rank_by, ascending=rank_ascending)
    json_path.write_text(json.dumps(ranked_rows, indent=2) + "\n", encoding="utf-8")

    columns = sorted({key for row in ranked_rows for key in row})
    preferred = [
        "dataset",
        "holdout_family",
        "view",
        "method",
        "model",
        "n",
        "alpha",
        "train_samples",
    ]
    columns = [key for key in preferred if key in columns] + [
        key for key in columns if key not in preferred
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ranked_rows)


def _rank_rows(
    rows: list[dict[str, Any]],
    *,
    rank_by: str,
    ascending: bool,
) -> list[dict[str, Any]]:
    if not rows or all(rank_by not in row for row in rows):
        return rows

    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        value = row.get(rank_by)
        if isinstance(value, int | float):
            return (0, float(value))
        return (1, 0.0)

    return sorted(rows, key=sort_key, reverse=not ascending)


def main() -> None:
    args = _parse_args()
    generated_root = Path(args.generated_root)
    eval_root = Path(args.eval_root)
    preds_root = Path(args.preds_root)
    metrics_root = Path(args.metrics_root)
    datasets = args.datasets or _discover_datasets(generated_root)
    tasks = tuple(args.tasks)
    configs = _model_configs(args)

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
            train_records = _limit_train_records(
                bundle.records["train"],
                total=args.train_samples,
                seed=args.seed,
            )
            print(
                f"\n== dataset={dataset} holdout={holdout_family} "
                f"train_families={bundle.train_families} counts={bundle.counts()} "
                f"train_used={len(train_records)}"
            )
            summary_rows: list[dict[str, Any]] = []

            for config in configs:
                model_name = config["model"]
                method_label = config["label"]
                print(
                    f"--> fitting {method_label} "
                    f"(model={model_name} n={config['n']} alpha={config['alpha']})"
                )
                model = build_classic_baseline(
                    model_name,
                    train_records,
                    n=config["n"],
                    alpha=config["alpha"],
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
                    _assert_variant_eval_set(eval_dir, tasks)
                    pred_dir = (
                        preds_root / dataset / f"holdout_{holdout_family}" / view / method_label
                    )
                    metrics_dir = (
                        metrics_root / dataset / f"holdout_{holdout_family}" / view / method_label
                    )
                    print(f"--> evaluating {method_label} on fixed {view} set: {eval_dir}")
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
                    run_config = {
                        **config,
                        "dataset": dataset,
                        "holdout_family": holdout_family,
                        "view": view,
                        "train_samples": len(train_records),
                        "train_sample_seed": args.seed,
                    }
                    _write_results(metrics_dir, results, tuning, run_config)
                    summary_rows.append(
                        _compact_metric_row(
                            dataset=dataset,
                            holdout_family=holdout_family,
                            view=view,
                            config=config,
                            train_samples=len(train_records),
                            results=results,
                            tuning=tuning,
                        )
                    )
                    compact = {
                        task: metrics.get("all", metrics) for task, metrics in results.items()
                    }
                    print(json.dumps(compact, indent=2))

            if summary_rows:
                summary_path = (
                    metrics_root / dataset / f"holdout_{holdout_family}" / "classic_search_summary"
                )
                _write_search_summary(
                    summary_path,
                    summary_rows,
                    rank_by=args.rank_by,
                    rank_ascending=args.rank_ascending,
                )
                print(f"wrote search summary: {summary_path}.json")


if __name__ == "__main__":
    main()
