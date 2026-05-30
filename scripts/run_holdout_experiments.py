#!/usr/bin/env python3
import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from zero_hack import PROJECT_ROOT
from zero_hack.data import SequenceRecord
from zero_hack.eval.anomaly import score_anomaly
from zero_hack.eval.anomaly_synth import build_rule_stratified_corruptions
from zero_hack.eval.completion import score_completion
from zero_hack.eval.next_step import score_next_step
from zero_hack.eval.score import TASKS
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold
from zero_hack.models.classic_baselines import (
    CLASSIC_BASELINES,
    build_classic_baseline,
    complete_sequence,
    predict_anomaly,
)
from zero_hack.models.common import FAMILIES, family_test_split, load_split_records

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


def _records_by_family(records: list[SequenceRecord]) -> dict[str, list[SequenceRecord]]:
    by_family: dict[str, list[SequenceRecord]] = {}
    for record in records:
        by_family.setdefault(record.family, []).append(record)
    return by_family


def _limit_records_per_family(
    records: list[SequenceRecord],
    limit: int | None,
    *,
    seed: int,
) -> list[SequenceRecord]:
    if limit is None:
        return records
    rng = random.Random(seed)
    selected: list[SequenceRecord] = []
    for _family, family_records in sorted(_records_by_family(records).items()):
        family_records = list(family_records)
        rng.shuffle(family_records)
        selected.extend(family_records[:limit])
    return selected


def _records_for_view(bundle: Any, view: str) -> list[SequenceRecord]:
    if view == "id":
        return bundle.records["test"]
    if bundle.holdout_family is None:
        return []
    return bundle.records[family_test_split(bundle.holdout_family)]


def _score_next_step(model: Any, records: list[SequenceRecord]) -> dict:
    truth: dict[str, str] = {}
    predictions: dict[str, list[str]] = {}
    families: dict[str, str] = {}
    for record in records:
        for position, gold_step in enumerate(record.steps):
            example_id = f"{record.family}_{record.sequence_id}_{position:04d}"
            truth[example_id] = gold_step
            predictions[example_id] = model.predict_topk(
                record.family,
                record.steps[:position],
                k=5,
            )
            families[example_id] = record.family
    return score_next_step(truth, predictions, families=families)


def _score_completion(
    model: Any,
    records: list[SequenceRecord],
    *,
    fractions: tuple[float, ...],
) -> dict:
    truth: dict[str, list[str]] = {}
    predictions: dict[str, list[str]] = {}
    families: dict[str, str] = {}
    for record in records:
        steps = list(record.steps)
        for fraction in fractions:
            cut = int(len(steps) * fraction)
            cut = max(1, min(cut, len(steps) - 1))
            example_id = f"{record.family}_{record.sequence_id}_f{int(fraction * 100)}"
            prefix = steps[:cut]
            truth[example_id] = steps[cut:]
            predictions[example_id] = complete_sequence(model, record.family, prefix)
            families[example_id] = record.family
    return score_completion(truth, predictions, families=families)


def _score_anomaly_task(
    model: Any,
    records: list[SequenceRecord],
    *,
    threshold: float,
    n_valid_per_family: int,
    n_invalid_per_family: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    truth: dict[str, dict] = {}
    predictions: dict[str, dict] = {}
    families: dict[str, str] = {}

    for family, family_records in sorted(_records_by_family(records).items()):
        family_records = list(family_records)
        rng.shuffle(family_records)
        valid_records = family_records[:n_valid_per_family]

        invalid_examples = build_rule_stratified_corruptions(
            family_records,
            n_invalid=n_invalid_per_family,
            rng=rng,
        )
        for record in valid_records:
            example_id = f"{family}_{record.sequence_id}_ok"
            truth[example_id] = {"is_valid": 1, "rule": None}
            predictions[example_id] = predict_anomaly(
                model,
                family,
                list(record.steps),
                "likelihood",
                threshold,
            )
            families[example_id] = family

        for idx, example in enumerate(invalid_examples):
            example_id = f"{family}_{example.sequence_id}_bad_{idx:04d}_{example.rule}"
            truth[example_id] = {"is_valid": 0, "rule": example.rule}
            predictions[example_id] = predict_anomaly(
                model,
                family,
                list(example.steps),
                "likelihood",
                threshold,
            )
            families[example_id] = family

    return score_anomaly(truth, predictions, families=families)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classic baselines and score holdout splits directly from split records.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--generated-root", default=str(PROJECT_ROOT / "data" / "generated"))
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
        default=["most_frequent", "ngram"],
    )
    parser.add_argument("--views", nargs="+", choices=_VIEWS, default=list(_VIEWS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument(
        "--max-eval-per-family",
        type=int,
        default=None,
        help="Optional cap on records/family for next-step and completion scoring.",
    )
    parser.add_argument("--completion-fractions", type=float, nargs="+", default=[0.6, 0.8])
    parser.add_argument(
        "--val-anomaly-valid",
        type=int,
        default=200,
        help="Valid sequences/family in the tuning set.",
    )
    parser.add_argument(
        "--val-anomaly-invalid",
        type=int,
        default=129,
        help="Invalid sequences/family in the tuning set.",
    )
    parser.add_argument(
        "--eval-anomaly-valid",
        type=int,
        default=200,
        help="Valid sequences/family for direct anomaly scoring.",
    )
    parser.add_argument(
        "--eval-anomaly-invalid",
        type=int,
        default=129,
        help="Invalid sequences/family for direct anomaly scoring.",
    )
    parser.add_argument("--val-seed", type=int, default=1729)
    parser.add_argument("--eval-seed", type=int, default=1729)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--bucket", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def _resolve_anomaly_threshold(
    model: Any,
    bundle: Any,
    *,
    n_valid: int,
    n_invalid: int,
    seed: int,
    tasks: tuple[str, ...],
) -> tuple[float, dict | None]:
    if "anomaly" not in tasks:
        return -1.0, None

    result = tune_anomaly_threshold(
        model,
        bundle.records["valid"],
        n_valid=n_valid,
        n_invalid=n_invalid,
        seed=seed,
    )
    record = {
        "source": "auto",
        "objective": "f1",
        "tuned_on": "id_validation_train_families",
        "train_families": list(bundle.train_families),
        "threshold": result.threshold,
        "val_f1": result.f1,
        "val_precision": result.precision,
        "val_recall": result.recall,
        "n_valid_per_family": n_valid,
        "n_invalid_per_family": n_invalid,
        "seed": seed,
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
    metrics_root = Path(args.metrics_root)
    datasets = args.datasets or _discover_datasets(generated_root)
    tasks = tuple(args.tasks)
    completion_fractions = tuple(args.completion_fractions)

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
                    bucket=args.bucket,
                    seed=args.seed,
                )

                threshold, tuning = _resolve_anomaly_threshold(
                    model,
                    bundle,
                    n_valid=args.val_anomaly_valid,
                    n_invalid=args.val_anomaly_invalid,
                    seed=args.val_seed,
                    tasks=tasks,
                )
                if tuning is not None:
                    print(
                        f"--> tuned anomaly threshold for {model_name}: {threshold:.4f} "
                        f"(val F1={tuning['val_f1']:.4f} "
                        f"P={tuning['val_precision']:.4f} R={tuning['val_recall']:.4f})"
                    )

                for view in args.views:
                    records = _records_for_view(bundle, view)
                    records = _limit_records_per_family(
                        records,
                        args.max_eval_per_family,
                        seed=args.eval_seed,
                    )
                    print(f"--> evaluating {model_name} on {view}: {len(records)} records")

                    results: dict[str, dict] = {}
                    if "next_step" in tasks:
                        results["next_step"] = _score_next_step(model, records)
                    if "completion" in tasks:
                        results["completion"] = _score_completion(
                            model,
                            records,
                            fractions=completion_fractions,
                        )
                    if "anomaly" in tasks:
                        results["anomaly"] = _score_anomaly_task(
                            model,
                            records,
                            threshold=threshold,
                            n_valid_per_family=args.eval_anomaly_valid,
                            n_invalid_per_family=args.eval_anomaly_invalid,
                            seed=args.eval_seed,
                        )

                    metrics_dir = (
                        metrics_root / dataset / f"holdout_{holdout_family}" / view / model_name
                    )
                    _write_results(metrics_dir, results, tuning)
                    compact = {
                        task: metrics.get("all", metrics) for task, metrics in results.items()
                    }
                    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
