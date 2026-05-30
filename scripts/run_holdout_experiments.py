#!/usr/bin/env python3
import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from zero_hack import PROJECT_ROOT
from zero_hack.eval import io
from zero_hack.eval.score import TASKS, score_task
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold
from zero_hack.models.classic_baselines import (
    CLASSIC_BASELINES,
    build_classic_baseline,
    complete_sequence,
    predict_anomaly,
)
from zero_hack.models.common import FAMILIES, load_split_records, pick_device
from zero_hack.models.gpt.model import GPTConfig, GPTNextStepModel
from zero_hack.models.gpt.train import CausalLMAdapter, fit_causal_lm
from zero_hack.models.neurosymbolic import SymbolicMaskAdapter
from zero_hack.models.phase_loss import NextPhaseLoss

_DATASET_SIZE = re.compile(r"_s(\d+)k$")
_VIEWS = ("id", "ood")
NEURAL_MODELS = ("gpt",)
_SCALE_METRICS = (
    ("next_step", "top1"),
    ("completion", "block_accuracy"),
    ("anomaly", "roc_auc"),
)


def _dataset_sort_key(name: str) -> tuple[int, str]:
    match = _DATASET_SIZE.search(name)
    if match:
        return int(match.group(1)), name
    return 10**12, name


def _discover_datasets(generated_root: Path) -> list[str]:
    datasets = []
    for path in generated_root.iterdir():
        if path.is_dir() and (path / "splits").exists():
            datasets.append(path.name)
    return sorted(datasets, key=_dataset_sort_key)


def _eval_input_path(eval_dir: Path, task: str) -> Path:
    if task == "anomaly":
        return eval_dir / "eval_input_anomaly.csv"
    return eval_dir / "eval_input_valid.csv"


def _truth_path(eval_dir: Path, task: str) -> Path:
    return (
        eval_dir
        / {
            "next_step": "nextstep_truth.csv",
            "completion": "completion_truth.csv",
            "anomaly": "anomaly_truth.csv",
        }[task]
    )


def _pred_path(pred_dir: Path, task: str) -> Path:
    return (
        pred_dir
        / {
            "next_step": "nextstep.csv",
            "completion": "completion.csv",
            "anomaly": "anomaly.csv",
        }[task]
    )


def _write_predictions(
    model: Any,
    *,
    eval_dir: Path,
    pred_dir: Path,
    tasks: tuple[str, ...],
    anomaly_method: str,
    anomaly_threshold: float,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)

    if "next_step" in tasks or "completion" in tasks:
        valid_inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")

        if "next_step" in tasks:
            rows = [
                {
                    "example_id": row["example_id"],
                    "ranks": model.predict_topk(row["family"], row["partial_sequence"], k=5),
                }
                for row in valid_inputs
            ]
            io.write_next_step_predictions(pred_dir / "nextstep.csv", rows)

        if "completion" in tasks:
            rows = [
                {
                    "example_id": row["example_id"],
                    "steps": complete_sequence(model, row["family"], row["partial_sequence"]),
                }
                for row in valid_inputs
            ]
            io.write_completion_predictions(pred_dir / "completion.csv", rows)

    if "anomaly" in tasks:
        anomaly_inputs = io.read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
        rows = [
            {
                "example_id": row["example_id"],
                **predict_anomaly(
                    model,
                    row["family"],
                    row["sequence"],
                    anomaly_method,
                    anomaly_threshold,
                ),
            }
            for row in anomaly_inputs
        ]
        io.write_anomaly_predictions(pred_dir / "anomaly.csv", rows)


def _score_predictions(
    *,
    eval_dir: Path,
    pred_dir: Path,
    metrics_dir: Path,
    tasks: tuple[str, ...],
) -> dict[str, dict]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results = {}
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
    (metrics_dir / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        choices=list(CLASSIC_BASELINES) + list(NEURAL_MODELS),
        default=["most_frequent", "ngram"],
    )
    parser.add_argument("--views", nargs="+", choices=_VIEWS, default=list(_VIEWS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument(
        "--anomaly-method",
        choices=("validator", "likelihood"),
        default="likelihood",
        help="Use likelihood for model-based anomaly detection; validator is an oracle baseline.",
    )
    parser.add_argument(
        "--val-anomaly-valid",
        type=int,
        default=200,
        help="Valid sequences/family in the tuning set (matches the eval ~0.39 anomaly mix).",
    )
    parser.add_argument(
        "--val-anomaly-invalid",
        type=int,
        default=129,
        help="Invalid sequences/family in the tuning set (matches the eval ~0.39 anomaly mix).",
    )
    parser.add_argument("--val-seed", type=int, default=1729)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--bucket", type=int, default=5)
    parser.add_argument(
        "--hmm-states",
        type=int,
        default=None,
        help="Hidden states for the HMM baseline. Defaults to --n.",
    )
    parser.add_argument("--hmm-iterations", type=int, default=8)
    parser.add_argument("--hmm-smoothing", type=float, default=1e-2)
    parser.add_argument("--gpt-epochs", type=int, default=30)
    parser.add_argument("--gpt-patience", type=int, default=4)
    parser.add_argument("--gpt-d-model", type=int, default=256)
    parser.add_argument("--gpt-layers", type=int, default=4)
    parser.add_argument("--gpt-batch-size", type=int, default=128)
    parser.add_argument("--gpt-lr", type=float, default=6e-4)
    parser.add_argument("--gpt-max-context", type=int, default=256)
    parser.add_argument("--gpt-valid-limit", type=int, default=2000)
    parser.add_argument("--phase-loss-weight", type=float, default=0.0)
    parser.add_argument("--neurosymbolic", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--train-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--checkpoints-root", default=str(PROJECT_ROOT / "outputs" / "checkpoints"))
    parser.add_argument("--plots-root", default=str(PROJECT_ROOT / "outputs" / "plots"))
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def _resolve_anomaly_threshold(
    model: Any,
    bundle: Any,
    *,
    method: str,
    n_valid: int,
    n_invalid: int,
    seed: int,
    tasks: tuple[str, ...],
) -> tuple[float, dict | None]:
    # Tune one global threshold per fit on the ID validation split (train
    # families only -> leakage-free), then apply it to both id and ood views.
    if "anomaly" not in tasks or method != "likelihood":
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


def _subsample_per_family(records, size):
    out, seen = [], {}
    for record in records:
        taken = seen.get(record.family, 0)
        if taken < size:
            out.append(record)
            seen[record.family] = taken + 1
    return out


def _save_gpt_checkpoint(path, model, config, bundle, train_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(config),
            "vocabulary": {
                "token_to_id": bundle.vocabulary.token_to_id,
                "id_to_token": bundle.vocabulary.id_to_token,
            },
            "train_families": list(bundle.train_families),
            "train_size": train_size,
        },
        path,
    )
    print(f"--> saved checkpoint {path}")


def _collect_scale_row(results, dataset, holdout, model_name, size, view):
    metrics = {}
    for task, key in _SCALE_METRICS:
        block = results.get(task, {})
        block = block.get("all", block)
        metrics[f"{task}.{key}"] = block.get(key)
    return {
        "dataset": dataset,
        "holdout": holdout,
        "model": model_name,
        "size": size,
        "view": view,
        "metrics": metrics,
    }


def _plot_scaling(rows, plots_root):
    plots_root = Path(plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)
    (plots_root / "scaling.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; wrote scaling.json only")
        return

    labels = [f"{task}.{key}" for task, key in _SCALE_METRICS]
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["holdout"], row["model"]), []).append(row)
    for (dataset, holdout, model_name), group in groups.items():
        sizes = sorted({row["size"] for row in group})
        fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4))
        if len(labels) == 1:
            axes = [axes]
        for ax, label in zip(axes, labels, strict=False):
            for view in ("id", "ood"):
                ys = []
                for size in sizes:
                    match = [r for r in group if r["size"] == size and r["view"] == view]
                    ys.append(match[0]["metrics"].get(label) if match else None)
                ax.plot(sizes, ys, marker="o", label=view)
            ax.set_title(label)
            ax.set_xlabel("train examples / family")
            ax.set_ylabel(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"{dataset} holdout={holdout} {model_name}")
        fig.tight_layout()
        out = plots_root / f"{dataset}_holdout_{holdout}_{model_name}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"--> wrote plot {out}")


def main() -> None:
    args = _parse_args()
    generated_root = Path(args.generated_root)
    eval_root = Path(args.eval_root)
    preds_root = Path(args.preds_root)
    metrics_root = Path(args.metrics_root)
    datasets = args.datasets or _discover_datasets(generated_root)
    tasks = tuple(args.tasks)
    device = pick_device(None)

    if not datasets:
        raise SystemExit(f"No datasets found under {generated_root}")

    sizes = args.train_sizes if args.train_sizes else [None]
    is_sweep = bool(args.train_sizes) and len(args.train_sizes) > 1
    scaling_rows = []

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

            for size in sizes:
                if size is None:
                    train_records = bundle.records["train"]
                    suffix = ""
                else:
                    train_records = _subsample_per_family(bundle.records["train"], size)
                    suffix = f"_n{size}"
                    print(f"\n-- train size {size}/family -> {len(train_records)} sequences")

                for model_name in args.models:
                    tagged = f"{model_name}{suffix}"
                    print(f"--> fitting {tagged}")
                    if model_name in NEURAL_MODELS:
                        torch.manual_seed(args.seed)
                        config = GPTConfig(
                            d_model=args.gpt_d_model,
                            nhead=8,
                            num_layers=args.gpt_layers,
                            dim_feedforward=args.gpt_d_model * 4,
                            max_context=args.gpt_max_context,
                        )
                        net = GPTNextStepModel(
                            len(bundle.vocabulary.id_to_token), config, bundle.vocabulary.pad_id
                        )
                        phase_loss = None
                        if args.phase_loss_weight > 0:
                            phase_loss = NextPhaseLoss.from_vocabulary(
                                bundle.vocabulary, weight=args.phase_loss_weight
                            ).to(device)
                        valid_records = _subsample_per_family(
                            bundle.records["valid"], args.gpt_valid_limit
                        )
                        trained = fit_causal_lm(
                            net,
                            train_records,
                            valid_records,
                            bundle.vocabulary,
                            device,
                            args.gpt_epochs,
                            args.gpt_batch_size,
                            args.gpt_lr,
                            args.gpt_patience,
                            args.num_workers,
                            phase_loss,
                        )
                        model = CausalLMAdapter(trained, bundle.vocabulary, device)
                        if args.save_checkpoints:
                            ckpt = (
                                Path(args.checkpoints_root)
                                / dataset
                                / f"holdout_{holdout_family}"
                                / f"{tagged}.pt"
                            )
                            _save_gpt_checkpoint(ckpt, trained, trained.config, bundle, size)
                    elif not train_records:
                        print(f"--> skip {tagged}: no training data")
                        continue
                    else:
                        model = build_classic_baseline(
                            model_name,
                            train_records,
                            n=args.n,
                            alpha=args.alpha,
                            bucket=args.bucket,
                            hmm_states=args.hmm_states,
                            hmm_iterations=args.hmm_iterations,
                            hmm_smoothing=args.hmm_smoothing,
                            seed=args.seed,
                        )

                    if args.neurosymbolic:
                        model = SymbolicMaskAdapter(model, bundle.vocabulary)

                    threshold, tuning = _resolve_anomaly_threshold(
                        model,
                        bundle,
                        method=args.anomaly_method,
                        n_valid=args.val_anomaly_valid,
                        n_invalid=args.val_anomaly_invalid,
                        seed=args.val_seed,
                        tasks=tasks,
                    )
                    if tuning is not None:
                        print(
                            f"--> tuned anomaly threshold for {tagged}: {threshold:.4f} "
                            f"(val F1={tuning['val_f1']:.4f} "
                            f"P={tuning['val_precision']:.4f} R={tuning['val_recall']:.4f})"
                        )

                    for view in args.views:
                        eval_dir = eval_root / dataset / f"holdout_{holdout_family}" / view
                        if not eval_dir.exists():
                            raise SystemExit(
                                f"Missing eval directory: {eval_dir}. "
                                "Run local_scripts/make_eval_sets.sh first."
                            )
                        pred_dir = (
                            preds_root / dataset / f"holdout_{holdout_family}" / view / tagged
                        )
                        metrics_dir = (
                            metrics_root / dataset / f"holdout_{holdout_family}" / view / tagged
                        )
                        print(f"--> evaluating {tagged} on {view}: {eval_dir}")
                        _write_predictions(
                            model,
                            eval_dir=eval_dir,
                            pred_dir=pred_dir,
                            tasks=tasks,
                            anomaly_method=args.anomaly_method,
                            anomaly_threshold=threshold,
                        )
                        results = _score_predictions(
                            eval_dir=eval_dir,
                            pred_dir=pred_dir,
                            metrics_dir=metrics_dir,
                            tasks=tasks,
                        )
                        if tuning is not None:
                            (metrics_dir / "anomaly_threshold.json").write_text(
                                json.dumps(tuning, indent=2) + "\n", encoding="utf-8"
                            )
                        compact = {
                            task: metrics.get("all", metrics) for task, metrics in results.items()
                        }
                        print(json.dumps(compact, indent=2))
                        if model_name in NEURAL_MODELS:
                            point = size if size is not None else len(train_records)
                            scaling_rows.append(
                                _collect_scale_row(
                                    results, dataset, holdout_family, model_name, point, view
                                )
                            )

    if scaling_rows and is_sweep:
        _plot_scaling(scaling_rows, args.plots_root)


if __name__ == "__main__":
    main()
