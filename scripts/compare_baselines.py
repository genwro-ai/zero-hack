#!/usr/bin/env python3
"""Score every model on all three tasks and save a side-by-side comparison.

Reads the eval set (``outputs/eval``) and each model's submission files
(``outputs/preds/<model>``), scores Tasks 1-3, and writes:

- ``outputs/metrics/comparison.json`` — full metrics, all models and tasks.
- ``outputs/metrics/comparison.md``   — the human-readable comparison tables.

This is the persisted form of the baseline-vs-trained comparison: add a trained
model's predictions under ``outputs/preds/<name>`` and pass ``--models`` to put
it in the same tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.eval.score import score_task

# (column label, metric key, format) per task, in display order.
_COLUMNS = {
    "next_step": [
        ("top-1", "top1", "{:.4f}"),
        ("top-3", "top3", "{:.4f}"),
        ("top-5", "top5", "{:.4f}"),
        ("MRR", "mrr", "{:.4f}"),
    ],
    "completion": [
        ("exact", "exact_match", "{:.4f}"),
        ("norm-edit-dist", "norm_edit_distance", "{:.4f}"),
        ("token-acc", "token_accuracy", "{:.4f}"),
        ("block-acc", "block_accuracy", "{:.4f}"),
    ],
    "anomaly": [
        ("accuracy", "accuracy", "{:.4f}"),
        ("precision", "precision", "{:.4f}"),
        ("recall", "recall", "{:.4f}"),
        ("F1", "f1", "{:.4f}"),
        ("ROC-AUC", "roc_auc", "{}"),
        ("rule-attr", "rule_attribution_accuracy", "{}"),
        ("detected", "n_detected_violations", "{}"),
    ],
}
_TITLES = {
    "next_step": "Task 1 — Next-step prediction",
    "completion": "Task 2 — Sequence completion",
    "anomaly": "Task 3 — Anomaly detection",
}


def _gt_path(eval_dir: Path, task: str) -> Path:
    return (
        eval_dir
        / {
            "next_step": "nextstep_truth.csv",
            "completion": "completion_truth.csv",
            "anomaly": "anomaly_truth.csv",
        }[task]
    )


def _pred_path(preds_dir: Path, model: str, task: str) -> Path:
    name = {"next_step": "nextstep.csv", "completion": "completion.csv", "anomaly": "anomaly.csv"}[
        task
    ]
    return preds_dir / model / name


def _fmt(value, spec: str) -> str:
    if value is None:
        return "n/a"
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return str(value)


def _markdown_table(task: str, per_model: dict[str, dict]) -> str:
    cols = _COLUMNS[task]
    header = "| Model | " + " | ".join(label for label, _, _ in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [f"### {_TITLES[task]}", "", header, sep]
    for model, metrics in per_model.items():
        # next_step / completion expose an "all" group; anomaly is flat.
        row_metrics = metrics.get("all", metrics)
        cells = [_fmt(row_metrics.get(key), spec) for _, key, spec in cols]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=["ngram", "most_frequent"])
    parser.add_argument("--eval-dir", default=str(PROJECT_ROOT / "outputs" / "eval"))
    parser.add_argument("--preds-dir", default=str(PROJECT_ROOT / "outputs" / "preds"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    parser.add_argument("--tasks", nargs="+", default=["next_step", "completion", "anomaly"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    preds_dir = Path(args.preds_dir)
    out_dir = Path(args.out_dir)

    # results[task][model] = metrics dict
    results: dict[str, dict[str, dict]] = {task: {} for task in args.tasks}
    for model in args.models:
        for task in args.tasks:
            pred = _pred_path(preds_dir, model, task)
            if not pred.exists():
                print(f"skip {model}/{task}: missing {pred}")
                continue
            results[task][model] = score_task(
                task,
                ground_truth=_gt_path(eval_dir, task),
                predictions=pred,
                eval_input=eval_dir / "eval_input_valid.csv",
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    md_sections = ["# Baseline comparison", "", f"Models: {', '.join(args.models)}", ""]
    for task in args.tasks:
        if results[task]:
            md_sections.append(_markdown_table(task, results[task]))
    md_text = "\n".join(md_sections)
    md_path = out_dir / "comparison.md"
    md_path.write_text(md_text + "\n", encoding="utf-8")

    print(md_text)
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
