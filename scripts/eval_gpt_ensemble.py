#!/usr/bin/env python3
import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.eval import io
from zero_hack.eval.anomaly_synth import build_rule_stratified_corruptions
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold
from zero_hack.models.classic_baselines import complete_sequence, predict_anomaly
from zero_hack.models.common import load_split_records, pick_device
from zero_hack.models.gpt_lm import EnsemblePredictor, load_member

EVAL_METRICS = PROJECT_ROOT / "data/industrial/eval_metrics.py"

TASK_METRICS = {
    "next-step": [
        ("top1", "Top-1 Accuracy"),
        ("top3", "Top-3 Accuracy"),
        ("top5", "Top-5 Accuracy"),
        ("mrr", "MRR"),
    ],
    "completion": [
        ("ned", "Mean Normalized Edit Distance"),
        ("exact", "Exact Match Rate"),
        ("token", "Mean Token Accuracy"),
        ("block", "Mean Block-level Accuracy"),
    ],
    "anomaly": [("acc", "Binary Accuracy"), ("f1", "F1 (invalid class)"), ("auc", "ROC-AUC")],
}


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def build_ground_truth(records, view, gt_dir, n_valid, n_invalid, fractions, seed):
    gt_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)

    valid_rows, supplement_rows = [], []
    for i, record in enumerate(pool[:n_valid]):
        full = list(record.steps)
        supplement_rows.append((f"{view}_valid_{i:04d}", record.family, "|".join(full)))
        for frac in fractions:
            cut = max(1, min(int(len(full) * frac), len(full) - 1))
            valid_rows.append(
                (
                    f"{view}_v{i:04d}_f{int(frac * 100)}",
                    record.family,
                    str(frac),
                    "|".join(full[:cut]),
                    full[cut],
                    "|".join(full),
                )
            )

    corruptions = build_rule_stratified_corruptions(pool, n_invalid=n_invalid, rng=rng)
    forbidden_rows = [
        (f"{view}_forb_{i:04d}", c.family, "|".join(c.steps), c.rule)
        for i, c in enumerate(corruptions)
    ]

    write_csv(
        gt_dir / "valid.csv",
        [
            "EXAMPLE_ID",
            "FAMILY",
            "COMPLETION_FRACTION",
            "PARTIAL_SEQUENCE",
            "NEXT_STEP",
            "FULL_SEQUENCE",
        ],
        valid_rows,
    )
    write_csv(
        gt_dir / "forbidden.csv",
        ["EXAMPLE_ID", "FAMILY", "SEQUENCE", "VIOLATION_RULE"],
        forbidden_rows,
    )
    write_csv(
        gt_dir / "valid_supplement.csv", ["EXAMPLE_ID", "FAMILY", "SEQUENCE"], supplement_rows
    )

    anomaly_input = [(eid, fam, seq) for eid, fam, seq, _ in forbidden_rows] + supplement_rows
    rng.shuffle(anomaly_input)
    write_csv(gt_dir / "anomaly_input.csv", ["EXAMPLE_ID", "FAMILY", "SEQUENCE"], anomaly_input)


def run_eval_metrics(task, ground_truth, predictions, supplement=None):
    cmd = [
        sys.executable,
        str(EVAL_METRICS),
        "--task",
        task,
        "--ground-truth",
        str(ground_truth),
        "--predictions",
        str(predictions),
    ]
    if supplement is not None:
        cmd += ["--valid-supplement", str(supplement)]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def grab(text, label):
    for line in text.splitlines():
        if label in line and ":" in line:
            token = line.split(":", 1)[1].strip().split()[0]
            try:
                return float(token)
            except ValueError:
                return None
    return None


def predict_and_score(predictor, gt_dir, pred_dir, threshold):
    pred_dir.mkdir(parents=True, exist_ok=True)
    valid = io.read_eval_input_valid(gt_dir / "valid.csv")
    io.write_next_step_predictions(
        pred_dir / "nextstep.csv",
        [
            {
                "example_id": r["example_id"],
                "ranks": predictor.predict_topk(r["family"], r["partial_sequence"], 5),
            }
            for r in valid
        ],
    )
    io.write_completion_predictions(
        pred_dir / "completion.csv",
        [
            {
                "example_id": r["example_id"],
                "steps": complete_sequence(predictor, r["family"], r["partial_sequence"]),
            }
            for r in valid
        ],
    )
    anomaly = io.read_eval_input_anomaly(gt_dir / "anomaly_input.csv")
    io.write_anomaly_predictions(
        pred_dir / "anomaly.csv",
        [
            {
                "example_id": r["example_id"],
                **predict_anomaly(predictor, r["family"], r["sequence"], "likelihood", threshold),
            }
            for r in anomaly
        ],
    )

    out = {}
    out["next-step"] = run_eval_metrics(
        "next-step", gt_dir / "valid.csv", pred_dir / "nextstep.csv"
    )
    out["completion"] = run_eval_metrics(
        "completion", gt_dir / "valid.csv", pred_dir / "completion.csv"
    )
    out["anomaly"] = run_eval_metrics(
        "anomaly",
        gt_dir / "forbidden.csv",
        pred_dir / "anomaly.csv",
        gt_dir / "valid_supplement.csv",
    )
    return {
        task: {key: grab(out[task], label) for key, label in cols}
        for task, cols in TASK_METRICS.items()
    }


def cell(value):
    return f"{value:7.3f}" if isinstance(value, (int, float)) else f"{'-':>7}"


def delta(after, before):
    if isinstance(after, (int, float)) and isinstance(before, (int, float)):
        return f"{after - before:+7.3f}"
    return f"{'-':>7}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members-dir", required=True)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), required=True)
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--n-valid", type=int, default=100)
    parser.add_argument("--n-invalid", type=int, default=130)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(None)
    paths = sorted(Path(args.members_dir).glob("*.pt"))
    if not paths:
        raise SystemExit(f"no members in {args.members_dir}")
    models, vocab, config = [], None, None
    for path in paths:
        model, vocab, config = load_member(path, device)
        models.append(model)
    print(f"loaded {len(models)} members")

    bundle = load_split_records(
        args.splits_dir, holdout_family=args.holdout_family, limit_per_family=args.limit_per_family
    )
    out = Path(args.out)
    sources = {
        "id": bundle.records["test"],
        f"ood_{args.holdout_family}": bundle.records[f"test_{args.holdout_family}"],
    }
    for view, records in sources.items():
        build_ground_truth(
            records, view, out / "gt" / view, args.n_valid, args.n_invalid, [0.6, 0.8], 1729
        )
        print(f"built ground truth: {view}", flush=True)

    configs = {"1 GPT": [models[0]], f"{len(models)} GPT": models}
    results = {}
    for label, member_list in configs.items():
        predictor = EnsemblePredictor(member_list, vocab, device, config["max_len"])
        threshold = tune_anomaly_threshold(
            predictor, bundle.records["valid"], n_valid=200, n_invalid=129, seed=1729
        ).threshold
        print(f"[{label}] threshold={threshold:.4f}", flush=True)
        results[label] = {}
        for view in sources:
            print(f"[{label}] scoring {view} ...", flush=True)
            results[label][view] = predict_and_score(
                predictor, out / "gt" / view, out / label.replace(" ", "") / view, threshold
            )

    labels = list(configs)
    print(f"\n#### holdout={args.holdout_family} | {labels[0]} vs {labels[1]} ####")
    for task, cols in TASK_METRICS.items():
        print(f"\n== {task} ==")
        print(f"{'':16s}" + "".join(f"{short:>8}" for short, _ in cols))
        for view in sources:
            for label in labels:
                row = results[label][view][task]
                print(f"{label + ' ' + view:16s}" + "".join(cell(row[key]) for key, _ in cols))
            before = results[labels[0]][view][task]
            after = results[labels[1]][view][task]
            print(
                f"{'diff ' + view:16s}" + "".join(delta(after[key], before[key]) for key, _ in cols)
            )


if __name__ == "__main__":
    main()
