#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

from zero_hack import INDUSTRIAL_DATA_DIR
from zero_hack.data.dataio import FRACTIONS, cut, example_id, read_csv
from zero_hack.data.datasets import load_sequence_records
from zero_hack.eval import metrics


def load_validator():
    path = INDUSTRIAL_DATA_DIR / "generate_sequences.py"
    spec = importlib.util.spec_from_file_location("generate_sequences", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_sequence


validate = load_validator()


def task_next_step(records, path):
    cols = ("RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5")
    pred = {row["EXAMPLE_ID"]: [row[c] for c in cols if row.get(c)] for row in read_csv(path)}
    pairs = [
        (pred.get(example_id(r, f), []), r.steps[cut(len(r.steps), f)])
        for r in records for f in FRACTIONS
    ]
    return metrics.next_step(pairs)


def task_completion(records, path):
    pred = {
        row["EXAMPLE_ID"]: row["PREDICTED_SEQUENCE"].split("|") if row["PREDICTED_SEQUENCE"] else []
        for row in read_csv(path)
    }
    pairs = [
        (pred.get(example_id(r, f), []), list(r.steps[cut(len(r.steps), f):]))
        for r in records for f in FRACTIONS
    ]
    return metrics.completion(pairs)


def task_anomaly(sequences_path, pred_path):
    pred = {}
    for row in read_csv(pred_path):
        score = float(row["SCORE"]) if row.get("SCORE") else 0.5
        pred[row["EXAMPLE_ID"]] = (int(row["IS_VALID"]), score, row.get("PREDICTED_RULE") or None)
    rows = []
    for row in read_csv(sequences_path):
        violations = validate(row["SEQUENCE"].split("|"))
        valid_true = int(not violations)
        rule_true = violations[0].rule if violations else None
        guess = pred.get(row["EXAMPLE_ID"], (1, 0.5, None))
        rows.append((*guess, valid_true, rule_true))
    return metrics.anomaly(rows)


def evaluate(sequences_path, pred_dir, anomaly_path):
    records = load_sequence_records(sequences_path)
    pred_dir = Path(pred_dir)
    result = {
        "next_step": task_next_step(records, pred_dir / "nextstep.csv"),
        "completion": task_completion(records, pred_dir / "completion.csv"),
    }
    if anomaly_path:
        result["anomaly"] = task_anomaly(anomaly_path, pred_dir / "anomaly.csv")
    return result


def show(title, result):
    print(title)
    for task, scores in result.items():
        line = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
        print(f"  {task:<11} {line}")


def show_drop(id_result, ood_result):
    print("drop (ID - OOD)")
    for task in id_result:
        if task in ood_result:
            line = "  ".join(
                f"{k}={id_result[task][k] - ood_result[task][k]:+.3f}" for k in id_result[task]
            )
            print(f"  {task:<11} {line}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    parser.add_argument("--ood", required=True)
    parser.add_argument("--id-pred", required=True)
    parser.add_argument("--ood-pred", required=True)
    parser.add_argument("--id-anomaly")
    parser.add_argument("--ood-anomaly")
    args = parser.parse_args()

    id_result = evaluate(args.id, args.id_pred, args.id_anomaly)
    ood_result = evaluate(args.ood, args.ood_pred, args.ood_anomaly)
    show("ID", id_result)
    show("OOD", ood_result)
    show_drop(id_result, ood_result)


if __name__ == "__main__":
    main()
