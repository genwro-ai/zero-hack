import argparse
import json
import math
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.eval import io
from zero_hack.eval.score import score_task
from zero_hack.eval.validator import first_violated_rule
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold_from_eval_dir
from zero_hack.models.lstm.inference import load_lstm_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the LSTM checkpoint (.pt).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), required=True)
    parser.add_argument("--eval-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--preds-root", default=str(PROJECT_ROOT / "outputs" / "preds"))
    parser.add_argument("--metrics-root", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    parser.add_argument("--views", nargs="+", default=["id", "ood"])
    parser.add_argument("--method-name", default="lstm")
    parser.add_argument("--calibration-dir", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed avg-logprob threshold. Omit to tune on holdout calibration set.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def main() -> None:
    args = parse_args()
    model = load_lstm_checkpoint(args.checkpoint, device=args.device or "cpu")
    print(f"loaded {args.checkpoint} (meta: {model.meta})")

    if args.threshold is not None:
        threshold = args.threshold
        threshold_record = {"source": "fixed_arg", "threshold": threshold}
    else:
        calibration_dir = (
            Path(args.calibration_dir)
            if args.calibration_dir
            else Path(args.eval_root)
            / args.dataset
            / f"holdout_{args.holdout_family}"
            / "calibration"
        )
        if not calibration_dir.exists():
            raise SystemExit(
                f"Missing threshold calibration set: {calibration_dir}. "
                "Run scripts/make_all_eval_sets.py first."
            )
        tuning = tune_anomaly_threshold_from_eval_dir(model, calibration_dir)
        threshold = tuning.threshold
        threshold_record = {
            "source": "threshold_calibration",
            "tuned_on": str(calibration_dir),
            "threshold": tuning.threshold,
            "val_f1": tuning.f1,
            "val_precision": tuning.precision,
            "val_recall": tuning.recall,
        }
    print(f"threshold={threshold:.4f} source={threshold_record['source']}")

    for view in args.views:
        eval_dir = Path(args.eval_root) / args.dataset / f"holdout_{args.holdout_family}" / view
        if not eval_dir.exists():
            raise SystemExit(f"Missing eval dir: {eval_dir}. Run scripts/make_all_eval_sets.py.")
        examples = io.read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
        rows = []
        for ex in examples:
            sequence = list(ex["sequence"])
            avg = model.score_sequence(ex["family"], sequence) / max(1, len(sequence))
            rows.append(
                {
                    "example_id": ex["example_id"],
                    **_prediction_at(threshold, avg, sequence),
                }
            )
        pred_dir = (
            Path(args.preds_root)
            / args.dataset
            / f"holdout_{args.holdout_family}"
            / view
            / args.method_name
        )
        pred_path = pred_dir / "anomaly.csv"
        io.write_anomaly_predictions(pred_path, rows)
        metrics = score_task(
            "anomaly",
            ground_truth=eval_dir / "anomaly_truth.csv",
            predictions=pred_path,
            eval_input=eval_dir / "eval_input_anomaly.csv",
        )
        metrics_dir = (
            Path(args.metrics_root)
            / args.dataset
            / f"holdout_{args.holdout_family}"
            / view
            / args.method_name
        )
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "anomaly.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        (metrics_dir / "anomaly_threshold.json").write_text(
            json.dumps(threshold_record, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{view}: n={len(rows)} metrics={metrics.get('all', metrics)}")


def _prediction_at(threshold: float, avg_logprob: float, sequence: list[str]) -> dict:
    valid = avg_logprob >= threshold
    return {
        "is_valid": int(valid),
        "score": _sigmoid(avg_logprob - threshold),
        "predicted_rule": None if valid else (first_violated_rule(sequence) or "RULE_DEP_NO_CLEAN"),
    }


if __name__ == "__main__":
    main()
