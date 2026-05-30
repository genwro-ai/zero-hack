import argparse
import json
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.eval import io
from zero_hack.eval.score import score_task
from zero_hack.eval.validator import validate_sequence
from zero_hack.models.classic_baselines import complete_sequence
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
    parser.add_argument("--max-steps", type=int, default=400, help="Generation length cap.")
    parser.add_argument(
        "--enforce-rules",
        action="store_true",
        help="Apply the ViolationMask during generation (mask rule-violating next steps).",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or "cpu"
    model = load_lstm_checkpoint(args.checkpoint, device=device, enforce_rules=args.enforce_rules)
    print(f"loaded {args.checkpoint} (meta: {model.meta}, enforce_rules={args.enforce_rules})")

    for view in args.views:
        eval_dir = Path(args.eval_root) / args.dataset / f"holdout_{args.holdout_family}" / view
        if not eval_dir.exists():
            raise SystemExit(f"Missing eval dir: {eval_dir}. Run scripts/make_all_eval_sets.py.")
        inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
        pred_rows = []
        valid_hits = 0
        for row in inputs:
            prefix = list(row["partial_sequence"])
            pred_suffix = complete_sequence(
                model,
                row["family"],
                prefix,
                max_steps=args.max_steps,
            )
            pred_rows.append({"example_id": row["example_id"], "steps": pred_suffix})
            valid_hits += int(not validate_sequence(prefix + pred_suffix))

        pred_dir = (
            Path(args.preds_root)
            / args.dataset
            / f"holdout_{args.holdout_family}"
            / view
            / args.method_name
        )
        pred_path = pred_dir / "completion.csv"
        io.write_completion_predictions(pred_path, pred_rows)
        metrics = score_task(
            "completion",
            ground_truth=eval_dir / "completion_truth.csv",
            predictions=pred_path,
            eval_input=eval_dir / "eval_input_valid.csv",
        )
        metrics_dir = (
            Path(args.metrics_root)
            / args.dataset
            / f"holdout_{args.holdout_family}"
            / view
            / args.method_name
        )
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "completion.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        validity_rate = round(valid_hits / len(inputs), 4) if inputs else 0.0
        print(
            f"{view}: n={len(inputs)} validity={validity_rate:.4f} "
            f"metrics={metrics.get('all', metrics)}"
        )


if __name__ == "__main__":
    main()
