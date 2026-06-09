#!/usr/bin/env python3
import argparse
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.eval import io
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold
from zero_hack.models.classic_baselines import complete_sequence, predict_anomaly
from zero_hack.models.common import load_split_records, pick_device
from zero_hack.models.gpt_lm import EnsemblePredictor, load_member


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members-dir", required=True)
    parser.add_argument("--max-members", type=int, default=None)
    parser.add_argument("--tune-splits-dir", required=True)
    parser.add_argument(
        "--valid-input", default=str(PROJECT_ROOT / "data/industrial/eval_input_valid.csv")
    )
    parser.add_argument(
        "--anomaly-input", default=str(PROJECT_ROOT / "data/industrial/eval_input_anomaly.csv")
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(None)
    paths = sorted(Path(args.members_dir).glob("*.pt"))[: args.max_members]
    if not paths:
        raise SystemExit(f"no members in {args.members_dir}")
    models, vocab, config = [], None, None
    for path in paths:
        model, vocab, config = load_member(path, device)
        models.append(model)
    predictor = EnsemblePredictor(models, vocab, device, config["max_len"])
    print(f"loaded {len(models)} members")

    bundle = load_split_records(args.tune_splits_dir)
    threshold = tune_anomaly_threshold(
        predictor, bundle.records["valid"], n_valid=200, n_invalid=129, seed=1729
    ).threshold
    print(f"anomaly threshold {threshold:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    valid = io.read_eval_input_valid(Path(args.valid_input))
    io.write_next_step_predictions(
        out / "nextstep.csv",
        [
            {
                "example_id": r["example_id"],
                "ranks": predictor.predict_topk(r["family"], r["partial_sequence"], 5),
            }
            for r in valid
        ],
    )
    io.write_completion_predictions(
        out / "completion.csv",
        [
            {
                "example_id": r["example_id"],
                "steps": complete_sequence(predictor, r["family"], r["partial_sequence"]),
            }
            for r in valid
        ],
    )
    anomaly = io.read_eval_input_anomaly(Path(args.anomaly_input))
    io.write_anomaly_predictions(
        out / "anomaly.csv",
        [
            {
                "example_id": r["example_id"],
                **predict_anomaly(predictor, r["family"], r["sequence"], "likelihood", threshold),
            }
            for r in anomaly
        ],
    )
    print(f"wrote 3 csv to {out}  ({len(valid)} valid, {len(anomaly)} anomaly)")


if __name__ == "__main__":
    main()
