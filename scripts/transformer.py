#!/usr/bin/env python3
import argparse
from pathlib import Path

from zero_hack.data.dataio import FRACTIONS, cut, example_id, write_csv
from zero_hack.data.datasets import build_vocabulary, load_sequence_records
from zero_hack.models.transformer import StepTransformer


def write_predictions(model, records, out_dir):
    out = Path(out_dir)
    write_csv(
        out / "seqs.csv",
        ["FAMILY", "SEQUENCE_ID", "STEP"],
        [[r.family, r.sequence_id, s] for r in records for s in r.steps],
    )
    next_step, completion = [], []
    for record in records:
        for fraction in FRACTIONS:
            prefix = list(record.steps[: cut(len(record.steps), fraction)])
            ranked = (model.rank(prefix, record.family, 5) + [""] * 5)[:5]
            next_step.append([example_id(record, fraction), *ranked])
            rollout = model.complete(prefix, record.family, len(record.steps) + 20)
            completion.append([example_id(record, fraction), "|".join(rollout)])
    write_csv(out / "nextstep.csv", ["EXAMPLE_ID", *(f"RANK_{i}" for i in range(1, 6))], next_step)
    write_csv(out / "completion.csv", ["EXAMPLE_ID", "PREDICTED_SEQUENCE"], completion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.checkpoint and Path(args.checkpoint).exists():
        model = StepTransformer.load(args.checkpoint)
    else:
        train = load_sequence_records(args.train)[: args.limit]
        model = StepTransformer(build_vocabulary(train)).to(args.device)
        model.fit(train, epochs=args.epochs)
        if args.checkpoint:
            model.save(args.checkpoint)

    model.to(args.device)
    records = load_sequence_records(args.eval)[: args.limit]
    write_predictions(model, records, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
