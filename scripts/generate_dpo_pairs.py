#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from zero_hack.data import SequenceRecord
from zero_hack.eval.anomaly_synth import RULE_IDS, corrupt_steps
from zero_hack.eval.validator import first_violated_rule, is_valid
from zero_hack.models.common import FAMILIES, load_split_records
from zero_hack.models.gpt.train import _augment_training_records, _load_augmentation_records


def _prefix_cut(
    steps: tuple[str, ...],
    rng: random.Random,
    *,
    min_fraction: float,
    max_fraction: float,
) -> int:
    fraction = rng.uniform(min_fraction, max_fraction)
    cut = int(len(steps) * fraction)
    return max(1, min(cut, len(steps) - 2))


def _rule_invalid_suffix(
    record: SequenceRecord,
    cut: int,
    rng: random.Random,
    *,
    max_tries: int,
) -> tuple[list[str], str] | None:
    prefix = list(record.steps[:cut])
    rules = list(RULE_IDS)
    rng.shuffle(rules)

    for rule in rules:
        for _ in range(max_tries):
            corrupted = corrupt_steps(list(record.steps), rng, target_rule=rule)
            if corrupted is None:
                continue
            corrupted_steps, observed_rule = corrupted
            rejected = corrupted_steps[cut:]
            full = prefix + rejected
            violated = first_violated_rule(full)
            if rejected and tuple(full) != record.steps and violated is not None:
                return rejected, violated or observed_rule
    return None


def _valid_mismatch_suffix(
    record: SequenceRecord,
    cut: int,
    records: list[SequenceRecord],
    rng: random.Random,
    *,
    max_tries: int,
) -> list[str] | None:
    prefix = list(record.steps[:cut])
    candidates = list(records)
    rng.shuffle(candidates)

    for other in candidates[:max_tries]:
        if other.sequence_id == record.sequence_id or len(other.steps) <= cut + 1:
            continue
        rejected = list(other.steps[cut:])
        full = prefix + rejected
        if rejected and tuple(full) != record.steps and is_valid(full):
            return rejected
    return None


def _build_pair(
    pair_id: int,
    record: SequenceRecord,
    records: list[SequenceRecord],
    rng: random.Random,
    args: argparse.Namespace,
) -> dict | None:
    if len(record.steps) < 8:
        return None

    cut = _prefix_cut(
        record.steps,
        rng,
        min_fraction=args.min_prefix_fraction,
        max_fraction=args.max_prefix_fraction,
    )
    prefix = list(record.steps[:cut])
    chosen = list(record.steps[cut:])
    negative_type = rng.choices(
        ["rule_invalid", "valid_mismatch"],
        weights=[args.invalid_weight, args.valid_mismatch_weight],
        k=1,
    )[0]

    rule = None
    if negative_type == "rule_invalid":
        invalid = _rule_invalid_suffix(record, cut, rng, max_tries=args.max_tries)
        if invalid is None:
            rejected = None
        else:
            rejected, rule = invalid
    else:
        rejected = _valid_mismatch_suffix(record, cut, records, rng, max_tries=args.max_tries)

    if rejected is None:
        fallback = _rule_invalid_suffix(record, cut, rng, max_tries=args.max_tries)
        if fallback is None:
            return None
        rejected, rule = fallback
        negative_type = "rule_invalid"

    return {
        "pair_id": f"pair_{pair_id:08d}",
        "family": record.family,
        "sequence_id": record.sequence_id,
        "prefix": prefix,
        "chosen": chosen,
        "rejected": rejected,
        "negative_type": negative_type,
        "rule": rule,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--holdout-family", choices=FAMILIES, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--augment-train-csv", default=None)
    parser.add_argument(
        "--augment-family-mode",
        choices=("unknown", "preserve-known"),
        default="unknown",
    )
    parser.add_argument("--pairs", type=int, default=100_000)
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--min-prefix-fraction", type=float, default=0.25)
    parser.add_argument("--max-prefix-fraction", type=float, default=0.75)
    parser.add_argument("--invalid-weight", type=float, default=0.7)
    parser.add_argument("--valid-mismatch-weight", type=float, default=0.3)
    parser.add_argument("--max-tries", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    bundle = load_split_records(
        args.splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    if args.augment_train_csv:
        augmentation = _load_augmentation_records(
            args.augment_train_csv,
            family_mode=args.augment_family_mode,
        )
        bundle = _augment_training_records(bundle, augmentation)

    records = list(bundle.records["train"])
    if not records:
        raise SystemExit("No training records available for DPO pair generation")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    attempts = 0
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        while written < args.pairs and attempts < args.pairs * 20:
            attempts += 1
            record = rng.choice(records)
            pair = _build_pair(written + 1, record, records, rng, args)
            if pair is None:
                continue
            handle.write(json.dumps(pair) + "\n")
            counts[pair["negative_type"]] = counts.get(pair["negative_type"], 0) + 1
            written += 1

    metadata = {
        "splits_dir": str(Path(args.splits_dir)),
        "holdout_family": args.holdout_family,
        "train_families": list(bundle.train_families),
        "augment_train_csv": args.augment_train_csv,
        "pairs_requested": args.pairs,
        "pairs_written": written,
        "attempts": attempts,
        "negative_type_counts": counts,
        "seed": args.seed,
    }
    (out.with_suffix(out.suffix + ".meta.json")).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
