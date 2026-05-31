"""Evaluate a trained LSTM on the sequence-completion task (Task 2).

Unlike the teacher-forced next-step accuracy printed during training, this
measures *actual sequence prediction*: each test sequence is cut at a fraction,
the kept prefix is handed to the model, and the model generates the rest
autoregressively from its own outputs (free running). The generated suffix is
scored against the true suffix with the standard completion metrics
(exact-match, normalized edit distance, token accuracy, block accuracy), and we
additionally report how often the reconstructed full route is process-valid.

Usage:
    uv run python scripts/eval_lstm_completion.py \
        --checkpoint outputs/models/valid_s005k/lstm_scheduled_sampling/best.pt \
        --out outputs/metrics/valid_s005k/lstm_scheduled_sampling/completion
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from zero_hack.data import FAMILY_FILE_NAMES, load_sequence_records
from zero_hack.eval.completion import score_completion
from zero_hack.eval.io import write_completion_predictions
from zero_hack.eval.validator import validate_sequence
from zero_hack.models.classic_baselines import complete_sequence
from zero_hack.models.common import DEFAULT_SPLITS_DIR
from zero_hack.models.lstm.inference import load_lstm_checkpoint

FAMILIES = tuple(FAMILY_FILE_NAMES)


def _test_split_path(splits_dir: Path, family: str) -> Path:
    stem = FAMILY_FILE_NAMES[family].removesuffix(".csv")
    return splits_dir / f"{stem}_test.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to the LSTM checkpoint (.pt).")
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.6, 0.8],
        help="Prefix fractions to keep before generating the rest.",
    )
    parser.add_argument(
        "--max-examples-per-family",
        type=int,
        default=100,
        help="Cap sequences sampled per family (per fraction). 0 = no cap.",
    )
    parser.add_argument("--max-steps", type=int, default=400, help="Generation length cap.")
    parser.add_argument(
        "--enforce-rules",
        action="store_true",
        help="Apply the ViolationMask during generation (mask rule-violating next steps).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for results.{json,md} and predictions.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or "cpu"
    model = load_lstm_checkpoint(args.checkpoint, device=device, enforce_rules=args.enforce_rules)
    print(f"loaded {args.checkpoint} (meta: {model.meta}, enforce_rules={args.enforce_rules})")

    splits_dir = Path(args.splits_dir)
    rng = random.Random(args.seed)

    truth: dict[str, list[str]] = {}
    predictions: dict[str, list[str]] = {}
    families: dict[str, str] = {}
    fractions: dict[str, float] = {}
    pred_rows: list[dict] = []
    valid_hits = 0
    valid_total = 0

    for family in args.families:
        records = load_sequence_records(_test_split_path(splits_dir, family), family=family)
        records = [r for r in records if len(r.steps) >= 2]
        rng.shuffle(records)
        if args.max_examples_per_family:
            records = records[: args.max_examples_per_family]
        print(f"{family}: {len(records)} sequences")

        for record in records:
            steps = list(record.steps)
            for fraction in args.fractions:
                cut = max(1, int(round(fraction * len(steps))))
                if cut >= len(steps):
                    continue  # nothing left to predict
                prefix, gold_suffix = steps[:cut], steps[cut:]
                pred_suffix = complete_sequence(model, family, prefix, max_steps=args.max_steps)

                example_id = f"{family}_{record.sequence_id}_f{int(round(fraction * 100))}"
                truth[example_id] = gold_suffix
                predictions[example_id] = pred_suffix
                families[example_id] = family
                fractions[example_id] = fraction
                pred_rows.append({"example_id": example_id, "steps": pred_suffix})

                valid_total += 1
                valid_hits += int(not validate_sequence(prefix + pred_suffix))

    if not truth:
        raise SystemExit("No completion examples were produced; check the splits dir / fractions.")

    overall = score_completion(truth, predictions, families)
    by_fraction = {}
    for fraction in sorted(set(fractions.values())):
        ids = {eid for eid, f in fractions.items() if f == fraction}
        sub_truth = {eid: truth[eid] for eid in ids}
        sub_pred = {eid: predictions[eid] for eid in ids}
        sub_fam = {eid: families[eid] for eid in ids}
        by_fraction[f"{fraction:.2f}"] = score_completion(sub_truth, sub_pred, sub_fam)

    validity_rate = round(valid_hits / valid_total, 4) if valid_total else 0.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(args.checkpoint),
        "meta": model.meta,
        "enforce_rules": args.enforce_rules,
        "n_examples": valid_total,
        "fractions": args.fractions,
        "overall": overall,
        "by_fraction": by_fraction,
        "generated_route_validity_rate": validity_rate,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_completion_predictions(out_dir / "predictions.csv", pred_rows)

    md = _render_md(args.checkpoint, overall, by_fraction, validity_rate, valid_total)
    (out_dir / "results.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"wrote {out_dir / 'results.json'}")
    print(f"wrote {out_dir / 'results.md'}")


def _render_md(
    checkpoint: str,
    overall: dict,
    by_fraction: dict,
    validity_rate: float,
    n: int,
) -> str:
    lines = [
        "# LSTM - sequence completion (free running)",
        "",
        f"Checkpoint: `{checkpoint}`",
        f"Examples: {n} - generated-route validity rate: {validity_rate:.4f}",
        "",
        "## Overall (by family)",
        "",
        "| Group | n | exact_match | norm_edit_dist | token_acc | block_acc |",
        "|---|---|---|---|---|---|",
    ]
    for group, row in overall.items():
        lines.append(
            f"| {group} | {row['n']} | {row['exact_match']:.4f} | "
            f"{row['norm_edit_distance']:.4f} | {row['token_accuracy']:.4f} | "
            f"{row['block_accuracy']:.4f} |"
        )
    for fraction, groups in by_fraction.items():
        allg = groups.get("all", {})
        lines += [
            "",
            f"## Prefix fraction {fraction}",
            "",
            "| Group | n | exact_match | norm_edit_dist | token_acc | block_acc |",
            "|---|---|---|---|---|---|",
            f"| all | {allg.get('n', 0)} | {allg.get('exact_match', 0):.4f} | "
            f"{allg.get('norm_edit_distance', 0):.4f} | {allg.get('token_accuracy', 0):.4f} | "
            f"{allg.get('block_accuracy', 0):.4f} |",
        ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
