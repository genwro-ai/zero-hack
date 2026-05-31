"""Generate unseen process sequences for training.

Produces rule-valid sequences that reach beyond the three reference families:
every documented vocabulary token is exercised, family labels are decoupled from
content (synthetic families + UNK dropout), and validity is baked in by a
precondition-aware planner. Every sequence is still checked against the official
10-rule validator as a backstop, and the run is audited for full vocabulary
coverage. See
``docs/superpowers/specs/2026-05-30-neurosymbolic-sequence-generator-design.md``.

Usage:
    uv run python scripts/generate_unseen_data.py --count 10000
    uv run python scripts/generate_unseen_data.py --count 2000 --dataset unseen_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.data.synth import CANONICAL_VOCAB, generate_dataset, write_dataset_csv

DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "generated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10_000, help="Number of sequences.")
    parser.add_argument("--dataset", default="unseen", help="Dataset name (sub-directory).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-families", type=int, default=12)
    parser.add_argument("--unk-prob", type=float, default=0.25)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_dataset(
        args.count,
        seed=args.seed,
        synthetic_n=args.synthetic_families,
        unk_prob=args.unk_prob,
    )

    emitted = {step for rec in dataset for step in rec.steps}
    missing = CANONICAL_VOCAB - emitted
    lengths = [len(rec.steps) for rec in dataset]
    labels = {rec.family_label for rec in dataset}

    out_dir = Path(args.out_root) / args.dataset
    write_dataset_csv(out_dir / "raw.csv", dataset)

    covered = len(CANONICAL_VOCAB) - len(missing)
    total = len(CANONICAL_VOCAB)
    lo, mean, hi = min(lengths), sum(lengths) // len(lengths), max(lengths)

    print(f"wrote {len(dataset)} sequences -> {out_dir / 'raw.csv'}")
    print(f"  documented-vocab coverage : {covered}/{total}")
    if missing:
        print(f"  MISSING tokens            : {sorted(missing)}")
    print(f"  length min/mean/max       : {lo}/{mean}/{hi}")
    print(f"  distinct family labels    : {len(labels)}")


if __name__ == "__main__":
    main()
