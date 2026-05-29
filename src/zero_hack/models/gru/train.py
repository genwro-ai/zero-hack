"""Thin CLI to train/evaluate the small GRU baseline.

All data loading, splitting, training, and evaluation is delegated to
``zero_hack.models.common``; this module only wires the architecture in.
"""

from __future__ import annotations

import argparse

from zero_hack.models.common import (
    DEFAULT_RAW_DIR,
    DataBundle,
    TrainConfig,
    count_parameters,
    evaluate_model,
    load_record_splits,
    make_loaders,
    pick_device,
    train_model,
)
from zero_hack.models.gru.model import GRUConfig, GRUModel


def build_model(bundle: DataBundle, config: GRUConfig) -> GRUModel:
    """Construct a :class:`GRUModel` sized for ``bundle``'s vocabulary."""
    vocab = bundle.vocabulary
    return GRUModel(
        vocab_size=len(vocab.id_to_token),
        config=config,
        pad_id=vocab.pad_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the small GRU baseline.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=192)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_record_splits(
        raw_dir=args.raw_dir,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")

    loaders = make_loaders(
        bundle,
        batch_size=args.batch_size,
        max_context=args.max_context,
    )

    config = GRUConfig()
    model = build_model(bundle, config)
    print(f"parameters: {count_parameters(model)}")

    device = pick_device(args.device)
    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        k=args.k,
    )
    train_model(
        model,
        loaders,
        config=train_config,
        device=device,
        pad_id=bundle.vocabulary.pad_id,
    )

    summary = evaluate_model(
        model,
        loaders["test"],
        device=device,
        k=args.k,
        max_batches=args.max_eval_batches,
    )
    print(f"test summary: {summary}")


if __name__ == "__main__":
    main()
