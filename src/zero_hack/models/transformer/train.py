"""Train and evaluate the Transformer next-step baseline."""

from __future__ import annotations

import argparse

from zero_hack.models.common import (
    DEFAULT_METRICS_DIR,
    DEFAULT_SPLITS_DIR,
    DataBundle,
    TrainConfig,
    count_parameters,
    evaluate_and_report,
    load_split_records,
    make_loaders,
    pick_device,
    train_model,
)
from zero_hack.models.scheduled_sampling import (
    make_sequence_loader,
    train_model_scheduled_sampling,
)
from zero_hack.models.transformer.model import TransformerConfig, TransformerModel


def build_model(bundle: DataBundle, config: TransformerConfig) -> TransformerModel:
    vocab = bundle.vocabulary
    return TransformerModel(
        vocab_size=len(vocab.id_to_token),
        config=config,
        pad_id=vocab.pad_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the small Transformer baseline.")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=192)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--report-dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument(
        "--family-dropout",
        type=float,
        default=0.0,
        help=(
            "Probability of replacing the family conditioning token with "
            "<FAMILY_UNKNOWN> on the train split, so the model learns a "
            "family-agnostic mode for OOD/unknown-family eval. 0.0 disables it."
        ),
    )
    parser.add_argument(
        "--scheduled-sampling",
        action="store_true",
        help="Train with scheduled sampling (free-running rollout) instead of teacher forcing.",
    )
    parser.add_argument(
        "--ss-max-prob",
        type=float,
        default=0.25,
        help="Max probability of feeding the model's own prediction (reached at the final epoch).",
    )
    parser.add_argument(
        "--ss-schedule",
        choices=("linear", "constant"),
        default="linear",
        help="How the scheduled-sampling probability ramps across epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_split_records(
        splits_dir=args.splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")

    loaders = make_loaders(
        bundle,
        batch_size=args.batch_size,
        max_context=args.max_context,
        family_dropout=args.family_dropout,
    )

    config = TransformerConfig(max_context=args.max_context)
    model = build_model(bundle, config)
    print(f"parameters: {count_parameters(model)}")

    device = pick_device(args.device)
    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        k=args.k,
        scheduled_sampling=args.scheduled_sampling,
        ss_max_prob=args.ss_max_prob,
        ss_schedule=args.ss_schedule,
    )
    if args.scheduled_sampling:
        print(f"scheduled sampling: max_prob={args.ss_max_prob} schedule={args.ss_schedule}")
        seq_loader = make_sequence_loader(
            bundle,
            "train",
            batch_size=args.batch_size,
            max_context=args.max_context,
            family_dropout=args.family_dropout,
        )
        train_model_scheduled_sampling(
            model,
            seq_loader,
            bundle.vocabulary,
            config=train_config,
            device=device,
            eval_loader=loaders.get("valid"),
            max_context=args.max_context,
        )
    else:
        train_model(
            model,
            loaders,
            config=train_config,
            device=device,
            pad_id=bundle.vocabulary.pad_id,
        )

    evaluate_and_report(
        model,
        loaders,
        bundle,
        model_name="transformer",
        device=device,
        k=args.k,
        max_eval_batches=args.max_eval_batches,
        report_dir=args.report_dir,
    )


if __name__ == "__main__":
    main()
