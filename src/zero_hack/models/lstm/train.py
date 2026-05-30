import argparse
import json
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.models.common import (
    DEFAULT_METRICS_DIR,
    DEFAULT_SPLITS_DIR,
    DataBundle,
    TrainConfig,
    count_parameters,
    load_split_records,
    make_loaders,
    pick_device,
    train_model,
)
from zero_hack.models.lstm.inference import save_lstm_checkpoint
from zero_hack.models.lstm.model import LSTMConfig, LSTMModel
from zero_hack.models.neural_eval import write_neural_next_step_eval
from zero_hack.models.scheduled_sampling import (
    make_sequence_loader,
    train_model_scheduled_sampling,
)


def build_model(bundle: DataBundle, config: LSTMConfig) -> LSTMModel:
    vocab_size = len(bundle.vocabulary.id_to_token)
    return LSTMModel(
        vocab_size=vocab_size,
        config=config,
        pad_id=bundle.vocabulary.pad_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the small LSTM baseline.")
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
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--eval-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--preds-root", default=str(PROJECT_ROOT / "outputs" / "preds"))
    parser.add_argument("--metrics-root", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--method-name", default=None)
    parser.add_argument("--eval-views", nargs="+", default=["id", "ood"])
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
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Where to save the trained checkpoint (.pt). Skipped if not set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle = load_split_records(
        args.splits_dir,
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

    model = build_model(bundle, LSTMConfig())
    print(f"parameters: {count_parameters(model)}")

    device = pick_device(args.device)
    dataset = args.dataset or Path(args.splits_dir).parent.name
    method_name = args.method_name or (
        "lstm_scheduled_sampling" if args.scheduled_sampling else "lstm_teacher_forcing"
    )
    history_path = Path(args.report_dir) / "history.json"
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
        model = train_model_scheduled_sampling(
            model,
            seq_loader,
            bundle.vocabulary,
            config=train_config,
            device=device,
            eval_loader=loaders.get("valid"),
            max_context=args.max_context,
            history_path=history_path,
        )
    else:
        model = train_model(
            model,
            loaders,
            config=train_config,
            device=device,
            pad_id=bundle.vocabulary.pad_id,
            history_path=history_path,
        )

    if args.checkpoint_path:
        meta = {
            "scheduled_sampling": args.scheduled_sampling,
            "ss_max_prob": args.ss_max_prob,
            "ss_schedule": args.ss_schedule,
            "family_dropout": args.family_dropout,
            "epochs": args.epochs,
            "holdout_family": args.holdout_family,
            "train_families": list(bundle.train_families),
        }
        saved = save_lstm_checkpoint(
            args.checkpoint_path,
            model,
            bundle.vocabulary,
            max_context=args.max_context,
            meta=meta,
        )
        print(f"wrote checkpoint {saved}")

    if args.holdout_family is None:
        print("skip fixed eval: pass --holdout-family to select data/eval holdout views")
        return

    results = write_neural_next_step_eval(
        model,
        bundle.vocabulary,
        method_name=method_name,
        dataset=dataset,
        holdout_family=args.holdout_family,
        eval_root=args.eval_root,
        preds_root=args.preds_root,
        metrics_root=args.metrics_root,
        device=device,
        k=args.k,
        max_context=args.max_context,
        views=tuple(args.eval_views),
    )
    out_dir = Path(args.metrics_root) / dataset / f"holdout_{args.holdout_family}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{method_name}_next_step_summary.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
