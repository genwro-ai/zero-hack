import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from zero_hack import PROJECT_ROOT
from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, NextStepDataset, Vocabulary
from zero_hack.eval import io
from zero_hack.eval.score import score_task
from zero_hack.models.common import (
    DataBundle,
    count_parameters,
    evaluate_and_report,
    evaluate_model,
    load_split_records,
    pick_device,
)
from zero_hack.models.gpt import GPTConfig, GPTNextStepModel
from zero_hack.models.transformer.model import TransformerConfig, TransformerModel

ARCHITECTURES = ("transformer", "gpt")


def _collate_right_padded(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    for item in batch:
        ids = item["input_ids"]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "target_id": torch.tensor([item["target_id"] for item in batch], dtype=torch.long),
        "family": [item["family"] for item in batch],
        "sequence_id": [item["sequence_id"] for item in batch],
        "position": torch.tensor([item["position"] for item in batch], dtype=torch.long),
    }


def _make_loaders(
    bundle: DataBundle,
    *,
    batch_size: int,
    max_context: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for split, records in bundle.records.items():
        dataset = NextStepDataset(
            records=records,
            vocabulary=bundle.vocabulary,
            max_context=max_context,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=lambda batch, pad_id=bundle.vocabulary.pad_id: _collate_right_padded(
                batch, pad_id
            ),
        )
    return loaders


def _build_model(
    architecture: str,
    bundle: DataBundle,
    args: argparse.Namespace,
) -> tuple[nn.Module, Any]:
    vocab_size = len(bundle.vocabulary.id_to_token)
    if architecture == "gpt":
        config = GPTConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers or 3,
            dim_feedforward=args.dim_feedforward or 512,
            dropout=args.dropout,
            max_context=args.max_context,
        )
        return GPTNextStepModel(vocab_size, config, pad_id=bundle.vocabulary.pad_id), config

    if architecture == "transformer":
        config = TransformerConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers or 2,
            dim_feedforward=args.dim_feedforward or 256,
            dropout=args.dropout,
            max_context=args.max_context,
        )
        return TransformerModel(vocab_size, config, pad_id=bundle.vocabulary.pad_id), config

    raise ValueError(f"Unknown architecture: {architecture}")


def _lr_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
    log_every: int = 100,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    seen = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target_id"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += float(loss.item())
        seen += 1
        if training and log_every and (step + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"step {step + 1} train_loss={total_loss / seen:.4f} lr={lr:.2e}")

    return total_loss / max(1, seen)


def _invalid_prediction_ids(vocabulary: Vocabulary) -> list[int]:
    invalid = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
    return [vocabulary.token_to_id[token] for token in invalid if token in vocabulary.token_to_id]


def _encode_prefix(
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str] | tuple[str, ...],
    *,
    max_context: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    family_token = FAMILY_TOKENS.get(family.lower(), FAMILY_TOKENS["unknown"])
    tokens = ["<BOS>", family_token, *prefix][-max_context:]
    input_ids = torch.tensor([vocabulary.encode(tokens)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    return input_ids, attention_mask


@torch.no_grad()
def _predict_topk(
    model: nn.Module,
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str] | tuple[str, ...],
    *,
    k: int,
    max_context: int,
    device: torch.device,
    invalid_ids: list[int],
) -> list[str]:
    model.eval()
    input_ids, attention_mask = _encode_prefix(
        vocabulary,
        family,
        prefix,
        max_context=max_context,
        device=device,
    )
    logits = model(input_ids, attention_mask).squeeze(0)
    if invalid_ids:
        logits[torch.tensor(invalid_ids, device=device)] = -torch.inf
    top_ids = torch.topk(logits, k=min(k, logits.numel())).indices.tolist()
    return [vocabulary.id_to_token[token_id] for token_id in top_ids]


def _write_eval_set_next_step(
    model: nn.Module,
    bundle: DataBundle,
    *,
    architecture: str,
    dataset: str,
    holdout_family: str,
    eval_root: Path,
    preds_root: Path,
    metrics_root: Path,
    device: torch.device,
    k: int,
    max_context: int,
    eval_views: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    invalid_ids = _invalid_prediction_ids(bundle.vocabulary)
    for view in eval_views:
        eval_dir = eval_root / dataset / f"holdout_{holdout_family}" / view
        if not eval_dir.exists():
            print(f"skip eval-set {view}: missing {eval_dir}")
            continue

        inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
        rows = [
            {
                "example_id": row["example_id"],
                "ranks": _predict_topk(
                    model,
                    bundle.vocabulary,
                    row["family"],
                    row["partial_sequence"],
                    k=k,
                    max_context=max_context,
                    device=device,
                    invalid_ids=invalid_ids,
                ),
            }
            for row in inputs
        ]

        pred_dir = preds_root / dataset / f"holdout_{holdout_family}" / view / architecture
        pred_path = pred_dir / "nextstep.csv"
        io.write_next_step_predictions(pred_path, rows)

        metrics = score_task(
            "next_step",
            ground_truth=eval_dir / "nextstep_truth.csv",
            predictions=pred_path,
            eval_input=eval_dir / "eval_input_valid.csv",
        )
        metrics_dir = metrics_root / dataset / f"holdout_{holdout_family}" / view / architecture
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "next_step.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        results[view] = metrics
        print(f"eval-set {view}: {metrics.get('all', metrics)}")

    return results


def _save_best(
    path: Path,
    *,
    model: nn.Module,
    bundle: DataBundle,
    architecture: str,
    config: Any,
    args: argparse.Namespace,
    epoch: int,
    valid_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": architecture,
            "model_config": asdict(config),
            "vocabulary": {
                "token_to_id": bundle.vocabulary.token_to_id,
                "id_to_token": bundle.vocabulary.id_to_token,
            },
            "args": vars(args),
            "epoch": epoch,
            "valid_loss": valid_loss,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=ARCHITECTURES, required=True)
    parser.add_argument("--dataset", default="valid_s005k")
    parser.add_argument("--generated-root", default=str(PROJECT_ROOT / "data" / "generated"))
    parser.add_argument("--eval-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--preds-root", default=str(PROJECT_ROOT / "outputs" / "preds"))
    parser.add_argument("--metrics-root", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    parser.add_argument("--model-root", default=str(PROJECT_ROOT / "outputs" / "models"))
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default="ic")
    parser.add_argument("--limit-per-family", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--eval-views",
        nargs="+",
        default=None,
        help="Eval views. Defaults to standard/diverse views when present, else id/ood.",
    )

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dim-feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-context", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_root = Path(args.generated_root)
    splits_dir = (
        Path(args.splits_dir) if args.splits_dir else generated_root / args.dataset / "splits"
    )
    run_name = f"{args.model}_holdout_{args.holdout_family}"
    run_dir = Path(args.model_root) / args.dataset / run_name
    checkpoint_path = run_dir / "best.pt"

    bundle = load_split_records(
        splits_dir=splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    print(f"dataset={args.dataset} model={args.model} holdout={args.holdout_family}")
    print(f"counts={bundle.counts()}")

    loaders = _make_loaders(
        bundle,
        batch_size=args.batch_size,
        max_context=args.max_context,
    )
    model, model_config = _build_model(args.model, bundle, args)
    print(f"config={asdict(model_config)}")
    print(f"parameters={count_parameters(model)}")

    device = pick_device(args.device)
    model.to(device)
    criterion = nn.CrossEntropyLoss(
        ignore_index=bundle.vocabulary.pad_id,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    train_batches = len(loaders["train"])
    if args.max_train_batches is not None:
        train_batches = min(train_batches, args.max_train_batches)
    total_steps = max(1, train_batches * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )

    history = []
    best_loss = math.inf
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(
            model,
            loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            max_batches=args.max_train_batches,
            grad_clip=args.grad_clip,
        )
        valid_loss = _run_epoch(
            model,
            loaders["valid"],
            criterion=criterion,
            device=device,
            max_batches=args.max_eval_batches,
        )
        valid_topk = evaluate_model(
            model,
            loaders["valid"],
            device=device,
            k=args.k,
            max_batches=args.max_eval_batches,
        )["all"]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "valid_loss": round(valid_loss, 6),
            "valid_topk": valid_topk,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"valid_top1={valid_topk['top1']:.4f} "
            f"valid_top{args.k}={valid_topk[f'top{args.k}']:.4f}"
        )

        if valid_loss < best_loss - args.min_delta:
            best_loss = valid_loss
            stale_epochs = 0
            _save_best(
                checkpoint_path,
                model=model,
                bundle=bundle,
                architecture=args.model,
                config=model_config,
                args=args,
                epoch=epoch,
                valid_loss=valid_loss,
            )
            print(f"saved best checkpoint: {checkpoint_path}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early stopping at epoch {epoch}; best_valid_loss={best_loss:.4f}")
                break

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"loaded best epoch={checkpoint['epoch']} valid_loss={checkpoint['valid_loss']:.4f}")

    report_dir = Path(args.metrics_root) / args.dataset
    evaluate_and_report(
        model,
        loaders,
        bundle,
        model_name=run_name,
        device=device,
        k=args.k,
        max_eval_batches=args.max_eval_batches,
        report_dir=report_dir,
    )
    eval_set_results = _write_eval_set_next_step(
        model,
        bundle,
        architecture=args.model,
        dataset=args.dataset,
        holdout_family=args.holdout_family,
        eval_root=Path(args.eval_root),
        preds_root=Path(args.preds_root),
        metrics_root=Path(args.metrics_root),
        device=device,
        k=args.k,
        max_context=args.max_context,
        eval_views=tuple(
            args.eval_views
            or _default_eval_views(Path(args.eval_root), args.dataset, args.holdout_family)
        ),
    )
    (run_dir / "eval_set_next_step.json").write_text(
        json.dumps(eval_set_results, indent=2) + "\n",
        encoding="utf-8",
    )


def _default_eval_views(eval_root: Path, dataset: str, holdout_family: str) -> list[str]:
    base = eval_root / dataset / f"holdout_{holdout_family}"
    mixed_views = ["standard/id", "standard/ood", "diverse/id", "diverse/ood"]
    if all((base / view).exists() for view in mixed_views):
        return mixed_views
    return ["id", "ood"]


if __name__ == "__main__":
    main()
