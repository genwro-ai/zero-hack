"""Train a small GPT-style decoder on one generated dataset and holdout split."""

from __future__ import annotations

import argparse
import json
import math
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

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
from zero_hack.models.gpt.model import GPTConfig, GPTNextStepModel

MODEL_NAME = "gpt_decoder"
CAUSAL_IGNORE = -100


def build_model(bundle: DataBundle, config: GPTConfig) -> GPTNextStepModel:
    return GPTNextStepModel(
        vocab_size=len(bundle.vocabulary.id_to_token),
        config=config,
        pad_id=bundle.vocabulary.pad_id,
    )


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
def predict_topk(
    model: GPTNextStepModel,
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str] | tuple[str, ...],
    *,
    k: int,
    device: torch.device,
    invalid_ids: list[int],
) -> list[str]:
    model.eval()
    input_ids, attention_mask = _encode_prefix(
        vocabulary,
        family,
        prefix,
        max_context=model.config.max_context,
        device=device,
    )
    logits = model(input_ids, attention_mask).squeeze(0)
    if invalid_ids:
        logits[torch.tensor(invalid_ids, device=device)] = -torch.inf
    top_ids = torch.topk(logits, k=min(k, logits.numel())).indices.tolist()
    return [vocabulary.id_to_token[token_id] for token_id in top_ids]


class SequenceDataset(Dataset):
    def __init__(self, records, vocabulary, max_len):
        self.rows = []
        for record in records:
            tokens = ["<BOS>", FAMILY_TOKENS[record.family], *record.steps, "<EOS>"]
            ids = vocabulary.encode(tokens)[:max_len]
            if len(ids) >= 3:
                self.rows.append(ids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_sequences(batch, pad_id):
    width = max(len(ids) for ids in batch) - 1
    inputs, labels, mask = [], [], []
    for ids in batch:
        prefix = ids[:-1]
        pad = width - len(prefix)
        inputs.append(prefix + [pad_id] * pad)
        labels.append([CAUSAL_IGNORE] + ids[2:] + [CAUSAL_IGNORE] * pad)
        mask.append([1] * len(prefix) + [0] * pad)
    return (
        torch.tensor(inputs),
        torch.tensor(labels),
        torch.tensor(mask, dtype=torch.bool),
    )


def _causal_loss(model, loader, device):
    model.eval()
    use_cuda = device.type == "cuda"
    total, seen = 0.0, 0
    with torch.no_grad():
        for inputs, labels, mask in loader:
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                logits = model.sequence_logits(inputs, mask)
                loss = F.cross_entropy(
                    logits.flatten(0, 1), labels.flatten(), ignore_index=CAUSAL_IGNORE
                )
            total += float(loss.item())
            seen += 1
    return total / max(1, seen)


def fit_causal_lm(
    model, train_records, valid_records, vocabulary, device, epochs, batch_size, lr, patience,
    num_workers=0, phase_loss=None,
):
    model = model.to(device)
    train_data = SequenceDataset(train_records, vocabulary, model.config.max_context)
    if len(train_data) == 0:
        return model
    use_cuda = device.type == "cuda"
    collate = partial(collate_sequences, pad_id=vocabulary.pad_id)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    valid_data = SequenceDataset(valid_records, vocabulary, model.config.max_context)
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    total = max(1, len(train_loader) * epochs)
    warmup = max(1, int(0.05 * total))
    scheduler = LambdaLR(optimizer, lambda s: _lr_lambda(s, warmup_steps=warmup, total_steps=total))
    best_loss, best_state, stale = math.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for inputs, labels, mask in train_loader:
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                logits = model.sequence_logits(inputs, mask)
                flat_logits, flat_labels = logits.flatten(0, 1), labels.flatten()
                loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=CAUSAL_IGNORE)
                if phase_loss is not None:
                    loss = loss + phase_loss(flat_logits, flat_labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
        train_loss = running / max(1, len(train_loader))
        valid_loss = _causal_loss(model, valid_loader, device) if len(valid_data) else train_loss
        print(
            f"  gpt epoch {epoch}/{epochs} train={train_loss:.4f} valid={valid_loss:.4f}",
            flush=True,
        )
        if valid_loss < best_loss - 1e-3:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch} best_valid={best_loss:.4f}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class CausalLMAdapter:
    def __init__(self, model, vocabulary, device):
        self.model = model.to(device).eval()
        self.vocabulary = vocabulary
        self.device = device
        self.invalid_ids = _invalid_prediction_ids(vocabulary)

    @torch.no_grad()
    def predict_topk(self, family, prefix_steps, k=3):
        input_ids, mask = _encode_prefix(
            self.vocabulary,
            family,
            list(prefix_steps),
            max_context=self.model.config.max_context,
            device=self.device,
        )
        logits = self.model(input_ids, mask).squeeze(0)
        if self.invalid_ids:
            logits[torch.tensor(self.invalid_ids, device=self.device)] = -torch.inf
        top = torch.topk(logits, min(k, logits.numel())).indices.tolist()
        return [self.vocabulary.id_to_token[i] for i in top]

    @torch.no_grad()
    def score_sequence(self, family, steps):
        steps = list(steps)
        if not steps:
            return 0.0
        input_ids, mask = _encode_prefix(
            self.vocabulary,
            family,
            steps,
            max_context=self.model.config.max_context,
            device=self.device,
        )
        log_probs = F.log_softmax(self.model.sequence_logits(input_ids, mask)[0, :-1], dim=-1)
        chosen = log_probs.gather(1, input_ids[0, 1:, None]).squeeze(1)
        return float(chosen[1:].sum())


def _lr_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


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


def _make_gpt_loaders(
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


def _run_epoch(
    model: nn.Module,
    loader,
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


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    bundle: DataBundle,
    config: GPTConfig,
    args: argparse.Namespace,
    epoch: int,
    valid_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
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


def _write_eval_set_next_step(
    model: GPTNextStepModel,
    bundle: DataBundle,
    *,
    dataset: str,
    holdout_family: str,
    eval_root: Path,
    preds_root: Path,
    metrics_root: Path,
    device: torch.device,
    k: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    invalid_ids = _invalid_prediction_ids(bundle.vocabulary)
    for view in ("id", "ood"):
        eval_dir = eval_root / dataset / f"holdout_{holdout_family}" / view
        if not eval_dir.exists():
            print(f"skip eval-set {view}: missing {eval_dir}")
            continue

        inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
        rows = [
            {
                "example_id": row["example_id"],
                "ranks": predict_topk(
                    model,
                    bundle.vocabulary,
                    row["family"],
                    row["partial_sequence"],
                    k=k,
                    device=device,
                    invalid_ids=invalid_ids,
                ),
            }
            for row in inputs
        ]

        pred_dir = preds_root / dataset / f"holdout_{holdout_family}" / view / MODEL_NAME
        pred_path = pred_dir / "nextstep.csv"
        io.write_next_step_predictions(pred_path, rows)

        metrics = score_task(
            "next_step",
            ground_truth=eval_dir / "nextstep_truth.csv",
            predictions=pred_path,
            eval_input=eval_dir / "eval_input_valid.csv",
        )
        metrics_dir = metrics_root / dataset / f"holdout_{holdout_family}" / view / MODEL_NAME
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "next_step.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        results[view] = metrics
        print(f"eval-set {view}: {metrics.get('all', metrics)}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-context", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_root = Path(args.generated_root)
    splits_dir = (
        Path(args.splits_dir) if args.splits_dir else generated_root / args.dataset / "splits"
    )
    run_dir = Path(args.model_root) / args.dataset / f"{MODEL_NAME}_holdout_{args.holdout_family}"
    checkpoint_path = run_dir / "best.pt"

    bundle = load_split_records(
        splits_dir=splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    print(f"dataset={args.dataset} holdout={args.holdout_family} counts={bundle.counts()}")

    loaders = _make_gpt_loaders(
        bundle,
        batch_size=args.batch_size,
        max_context=args.max_context,
    )
    config = GPTConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_context=args.max_context,
    )
    model = build_model(bundle, config)
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
            _save_checkpoint(
                checkpoint_path,
                model=model,
                bundle=bundle,
                config=config,
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
        model_name=f"{MODEL_NAME}_holdout_{args.holdout_family}",
        device=device,
        k=args.k,
        max_eval_batches=args.max_eval_batches,
        report_dir=report_dir,
    )
    eval_set_results = _write_eval_set_next_step(
        model,
        bundle,
        dataset=args.dataset,
        holdout_family=args.holdout_family,
        eval_root=Path(args.eval_root),
        preds_root=Path(args.preds_root),
        metrics_root=Path(args.metrics_root),
        device=device,
        k=args.k,
    )
    (run_dir / "eval_set_next_step.json").write_text(
        json.dumps(eval_set_results, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
