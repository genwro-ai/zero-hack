"""DPO fine-tune a GPT decoder from JSONL process-sequence preference pairs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from zero_hack import PROJECT_ROOT
from zero_hack.data import FAMILY_TOKENS, Vocabulary
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold
from zero_hack.models.common import DataBundle, count_parameters, load_split_records, pick_device
from zero_hack.models.gpt.model import GPTConfig, GPTNextStepModel
from zero_hack.models.gpt.train import (
    _default_views,
    _evaluate_eval_sets,
    _GPTLikelihoodAdapter,
)


def _lr_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _load_checkpoint(
    path: str | Path, device: torch.device
) -> tuple[GPTNextStepModel, Vocabulary, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    vocabulary = Vocabulary(
        token_to_id=checkpoint["vocabulary"]["token_to_id"],
        id_to_token=tuple(checkpoint["vocabulary"]["id_to_token"]),
    )
    config = GPTConfig(**checkpoint["model_config"])
    model = GPTNextStepModel(
        len(vocabulary.id_to_token),
        config,
        pad_id=vocabulary.pad_id,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model, vocabulary, checkpoint


class DPOPairDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        vocabulary: Vocabulary,
        *,
        max_context: int,
        family_dropout: float = 0.0,
        step_dropout: float = 0.0,
        seed: int = 1729,
        limit: int | None = None,
    ) -> None:
        self.vocabulary = vocabulary
        self.max_context = max_context
        self.family_dropout = family_dropout
        self.step_dropout = step_dropout
        self.rng = random.Random(seed)
        rows = []
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit is not None and len(rows) >= limit:
                        break
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, row: dict[str, Any], continuation_key: str) -> dict[str, list[int]]:
        family = row["family"].lower()
        family_token = FAMILY_TOKENS.get(family, FAMILY_TOKENS["unknown"])
        if self.family_dropout and self.rng.random() < self.family_dropout:
            family_token = FAMILY_TOKENS["unknown"]

        prefix = list(row["prefix"])
        if self.step_dropout:
            prefix = [
                "<UNK_STEP>" if self.rng.random() < self.step_dropout else step for step in prefix
            ]
        continuation = list(row[continuation_key])
        tokens = ["<BOS>", family_token, *prefix, *continuation]
        prompt_len = 2 + len(prefix)
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        loss_mask = [0] * len(target_tokens)
        for idx in range(prompt_len - 1, len(target_tokens)):
            loss_mask[idx] = 1

        input_ids = self.vocabulary.encode(input_tokens)
        target_ids = self.vocabulary.encode(target_tokens)
        if len(input_ids) > self.max_context:
            input_ids = input_ids[-self.max_context :]
            target_ids = target_ids[-self.max_context :]
            loss_mask = loss_mask[-self.max_context :]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "chosen": self._encode(row, "chosen"),
            "rejected": self._encode(row, "rejected"),
            "negative_type": row.get("negative_type", "unknown"),
        }


def _pad_items(items: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in items)
    input_ids = []
    attention_mask = []
    target_ids = []
    loss_mask = []
    for item in items:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
        target_ids.append(item["target_ids"] + [pad_id] * pad_len)
        loss_mask.append(item["loss_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "target_ids": torch.tensor(target_ids, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
    }


def _collate(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    return {
        "chosen": _pad_items([item["chosen"] for item in batch], pad_id),
        "rejected": _pad_items([item["rejected"] for item in batch], pad_id),
        "negative_type": [item["negative_type"] for item in batch],
    }


def _sequence_logprobs(
    model: GPTNextStepModel,
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_ids = batch["target_ids"].to(device)
    loss_mask = batch["loss_mask"].to(device)

    logits = model.forward_all(input_ids, attention_mask)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * loss_mask
    lengths = loss_mask.sum(dim=-1).clamp_min(1.0)
    return token_log_probs.sum(dim=-1), lengths


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    vocabulary: Vocabulary,
    config: GPTConfig,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": config.to_dict(),
            "vocabulary": {
                "token_to_id": vocabulary.token_to_id,
                "id_to_token": vocabulary.id_to_token,
            },
            "args": vars(args),
            "history": history,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--dataset", default="valid_s100k_unseen_s050k")
    parser.add_argument("--splits-dir", required=True)
    parser.add_argument("--eval-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--preds-root", default=str(PROJECT_ROOT / "outputs" / "preds"))
    parser.add_argument("--metrics-root", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    parser.add_argument("--model-root", default=str(PROJECT_ROOT / "outputs" / "models"))
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), required=True)
    parser.add_argument("--method-name", default="gpt_dpo")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("next_step", "completion", "anomaly"),
        default=["next_step", "completion", "anomaly"],
    )
    parser.add_argument("--eval-views", nargs="+", default=None)
    parser.add_argument(
        "--eval-family-mode",
        choices=("as_given", "holdout_unknown", "all_unknown"),
        default="holdout_unknown",
    )

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft-weight", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--family-dropout", type=float, default=0.15)
    parser.add_argument("--step-dropout", type=float, default=0.05)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--pair-limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-completion-steps", type=int, default=240)
    parser.add_argument("--anomaly-val-valid", type=int, default=200)
    parser.add_argument("--anomaly-val-invalid", type=int, default=129)
    parser.add_argument("--anomaly-val-seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    policy, vocabulary, _checkpoint = _load_checkpoint(args.checkpoint, device)
    reference, _ref_vocab, _ = _load_checkpoint(args.checkpoint, device)
    policy.to(device)
    reference.to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    dataset = DPOPairDataset(
        args.pairs,
        vocabulary,
        max_context=policy.config.max_context,
        family_dropout=args.family_dropout,
        step_dropout=args.step_dropout,
        seed=args.seed,
        limit=args.pair_limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, vocabulary.pad_id),
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    train_batches = len(loader)
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

    run_name = f"{args.method_name}_holdout_{args.holdout_family}"
    run_dir = Path(args.model_root) / args.dataset / run_name
    checkpoint_path = run_dir / "best.pt"
    print(
        f"dpo pairs={len(dataset)} checkpoint={args.checkpoint} "
        f"parameters={count_parameters(policy)}"
    )

    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        running: dict[str, float] = {}
        seen = 0
        for step, batch in enumerate(loader):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

            pi_chosen, chosen_lengths = _sequence_logprobs(policy, batch["chosen"], device=device)
            pi_rejected, _ = _sequence_logprobs(policy, batch["rejected"], device=device)
            with torch.no_grad():
                ref_chosen, _ = _sequence_logprobs(reference, batch["chosen"], device=device)
                ref_rejected, _ = _sequence_logprobs(reference, batch["rejected"], device=device)

            pi_logratio = pi_chosen - pi_rejected
            ref_logratio = ref_chosen - ref_rejected
            dpo_loss = -F.logsigmoid(args.beta * (pi_logratio - ref_logratio)).mean()
            chosen_nll = -(pi_chosen / chosen_lengths).mean()
            loss = dpo_loss + args.sft_weight * chosen_nll

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            metrics = {
                "loss": float(loss.item()),
                "dpo_loss": float(dpo_loss.item()),
                "chosen_nll": float(chosen_nll.item()),
                "margin": float((pi_logratio - ref_logratio).mean().item()),
                "chosen_logprob": float(pi_chosen.mean().item()),
                "rejected_logprob": float(pi_rejected.mean().item()),
            }
            for key, value in metrics.items():
                running[key] = running.get(key, 0.0) + value
            seen += 1

            if args.log_every and (step + 1) % args.log_every == 0:
                avg = {key: value / seen for key, value in running.items()}
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} step={step + 1} loss={avg['loss']:.4f} "
                    f"dpo={avg['dpo_loss']:.4f} nll={avg['chosen_nll']:.4f} "
                    f"margin={avg['margin']:.4f} lr={lr:.2e}"
                )

        avg = {key: round(value / max(1, seen), 6) for key, value in running.items()}
        row = {"epoch": epoch, **avg}
        history.append(row)
        print(f"epoch={epoch} {json.dumps(avg, sort_keys=True)}")
        _save_checkpoint(
            checkpoint_path,
            model=policy,
            vocabulary=vocabulary,
            config=policy.config,
            args=args,
            history=history,
        )
        print(f"saved checkpoint: {checkpoint_path}")

    bundle = load_split_records(
        args.splits_dir,
        holdout_family=args.holdout_family,
    )
    eval_bundle = DataBundle(
        vocabulary=vocabulary,
        records=bundle.records,
        train_families=bundle.train_families,
        holdout_family=bundle.holdout_family,
    )

    anomaly_threshold = -math.inf
    if "anomaly" in args.tasks:
        adapter = _GPTLikelihoodAdapter(policy, vocabulary, device=device)
        threshold = tune_anomaly_threshold(
            adapter,
            eval_bundle.records["valid"],
            n_valid=args.anomaly_val_valid,
            n_invalid=args.anomaly_val_invalid,
            seed=args.anomaly_val_seed,
        )
        anomaly_threshold = threshold.threshold
        (run_dir / "anomaly_threshold.json").write_text(
            json.dumps(
                {
                    "source": "auto",
                    "objective": "f1",
                    "threshold": threshold.threshold,
                    "val_f1": threshold.f1,
                    "val_precision": threshold.precision,
                    "val_recall": threshold.recall,
                    "seed": args.anomaly_val_seed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    views = args.eval_views or _default_views(
        Path(args.eval_root),
        args.dataset,
        args.holdout_family,
    )
    eval_results = _evaluate_eval_sets(
        policy,
        eval_bundle,
        method_name=args.method_name,
        dataset=args.dataset,
        holdout_family=args.holdout_family,
        eval_root=Path(args.eval_root),
        preds_root=Path(args.preds_root),
        metrics_root=Path(args.metrics_root),
        views=views,
        tasks=tuple(args.tasks),
        device=device,
        k=args.k,
        max_completion_steps=args.max_completion_steps,
        anomaly_threshold=anomaly_threshold,
        eval_family_mode=args.eval_family_mode,
    )
    (run_dir / "eval_set_metrics.json").write_text(
        json.dumps(eval_results, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
