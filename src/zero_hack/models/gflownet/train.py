from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from zero_hack import PROJECT_ROOT
from zero_hack.data import (
    FAMILY_TOKENS,
    SPECIAL_TOKENS,
    NextStepDataset,
    Vocabulary,
    build_vocabulary,
)
from zero_hack.eval import io
from zero_hack.eval.score import TASKS, score_task
from zero_hack.models.common import (
    count_parameters,
    evaluate_model,
    load_split_records,
    pick_device,
)
from zero_hack.models.gflownet.model import (
    TERMINATOR,
    GFlowNetConfig,
    GFlowNetPolicy,
    encode_prefix,
    invalid_action_ids,
    mask_logits,
    sample_completion,
)
from zero_hack.models.gflownet.reward import ProcessReward, RewardConfig

_VIEWS = ("id", "ood", "standard/id", "standard/ood", "diverse/id", "diverse/ood")


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


def _make_loader(
    records,
    vocabulary: Vocabulary,
    *,
    batch_size: int,
    max_context: int,
    shuffle: bool,
) -> DataLoader:
    dataset = NextStepDataset(records, vocabulary, max_context=max_context)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch, pad_id=vocabulary.pad_id: _collate_right_padded(batch, pad_id),
    )


def _lr_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _prefix_cut(record_steps: tuple[str, ...], rng: random.Random, args: argparse.Namespace) -> int:
    if len(record_steps) <= 1:
        return 0
    low = max(1, int(len(record_steps) * args.min_prefix_fraction))
    high = max(low, int(len(record_steps) * args.max_prefix_fraction))
    high = min(high, len(record_steps) - 1)
    return rng.randint(low, high)


def _sample_tb_loss(
    model: GFlowNetPolicy,
    reward_fn: ProcessReward,
    vocabulary: Vocabulary,
    *,
    family: str,
    prefix: list[str],
    device: torch.device,
    invalid_ids: list[int],
    min_length: int,
    max_length: int,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    steps = list(prefix)
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    while len(steps) < max_length:
        input_ids, attention_mask = encode_prefix(
            vocabulary,
            family,
            steps,
            max_context=model.config.max_context,
            device=device,
        )
        logits = model(input_ids, attention_mask).squeeze(0)
        masked = mask_logits(
            logits,
            vocabulary=vocabulary,
            current_length=len(steps),
            min_length=min_length,
            invalid_ids=invalid_ids,
        )
        dist = torch.distributions.Categorical(logits=masked / temperature)
        action_id = dist.sample()
        log_probs.append(torch.log_softmax(masked, dim=-1)[action_id])
        entropies.append(dist.entropy())
        action = vocabulary.id_to_token[int(action_id)]
        steps.append(action)
        if action == TERMINATOR:
            break

    breakdown = reward_fn.evaluate(family, prefix, steps)
    family_id = model.family_ids([family], device=device)
    if log_probs:
        forward_logprob = torch.stack(log_probs).sum()
    else:
        forward_logprob = torch.tensor(0.0, device=device)
    flow_residual = model.log_z(family_id).squeeze(0) + forward_logprob
    tb_loss = (flow_residual - breakdown.log_reward) ** 2
    entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0, device=device)
    return tb_loss, {
        "log_reward": breakdown.log_reward,
        "is_valid": breakdown.is_valid,
        "length": len(steps),
        "n_generated": len(steps) - len(prefix),
        "violations": list(breakdown.violations),
        "entropy": float(entropy.detach().cpu()),
        "entropy_tensor": entropy,
    }


def _run_gflownet_batch(
    model: GFlowNetPolicy,
    reward_fn: ProcessReward,
    train_records,
    vocabulary: Vocabulary,
    *,
    args: argparse.Namespace,
    device: torch.device,
    invalid_ids: list[int],
    rng: random.Random,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses = []
    entropies = []
    log_rewards = []
    valid = 0
    lengths = []
    generated = []
    candidates = [record for record in train_records if len(record.steps) > 1]
    for _ in range(args.gfn_batch_size):
        record = rng.choice(candidates)
        cut = _prefix_cut(record.steps, rng, args)
        loss, stats = _sample_tb_loss(
            model,
            reward_fn,
            vocabulary,
            family=record.family,
            prefix=list(record.steps[:cut]),
            device=device,
            invalid_ids=invalid_ids,
            min_length=args.min_length,
            max_length=args.max_length,
            temperature=args.temperature,
        )
        losses.append(loss)
        entropies.append(stats["entropy_tensor"])
        log_rewards.append(float(stats["log_reward"]))
        valid += int(stats["is_valid"])
        lengths.append(float(stats["length"]))
        generated.append(float(stats["n_generated"]))

    entropy = torch.stack(entropies).mean()
    gfn_loss = torch.stack(losses).mean() - args.entropy_weight * entropy
    return gfn_loss, {
        "gfn_log_reward": sum(log_rewards) / len(log_rewards),
        "gfn_valid_rate": valid / len(log_rewards),
        "gfn_length": sum(lengths) / len(lengths),
        "gfn_generated": sum(generated) / len(generated),
        "gfn_entropy": float(entropy.detach().cpu()),
    }


def _save_checkpoint(
    path: Path,
    *,
    model: GFlowNetPolicy,
    vocabulary: Vocabulary,
    model_config: GFlowNetConfig,
    reward_config: RewardConfig,
    args: argparse.Namespace,
    epoch: int,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "reward_config": asdict(reward_config),
            "vocabulary": {
                "token_to_id": vocabulary.token_to_id,
                "id_to_token": vocabulary.id_to_token,
            },
            "args": vars(args),
            "epoch": epoch,
            "history": history,
        },
        path,
    )


def _all_records(bundle) -> list:
    records = []
    seen: set[tuple[str, str]] = set()
    for split_records in bundle.records.values():
        for record in split_records:
            key = (record.family, record.sequence_id)
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def _known_process_steps(vocabulary: Vocabulary) -> set[str]:
    blocked = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
    return {token for token in vocabulary.id_to_token if token not in blocked}


def _truth_path(eval_dir: Path, task: str) -> Path:
    return (
        eval_dir
        / {
            "next_step": "nextstep_truth.csv",
            "completion": "completion_truth.csv",
            "anomaly": "anomaly_truth.csv",
        }[task]
    )


def _eval_input_path(eval_dir: Path, task: str) -> Path:
    if task == "anomaly":
        return eval_dir / "eval_input_anomaly.csv"
    return eval_dir / "eval_input_valid.csv"


def _pred_path(pred_dir: Path, task: str) -> Path:
    return (
        pred_dir
        / {
            "next_step": "nextstep.csv",
            "completion": "completion.csv",
            "anomaly": "anomaly.csv",
        }[task]
    )


def _default_views(eval_root: Path, dataset: str, holdout_family: str) -> list[str]:
    base = eval_root / dataset / f"holdout_{holdout_family}"
    mixed_views = ["standard/id", "standard/ood", "diverse/id", "diverse/ood"]
    if all((base / view).exists() for view in mixed_views):
        return mixed_views
    return ["id", "ood"]


@torch.no_grad()
def _predict_topk(
    model: GFlowNetPolicy,
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str] | tuple[str, ...],
    *,
    k: int,
    device: torch.device,
    invalid_ids: list[int],
    min_length: int,
) -> list[str]:
    if prefix and prefix[-1] == TERMINATOR:
        return []
    input_ids, attention_mask = encode_prefix(
        vocabulary,
        family,
        prefix,
        max_context=model.config.max_context,
        device=device,
    )
    logits = model(input_ids, attention_mask).squeeze(0)
    logits = mask_logits(
        logits,
        vocabulary=vocabulary,
        current_length=len(prefix),
        min_length=min_length,
        invalid_ids=invalid_ids,
    )
    top_ids = torch.topk(logits, k=min(k, logits.numel())).indices.tolist()
    return [vocabulary.id_to_token[token_id] for token_id in top_ids]


def _complete_with_best_reward(
    model: GFlowNetPolicy,
    reward_fn: ProcessReward,
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str],
    *,
    device: torch.device,
    invalid_ids: list[int],
    min_length: int,
    max_length: int,
    samples: int,
    temperature: float,
) -> list[str]:
    if prefix and prefix[-1] == TERMINATOR:
        return []

    best_steps: list[str] = []
    best_log_reward = -math.inf
    for _ in range(max(1, samples)):
        continuation = sample_completion(
            model,
            vocabulary,
            family,
            prefix,
            device=device,
            min_length=min_length,
            max_length=max_length,
            temperature=temperature,
            invalid_ids=invalid_ids,
        )
        full_sequence = list(prefix) + continuation
        reward = reward_fn.evaluate(family, prefix, full_sequence)
        if reward.log_reward > best_log_reward:
            best_log_reward = reward.log_reward
            best_steps = continuation
    return best_steps


def _predict_anomaly(
    reward_fn: ProcessReward,
    family: str,
    sequence: list[str],
) -> dict[str, Any]:
    reward = reward_fn.evaluate(family, [], sequence)
    score = 1.0 / (1.0 + math.exp(-reward.log_reward / 10.0))
    predicted_rule = None
    if not reward.is_valid and reward.violations:
        predicted_rule = reward.violations[0]
    return {
        "is_valid": int(reward.is_valid),
        "score": round(score, 6),
        "predicted_rule": predicted_rule,
    }


def _score_predictions(
    *,
    eval_dir: Path,
    pred_dir: Path,
    metrics_dir: Path,
    tasks: tuple[str, ...],
) -> dict[str, dict]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in tasks:
        metrics = score_task(
            task,
            ground_truth=_truth_path(eval_dir, task),
            predictions=_pred_path(pred_dir, task),
            eval_input=_eval_input_path(eval_dir, task),
        )
        results[task] = metrics
        (metrics_dir / f"{task}.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
    (metrics_dir / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def _write_eval_predictions(
    model: GFlowNetPolicy,
    reward_fn: ProcessReward,
    vocabulary: Vocabulary,
    *,
    eval_dir: Path,
    pred_dir: Path,
    tasks: tuple[str, ...],
    device: torch.device,
    invalid_ids: list[int],
    args: argparse.Namespace,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    if "next_step" in tasks or "completion" in tasks:
        valid_inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
        if "next_step" in tasks:
            rows = [
                {
                    "example_id": row["example_id"],
                    "ranks": _predict_topk(
                        model,
                        vocabulary,
                        row["family"],
                        row["partial_sequence"],
                        k=5,
                        device=device,
                        invalid_ids=invalid_ids,
                        min_length=args.min_length,
                    ),
                }
                for row in valid_inputs
            ]
            io.write_next_step_predictions(pred_dir / "nextstep.csv", rows)

        if "completion" in tasks:
            rows = [
                {
                    "example_id": row["example_id"],
                    "steps": _complete_with_best_reward(
                        model,
                        reward_fn,
                        vocabulary,
                        row["family"],
                        row["partial_sequence"],
                        device=device,
                        invalid_ids=invalid_ids,
                        min_length=args.min_length,
                        max_length=args.max_length,
                        samples=args.completion_samples,
                        temperature=args.eval_temperature,
                    ),
                }
                for row in valid_inputs
            ]
            io.write_completion_predictions(pred_dir / "completion.csv", rows)

    if "anomaly" in tasks:
        anomaly_inputs = io.read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
        rows = [
            {
                "example_id": row["example_id"],
                **_predict_anomaly(reward_fn, row["family"], row["sequence"]),
            }
            for row in anomaly_inputs
        ]
        io.write_anomaly_predictions(pred_dir / "anomaly.csv", rows)


def _evaluate_eval_sets(
    model: GFlowNetPolicy,
    reward_fn: ProcessReward,
    vocabulary: Vocabulary,
    *,
    dataset: str,
    holdout_family: str,
    eval_root: Path,
    preds_root: Path,
    metrics_root: Path,
    views: list[str],
    tasks: tuple[str, ...],
    device: torch.device,
    invalid_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, dict[str, dict]]:
    all_results: dict[str, dict[str, dict]] = {}
    for view in views:
        eval_dir = eval_root / dataset / f"holdout_{holdout_family}" / view
        if not eval_dir.exists():
            print(f"skip eval view={view}: missing {eval_dir}")
            continue
        pred_dir = preds_root / dataset / f"holdout_{holdout_family}" / view / "gflownet"
        metrics_dir = metrics_root / dataset / f"holdout_{holdout_family}" / view / "gflownet"
        print(f"evaluating gflownet view={view} eval_dir={eval_dir}")
        _write_eval_predictions(
            model,
            reward_fn,
            vocabulary,
            eval_dir=eval_dir,
            pred_dir=pred_dir,
            tasks=tasks,
            device=device,
            invalid_ids=invalid_ids,
            args=args,
        )
        all_results[view] = _score_predictions(
            eval_dir=eval_dir,
            pred_dir=pred_dir,
            metrics_dir=metrics_dir,
            tasks=tasks,
        )
        compact = {task: metrics.get("all", metrics) for task, metrics in all_results[view].items()}
        print(f"eval view={view} metrics={json.dumps(compact, indent=2)}")
    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a prefix-conditioned GFlowNet process model."
    )
    parser.add_argument("--dataset", default="valid_s010k")
    parser.add_argument("--generated-root", default=str(PROJECT_ROOT / "data" / "generated"))
    parser.add_argument("--eval-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--preds-root", default=str(PROJECT_ROOT / "outputs" / "preds"))
    parser.add_argument("--metrics-root", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    parser.add_argument("--model-root", default=str(PROJECT_ROOT / "outputs" / "models"))
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default="ic")
    parser.add_argument(
        "--eval-views",
        nargs="+",
        choices=_VIEWS,
        default=None,
        help="Eval views. Defaults to standard/diverse views when present, else id/ood.",
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument(
        "--limit-per-family",
        type=int,
        default=5000,
        help="Default gives about 10k train sequences with one held-out family.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gfn-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--gfn-weight", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.8)
    parser.add_argument("--completion-samples", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=5)

    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--min-prefix-fraction", type=float, default=0.55)
    parser.add_argument("--max-prefix-fraction", type=float, default=0.85)
    parser.add_argument("--reward-style-weight", type=float, default=0.15)
    parser.add_argument("--reward-valid-bonus", type=float, default=20.0)
    parser.add_argument("--reward-phase-bonus", type=float, default=8.0)
    parser.add_argument("--reward-family-bonus", type=float, default=4.0)

    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-context", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    generated_root = Path(args.generated_root)
    splits_dir = (
        Path(args.splits_dir) if args.splits_dir else generated_root / args.dataset / "splits"
    )
    bundle = load_split_records(
        splits_dir=splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    # The public problem statement defines the step vocabulary. Use all loaded
    # split records for the action space so the OOD family can be emitted, while
    # keeping optimization data restricted to the two non-held-out families.
    vocabulary = build_vocabulary(_all_records(bundle))
    run_name = f"gflownet_holdout_{args.holdout_family}"
    run_dir = Path(args.model_root) / args.dataset / run_name
    checkpoint_path = run_dir / "best.pt"
    print(f"dataset={args.dataset} model=gflownet holdout={args.holdout_family}")
    print(f"splits_dir={splits_dir}")
    print(f"counts={bundle.counts()}")

    train_loader = _make_loader(
        bundle.records["train"],
        vocabulary,
        batch_size=args.batch_size,
        max_context=args.max_context,
        shuffle=True,
    )
    valid_loader = _make_loader(
        bundle.records["valid"],
        vocabulary,
        batch_size=args.batch_size,
        max_context=args.max_context,
        shuffle=False,
    )

    model_config = GFlowNetConfig(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_context=args.max_context,
    )
    reward_config = RewardConfig(
        min_length=args.min_length,
        max_length=args.max_length,
        style_weight=args.reward_style_weight,
        valid_bonus=args.reward_valid_bonus,
        phase_bonus=args.reward_phase_bonus,
        family_bonus=args.reward_family_bonus,
    )
    reward_fn = ProcessReward(
        bundle.records["train"],
        config=reward_config,
        known_steps=_known_process_steps(vocabulary),
    )
    model = GFlowNetPolicy(
        len(vocabulary.id_to_token),
        model_config,
        pad_id=vocabulary.pad_id,
    )
    print(f"config={model_config.to_dict()}")
    print(f"reward_config={asdict(reward_config)}")
    print(f"parameters={count_parameters(model)}")

    device = pick_device(args.device)
    model.to(device)
    invalid_ids = invalid_action_ids(vocabulary)
    criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_id, label_smoothing=0.02)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    train_batches = len(train_loader)
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

    history: list[dict[str, Any]] = []
    best_valid_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running: dict[str, float] = {}
        seen = 0
        for step, batch in enumerate(train_loader):
            if args.max_train_batches is not None and step >= args.max_train_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["target_id"].to(device)

            logits = model(input_ids, attention_mask)
            bc_loss = criterion(logits, target)
            gfn_loss, gfn_stats = _run_gflownet_batch(
                model,
                reward_fn,
                bundle.records["train"],
                vocabulary,
                args=args,
                device=device,
                invalid_ids=invalid_ids,
                rng=rng,
            )
            loss = args.bc_weight * bc_loss + args.gfn_weight * gfn_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            row = {
                "loss": float(loss.item()),
                "bc_loss": float(bc_loss.item()),
                "gfn_loss": float(gfn_loss.item()),
                **gfn_stats,
            }
            for key, value in row.items():
                running[key] = running.get(key, 0.0) + value
            seen += 1

            if args.log_every and (step + 1) % args.log_every == 0:
                avg = {key: value / seen for key, value in running.items()}
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"epoch={epoch} step={step + 1} loss={avg['loss']:.4f} "
                    f"bc={avg['bc_loss']:.4f} gfn={avg['gfn_loss']:.4f} "
                    f"valid_rate={avg['gfn_valid_rate']:.3f} "
                    f"log_reward={avg['gfn_log_reward']:.2f} lr={lr:.2e}"
                )

        valid_topk = evaluate_model(
            model,
            valid_loader,
            device=device,
            k=args.k,
            max_batches=args.max_eval_batches,
        )["all"]
        avg = {key: round(value / max(1, seen), 6) for key, value in running.items()}
        epoch_row = {"epoch": epoch, **avg, "valid_topk": valid_topk}
        history.append(epoch_row)
        valid_topk_key = f"top{args.k}"
        print(
            f"epoch={epoch} loss={avg.get('loss', 0.0):.4f} "
            f"valid_top1={valid_topk['top1']:.4f} "
            f"valid_top{args.k}={valid_topk[valid_topk_key]:.4f}"
        )

        if valid_topk["top1"] > best_valid_top1:
            best_valid_top1 = valid_topk["top1"]
            _save_checkpoint(
                checkpoint_path,
                model=model,
                vocabulary=vocabulary,
                model_config=model_config,
                reward_config=reward_config,
                args=args,
                epoch=epoch,
                history=history,
            )
            print(f"saved best checkpoint: {checkpoint_path}")

        if args.sample_every and epoch % args.sample_every == 0:
            record = rng.choice(bundle.records["valid"])
            cut = _prefix_cut(record.steps, rng, args)
            continuation = sample_completion(
                model,
                vocabulary,
                record.family,
                record.steps[:cut],
                device=device,
                min_length=args.min_length,
                max_length=args.max_length,
                invalid_ids=invalid_ids,
            )
            completed = list(record.steps[:cut]) + continuation
            reward = reward_fn.evaluate(record.family, record.steps[:cut], completed)
            print(
                f"sample family={record.family} prefix={cut} generated={len(continuation)} "
                f"log_reward={reward.log_reward:.2f} valid={reward.is_valid} "
                f"violations={list(reward.violations)}"
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"wrote history: {run_dir / 'history.json'}")

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        print(f"loaded best epoch={checkpoint['epoch']} for eval-set predictions")

    views = args.eval_views or _default_views(
        Path(args.eval_root), args.dataset, args.holdout_family
    )
    eval_results = _evaluate_eval_sets(
        model,
        reward_fn,
        vocabulary,
        dataset=args.dataset,
        holdout_family=args.holdout_family,
        eval_root=Path(args.eval_root),
        preds_root=Path(args.preds_root),
        metrics_root=Path(args.metrics_root),
        views=views,
        tasks=tuple(args.tasks),
        device=device,
        invalid_ids=invalid_ids,
        args=args,
    )
    (run_dir / "eval_set_metrics.json").write_text(
        json.dumps(eval_results, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
