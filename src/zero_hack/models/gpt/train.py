"""Train a small GPT-style decoder on one generated dataset and holdout split."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler

from zero_hack import PROJECT_ROOT
from zero_hack.data import (
    FAMILY_TOKENS,
    SPECIAL_TOKENS,
    NextStepDataset,
    SequenceRecord,
    Vocabulary,
    build_vocabulary,
)
from zero_hack.eval import io
from zero_hack.eval.score import score_task
from zero_hack.eval.validator import first_violated_rule
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold_from_eval_dir
from zero_hack.models.common import (
    DataBundle,
    count_parameters,
    evaluate_model,
    load_split_records,
    pick_device,
)
from zero_hack.models.gpt.model import GPTConfig, GPTNextStepModel
from zero_hack.models.phase_loss import NextPhaseLoss

MODEL_NAME = "gpt_decoder"
SEQUENCE_TERMINATOR = "SHIP LOT"
EVAL_FAMILY_MODES = ("as_given", "holdout_unknown", "all_unknown")


def build_model(bundle: DataBundle, config: GPTConfig) -> GPTNextStepModel:
    return GPTNextStepModel(
        vocab_size=len(bundle.vocabulary.id_to_token),
        config=config,
        pad_id=bundle.vocabulary.pad_id,
    )


def _invalid_prediction_ids(vocabulary: Vocabulary) -> list[int]:
    invalid = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
    return [vocabulary.token_to_id[token] for token in invalid if token in vocabulary.token_to_id]


def _load_augmentation_records(
    path: str | Path,
    *,
    family_mode: str = "unknown",
    limit: int | None = None,
    seed: int = 1729,
) -> list[SequenceRecord]:
    """Load long-form generated records for train-only augmentation.

    ``scripts/generate_augmented_sequences.py`` intentionally emits synthetic and UNK
    labels. The GPT data loader only has three known family tokens plus an
    unknown token, so the default maps every augmentation row to ``unknown``.
    """

    rows_by_key: dict[tuple[str, str], list[str]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_family = (row.get("FAMILY") or "unknown").strip().lower()
            if family_mode == "preserve-known" and raw_family in FAMILY_TOKENS:
                family = raw_family
            else:
                family = "unknown"

            sequence_id = (row.get("SEQUENCE_ID") or "").strip()
            step = (row.get("STEP") or "").strip()
            if sequence_id and step:
                rows_by_key.setdefault((family, sequence_id), []).append(step)

    records = [
        SequenceRecord(family=family, sequence_id=f"augment_{sequence_id}", steps=tuple(steps))
        for (family, sequence_id), steps in rows_by_key.items()
    ]
    if limit is not None and len(records) > limit:
        rng = random.Random(seed)
        records = list(records)
        rng.shuffle(records)
        records = records[:limit]
    return records


def _augment_training_records(
    bundle: DataBundle,
    augmentation: list[SequenceRecord],
) -> DataBundle:
    if not augmentation:
        return bundle

    records = dict(bundle.records)
    protected = {
        record.steps
        for split, split_records in records.items()
        if split != "train"
        for record in split_records
    }
    train_steps = {record.steps for record in records["train"]}
    filtered: list[SequenceRecord] = []
    skipped_protected = 0
    skipped_train_duplicate = 0
    for record in augmentation:
        if record.steps in protected:
            skipped_protected += 1
            continue
        if record.steps in train_steps:
            skipped_train_duplicate += 1
            continue
        filtered.append(record)
        train_steps.add(record.steps)

    if skipped_protected or skipped_train_duplicate:
        print(
            "filtered augmentation duplicates: "
            f"heldout_or_validation={skipped_protected} "
            f"train={skipped_train_duplicate}"
        )

    records["train"] = [*records["train"], *filtered]
    return DataBundle(
        vocabulary=build_vocabulary(records["train"]),
        records=records,
        train_families=bundle.train_families,
        holdout_family=bundle.holdout_family,
    )


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


def _model_family_for_eval(
    family: str,
    *,
    holdout_family: str | None,
    eval_family_mode: str,
) -> str:
    if eval_family_mode == "all_unknown":
        return "unknown"
    if eval_family_mode == "holdout_unknown" and family.lower() == (holdout_family or "").lower():
        return "unknown"
    return family


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


@torch.no_grad()
def _complete_greedy(
    model: GPTNextStepModel,
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str],
    *,
    device: torch.device,
    invalid_ids: list[int],
    max_steps: int,
) -> list[str]:
    if prefix and prefix[-1] == SEQUENCE_TERMINATOR:
        return []

    sequence = list(prefix)
    produced: list[str] = []
    while len(produced) < max_steps:
        top = predict_topk(
            model,
            vocabulary,
            family,
            sequence,
            k=1,
            device=device,
            invalid_ids=invalid_ids,
        )
        if not top:
            break
        next_step = top[0]
        produced.append(next_step)
        sequence.append(next_step)
        if next_step == SEQUENCE_TERMINATOR:
            break
    return produced


@torch.no_grad()
def _sequence_avg_logprob(
    model: GPTNextStepModel,
    vocabulary: Vocabulary,
    family: str,
    steps: list[str] | tuple[str, ...],
    *,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for position, step in enumerate(steps):
        input_ids, attention_mask = _encode_prefix(
            vocabulary,
            family,
            steps[:position],
            max_context=model.config.max_context,
            device=device,
        )
        logits = model(input_ids, attention_mask).squeeze(0)
        token_id = vocabulary.token_to_id.get(step, vocabulary.unk_id)
        total += float(torch.log_softmax(logits, dim=-1)[token_id].item())
    return total / max(1, len(steps))


class _GPTLikelihoodAdapter:
    def __init__(
        self,
        model: GPTNextStepModel,
        vocabulary: Vocabulary,
        *,
        device: torch.device,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.device = device

    def score_sequence(self, family: str, steps: list[str] | tuple[str, ...]) -> float:
        avg = _sequence_avg_logprob(
            self.model,
            self.vocabulary,
            family,
            steps,
            device=self.device,
        )
        return avg * max(1, len(steps))


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
    family_dropout: float = 0.0,
    step_dropout: float = 0.0,
    max_train_batches: int | None = None,
) -> dict[str, DataLoader]:
    loaders = {}
    for split, records in bundle.records.items():
        dataset = NextStepDataset(
            records=records,
            vocabulary=bundle.vocabulary,
            max_context=max_context,
            family_dropout=family_dropout if split == "train" else 0.0,
            step_dropout=step_dropout if split == "train" else 0.0,
        )
        sampler = None
        shuffle = split == "train"
        if split == "train" and max_train_batches is not None:
            # Avoid constructing a random permutation over tens of millions of
            # prefix examples when each epoch intentionally trains on a capped
            # random budget.
            sampler = RandomSampler(
                dataset,
                replacement=True,
                num_samples=max_train_batches * batch_size,
            )
            shuffle = False
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
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
    phase_loss_fn: NextPhaseLoss | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    max_batches: int | None = None,
    grad_clip: float = 1.0,
    log_every: int = 100,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_token_loss = 0.0
    total_phase_loss = 0.0
    seen = 0

    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target = batch["target_id"].to(device)

        with torch.set_grad_enabled(training):
            logits = model(input_ids, attention_mask)
            token_loss = criterion(logits, target)
            phase_loss_value = token_loss.detach() * 0.0
            loss = token_loss
            if phase_loss_fn is not None:
                phase_out = phase_loss_fn(
                    logits,
                    target,
                    return_output=True,
                )
                loss = loss + phase_out.loss
                phase_loss_value = phase_out.phase_loss.detach()
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += float(loss.item())
        total_token_loss += float(token_loss.item())
        total_phase_loss += float(phase_loss_value.item())
        seen += 1
        if training and log_every and (step + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step + 1} train_loss={total_loss / seen:.4f} "
                f"token={total_token_loss / seen:.4f} "
                f"phase={total_phase_loss / seen:.4f} lr={lr:.2e}"
            )

    denom = max(1, seen)
    return {
        "loss": total_loss / denom,
        "token_loss": total_token_loss / denom,
        "phase_loss": total_phase_loss / denom,
    }


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
    model: GPTNextStepModel,
    bundle: DataBundle,
    *,
    eval_dir: Path,
    pred_dir: Path,
    tasks: tuple[str, ...],
    device: torch.device,
    k: int,
    max_completion_steps: int,
    anomaly_threshold: float,
    holdout_family: str | None,
    eval_family_mode: str,
) -> None:
    pred_dir.mkdir(parents=True, exist_ok=True)
    invalid_ids = _invalid_prediction_ids(bundle.vocabulary)
    if "next_step" in tasks or "completion" in tasks:
        inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
        if "next_step" in tasks:
            rows = [
                {
                    "example_id": row["example_id"],
                    "ranks": predict_topk(
                        model,
                        bundle.vocabulary,
                        _model_family_for_eval(
                            row["family"],
                            holdout_family=holdout_family,
                            eval_family_mode=eval_family_mode,
                        ),
                        row["partial_sequence"],
                        k=k,
                        device=device,
                        invalid_ids=invalid_ids,
                    ),
                }
                for row in inputs
            ]
            io.write_next_step_predictions(pred_dir / "nextstep.csv", rows)

        if "completion" in tasks:
            rows = [
                {
                    "example_id": row["example_id"],
                    "steps": _complete_greedy(
                        model,
                        bundle.vocabulary,
                        _model_family_for_eval(
                            row["family"],
                            holdout_family=holdout_family,
                            eval_family_mode=eval_family_mode,
                        ),
                        row["partial_sequence"],
                        device=device,
                        invalid_ids=invalid_ids,
                        max_steps=max_completion_steps,
                    ),
                }
                for row in inputs
            ]
            io.write_completion_predictions(pred_dir / "completion.csv", rows)

    if "anomaly" in tasks:
        anomaly_rows = []
        for row in io.read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv"):
            avg_logprob = _sequence_avg_logprob(
                model,
                bundle.vocabulary,
                _model_family_for_eval(
                    row["family"],
                    holdout_family=holdout_family,
                    eval_family_mode=eval_family_mode,
                ),
                row["sequence"],
                device=device,
            )
            valid = avg_logprob >= anomaly_threshold
            anomaly_rows.append(
                {
                    "example_id": row["example_id"],
                    "is_valid": int(valid),
                    "score": 1.0 / (1.0 + math.exp(-(avg_logprob - anomaly_threshold))),
                    "predicted_rule": None
                    if valid
                    else (first_violated_rule(row["sequence"]) or "RULE_DEP_NO_CLEAN"),
                }
            )
        io.write_anomaly_predictions(pred_dir / "anomaly.csv", anomaly_rows)


def _evaluate_eval_sets(
    model: GPTNextStepModel,
    bundle: DataBundle,
    *,
    method_name: str,
    dataset: str,
    holdout_family: str,
    eval_root: Path,
    preds_root: Path,
    metrics_root: Path,
    views: list[str],
    tasks: tuple[str, ...],
    device: torch.device,
    k: int,
    max_completion_steps: int,
    anomaly_threshold: float,
    eval_family_mode: str,
) -> dict[str, dict[str, dict]]:
    all_results: dict[str, dict[str, dict]] = {}
    for view in views:
        eval_dir = eval_root / dataset / f"holdout_{holdout_family}" / view
        if not eval_dir.exists():
            print(f"skip eval view={view}: missing {eval_dir}")
            continue

        pred_dir = preds_root / dataset / f"holdout_{holdout_family}" / view / method_name
        metrics_dir = metrics_root / dataset / f"holdout_{holdout_family}" / view / method_name
        print(f"evaluating {method_name} view={view} eval_dir={eval_dir}")
        _write_eval_predictions(
            model,
            bundle,
            eval_dir=eval_dir,
            pred_dir=pred_dir,
            tasks=tasks,
            device=device,
            k=k,
            max_completion_steps=max_completion_steps,
            anomaly_threshold=anomaly_threshold,
            holdout_family=holdout_family,
            eval_family_mode=eval_family_mode,
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


def _resolve_calibration_dir(args: argparse.Namespace) -> Path | None:
    if args.calibration_dir:
        return Path(args.calibration_dir)
    return Path(args.eval_root) / args.dataset / f"holdout_{args.holdout_family}" / "calibration"


def _tune_threshold(
    adapter: _GPTLikelihoodAdapter,
    bundle: DataBundle,
    args: argparse.Namespace,
) -> tuple[float, dict[str, Any]]:
    calibration_dir = _resolve_calibration_dir(args)
    if calibration_dir is None or not calibration_dir.exists():
        raise FileNotFoundError(
            f"Missing threshold calibration set: {calibration_dir}. "
            "Run scripts/make_all_eval_sets.py first."
        )
    threshold_result = tune_anomaly_threshold_from_eval_dir(adapter, calibration_dir)
    return threshold_result.threshold, {
        "source": "threshold_calibration",
        "objective": "f1",
        "tuned_on": str(calibration_dir),
        "train_families": list(bundle.train_families),
        "threshold": threshold_result.threshold,
        "val_f1": threshold_result.f1,
        "val_precision": threshold_result.precision,
        "val_recall": threshold_result.recall,
    }


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
    parser.add_argument("--method-name", default=MODEL_NAME)
    parser.add_argument(
        "--tasks", nargs="+", choices=("next_step", "completion", "anomaly"), default=["next_step"]
    )
    parser.add_argument(
        "--eval-views",
        nargs="+",
        default=None,
        help="Eval views. Defaults to standard/diverse views when present, else id/ood.",
    )
    parser.add_argument(
        "--eval-family-mode",
        choices=EVAL_FAMILY_MODES,
        default="holdout_unknown",
        help="Map held-out-family eval prompts to <FAMILY_UNKNOWN> by default.",
    )
    parser.add_argument(
        "--calibration-dir",
        default=None,
        help=(
            "Optional fixed eval-set directory for anomaly threshold tuning. "
            "Defaults to <eval-root>/<dataset>/holdout_<family>/calibration."
        ),
    )
    parser.add_argument("--augment-train-csv", default=None)
    parser.add_argument("--augment-limit", type=int, default=None)
    parser.add_argument(
        "--augment-family-mode",
        choices=("unknown", "preserve-known"),
        default="unknown",
        help="How to map FAMILY labels from --augment-train-csv.",
    )
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
    parser.add_argument("--phase-loss-weight", type=float, default=0.0)
    parser.add_argument("--family-dropout", type=float, default=0.0)
    parser.add_argument("--step-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-completion-steps", type=int, default=240)

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
    run_name = f"{args.method_name}_holdout_{args.holdout_family}"
    run_dir = Path(args.model_root) / args.dataset / run_name
    checkpoint_path = run_dir / "best.pt"

    bundle = load_split_records(
        splits_dir=splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    if args.augment_train_csv:
        augmentation = _load_augmentation_records(
            args.augment_train_csv,
            family_mode=args.augment_family_mode,
            limit=args.augment_limit,
            seed=args.seed,
        )
        bundle = _augment_training_records(bundle, augmentation)
        print(
            f"loaded {len(augmentation)} augmentation records "
            f"from {args.augment_train_csv} family_mode={args.augment_family_mode}"
        )
    print(f"dataset={args.dataset} holdout={args.holdout_family} counts={bundle.counts()}")

    loaders = _make_gpt_loaders(
        bundle,
        batch_size=args.batch_size,
        max_context=args.max_context,
        family_dropout=args.family_dropout,
        step_dropout=args.step_dropout,
        max_train_batches=args.max_train_batches,
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
    phase_loss_fn = None
    if args.phase_loss_weight > 0:
        phase_loss_fn = NextPhaseLoss.from_vocabulary(
            bundle.vocabulary,
            weight=args.phase_loss_weight,
        ).to(device)
        print(f"phase_loss_weight={args.phase_loss_weight}")
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
        train_stats = _run_epoch(
            model,
            loaders["train"],
            criterion=criterion,
            device=device,
            phase_loss_fn=phase_loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            max_batches=args.max_train_batches,
            grad_clip=args.grad_clip,
        )
        valid_stats = _run_epoch(
            model,
            loaders["valid"],
            criterion=criterion,
            device=device,
            phase_loss_fn=phase_loss_fn,
            max_batches=args.max_eval_batches,
        )
        train_loss = train_stats["loss"]
        valid_loss = valid_stats["loss"]
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
            "train_token_loss": round(train_stats["token_loss"], 6),
            "train_phase_loss": round(train_stats["phase_loss"], 6),
            "valid_loss": round(valid_loss, 6),
            "valid_token_loss": round(valid_stats["token_loss"], 6),
            "valid_phase_loss": round(valid_stats["phase_loss"], 6),
            "valid_topk": valid_topk,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"valid_token={valid_stats['token_loss']:.4f} "
            f"valid_phase={valid_stats['phase_loss']:.4f} "
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

    anomaly_threshold = -math.inf
    if "anomaly" in args.tasks:
        adapter = _GPTLikelihoodAdapter(model, bundle.vocabulary, device=device)
        anomaly_threshold, tuning = _tune_threshold(adapter, bundle, args)
        (run_dir / "anomaly_threshold.json").write_text(
            json.dumps(tuning, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"tuned anomaly threshold={tuning['threshold']:.4f} "
            f"val_f1={tuning['val_f1']:.4f} source={tuning['source']}"
        )

    views = args.eval_views or _default_views(
        Path(args.eval_root),
        args.dataset,
        args.holdout_family,
    )
    eval_set_results = _evaluate_eval_sets(
        model,
        bundle,
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
        json.dumps(eval_set_results, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
