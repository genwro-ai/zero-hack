from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, Vocabulary
from zero_hack.eval import io
from zero_hack.eval.score import score_task


def invalid_prediction_ids(vocabulary: Vocabulary) -> list[int]:
    invalid = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
    return [vocabulary.token_to_id[token] for token in invalid if token in vocabulary.token_to_id]


def encode_prefix(
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
    model: nn.Module,
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str] | tuple[str, ...],
    *,
    k: int,
    max_context: int,
    device: torch.device,
    invalid_ids: list[int] | None = None,
) -> list[str]:
    model.eval()
    input_ids, attention_mask = encode_prefix(
        vocabulary,
        family,
        prefix,
        max_context=max_context,
        device=device,
    )
    logits = model(input_ids, attention_mask).squeeze(0)
    for token_id in invalid_ids or ():
        logits[token_id] = -torch.inf
    top_ids = torch.topk(logits, k=min(k, logits.numel())).indices.tolist()
    return [vocabulary.id_to_token[token_id] for token_id in top_ids]


def write_neural_next_step_eval(
    model: nn.Module,
    vocabulary: Vocabulary,
    *,
    method_name: str,
    dataset: str,
    holdout_family: str,
    eval_root: str | Path,
    preds_root: str | Path,
    metrics_root: str | Path,
    device: torch.device,
    k: int,
    max_context: int,
    views: tuple[str, ...] = ("id", "ood"),
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    invalid_ids = invalid_prediction_ids(vocabulary)

    for view in views:
        eval_dir = Path(eval_root) / dataset / f"holdout_{holdout_family}" / view
        if not eval_dir.exists():
            print(f"skip fixed eval {view}: missing {eval_dir}")
            continue

        rows = [
            {
                "example_id": row["example_id"],
                "ranks": predict_topk(
                    model,
                    vocabulary,
                    row["family"],
                    row["partial_sequence"],
                    k=k,
                    max_context=max_context,
                    device=device,
                    invalid_ids=invalid_ids,
                ),
            }
            for row in io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
        ]

        pred_dir = Path(preds_root) / dataset / f"holdout_{holdout_family}" / view / method_name
        pred_path = pred_dir / "nextstep.csv"
        io.write_next_step_predictions(pred_path, rows)

        metrics = score_task(
            "next_step",
            ground_truth=eval_dir / "nextstep_truth.csv",
            predictions=pred_path,
            eval_input=eval_dir / "eval_input_valid.csv",
        )
        metrics_dir = (
            Path(metrics_root) / dataset / f"holdout_{holdout_family}" / view / method_name
        )
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "next_step.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        results[view] = metrics
        print(f"fixed eval {view}: {metrics.get('all', metrics)}")

    return results
