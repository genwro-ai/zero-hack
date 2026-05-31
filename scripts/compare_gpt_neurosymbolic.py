#!/usr/bin/env python3
"""Compare bare GPT next-step logits with neurosymbolic decoding heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from zero_hack import PROJECT_ROOT
from zero_hack.data import FAMILY_TOKENS, Vocabulary
from zero_hack.eval import io
from zero_hack.eval.score import score_task
from zero_hack.models.common import pick_device
from zero_hack.models.gpt.model import GPTConfig, GPTNextStepModel
from zero_hack.models.gpt.train import EVAL_FAMILY_MODES, _model_family_for_eval
from zero_hack.models.neurosymbolic.decoding import shape_logits, topk_steps


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


def _load_gpt_checkpoint(path: Path, device: torch.device) -> tuple[GPTNextStepModel, Vocabulary]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    vocabulary = Vocabulary(
        token_to_id=dict(checkpoint["vocabulary"]["token_to_id"]),
        id_to_token=tuple(checkpoint["vocabulary"]["id_to_token"]),
    )
    config = GPTConfig(**checkpoint["model_config"])
    model = GPTNextStepModel(
        vocab_size=len(vocabulary.id_to_token),
        config=config,
        pad_id=vocabulary.pad_id,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, vocabulary


@torch.no_grad()
def _predict_all_heads(
    model: GPTNextStepModel,
    vocabulary: Vocabulary,
    *,
    family: str,
    prefix: list[str],
    k: int,
    device: torch.device,
) -> dict[str, list[str]]:
    input_ids, attention_mask = _encode_prefix(
        vocabulary,
        family,
        prefix,
        max_context=model.config.max_context,
        device=device,
    )
    logits = model(input_ids, attention_mask).squeeze(0)
    return {
        mode: topk_steps(shape_logits(prefix, logits, vocabulary, mode=mode), vocabulary, k=k)
        for mode in ("none", "hard", "shaped")
    }


def _metric_method_name(prefix: str, head: str) -> str:
    suffix = {"none": "bare", "hard": "ns_hard", "shaped": "ns_shaped"}[head]
    return f"{prefix}_{suffix}" if prefix else suffix


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a GPT checkpoint .pt file.")
    parser.add_argument(
        "--eval-input",
        default=None,
        help="eval_input_valid.csv path. Defaults to the dataset/holdout/view eval path.",
    )
    parser.add_argument("--dataset", default="valid_s100k_augmented_s050k")
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default="ic")
    parser.add_argument("--view", choices=("id", "ood"), default="ood")
    parser.add_argument("--eval-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--preds-root", default=str(PROJECT_ROOT / "outputs" / "preds"))
    parser.add_argument("--metrics-root", default=str(PROJECT_ROOT / "outputs" / "metrics"))
    parser.add_argument(
        "--output",
        default=None,
        help="Detailed JSONL comparison path. Defaults under outputs/neurosymbolic/.",
    )
    parser.add_argument("--method-prefix", default="gpt_phase_augmented")
    parser.add_argument(
        "--eval-family-mode",
        choices=EVAL_FAMILY_MODES,
        default="holdout_unknown",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = pick_device(args.device)
    model, vocabulary = _load_gpt_checkpoint(Path(args.checkpoint), device)
    eval_input = (
        Path(args.eval_input)
        if args.eval_input
        else Path(args.eval_root)
        / args.dataset
        / f"holdout_{args.holdout_family}"
        / args.view
        / "eval_input_valid.csv"
    )
    rows = io.read_eval_input_valid(eval_input)
    if args.limit is not None:
        rows = rows[: args.limit]

    output = (
        Path(args.output)
        if args.output
        else Path(PROJECT_ROOT)
        / "outputs"
        / "neurosymbolic"
        / args.dataset
        / f"holdout_{args.holdout_family}_{args.view}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pred_rows = {"none": [], "hard": [], "shaped": []}
    changed = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            eval_family = _model_family_for_eval(
                row["family"],
                holdout_family=args.holdout_family,
                eval_family_mode=args.eval_family_mode,
            )
            heads = _predict_all_heads(
                model,
                vocabulary,
                family=eval_family,
                prefix=row["partial_sequence"],
                k=args.k,
                device=device,
            )
            for head, ranks in heads.items():
                pred_rows[head].append({"example_id": row["example_id"], "ranks": ranks})
            changed += int(
                heads["none"][:1] != heads["hard"][:1] or heads["hard"][:1] != heads["shaped"][:1]
            )
            record: dict[str, Any] = {
                "example_id": row["example_id"],
                "family": row["family"],
                "model_family": eval_family,
                "prefix_len": len(row["partial_sequence"]),
                "bare_topk": heads["none"],
                "hard_topk": heads["hard"],
                "shaped_topk": heads["shaped"],
            }
            handle.write(json.dumps(record) + "\n")

    print(f"wrote {len(rows)} comparisons to {output}")
    print(f"top1 changed by at least one head for {changed}/{len(rows)} examples")

    if args.limit is not None:
        print("skip metrics: --limit was set")
        return

    eval_dir = eval_input.parent
    for head, predictions in pred_rows.items():
        method_name = _metric_method_name(args.method_prefix, head)
        pred_dir = (
            Path(args.preds_root)
            / args.dataset
            / f"holdout_{args.holdout_family}"
            / args.view
            / method_name
        )
        pred_path = pred_dir / "nextstep.csv"
        io.write_next_step_predictions(pred_path, predictions)

        metrics = score_task(
            "next_step",
            ground_truth=eval_dir / "nextstep_truth.csv",
            predictions=pred_path,
            eval_input=eval_dir / "eval_input_valid.csv",
        )
        metrics_dir = (
            Path(args.metrics_root)
            / args.dataset
            / f"holdout_{args.holdout_family}"
            / args.view
            / method_name
        )
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "next_step.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"{method_name}: {metrics.get('all', metrics)}")


if __name__ == "__main__":
    main()
