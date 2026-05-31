#!/usr/bin/env python
"""GRPO (Group Relative Policy Optimization) finetuning CLI.

RLVR-style finetuning of a pretrained next-step model (LSTM or transformer
decoder) using the process-rule validator as a verifiable reward, for the
sequence-completion objective. This is a thin CLI over
:mod:`zero_hack.models.grpo`.

Experimental integrity
----------------------
The held-out family is *purely* for evaluation and must stay unseen. GRPO
prompts therefore come from the TRAINING families by default (``--prompt-families``
defaults to the bundle's ``train_families``, which excludes the holdout).
Pointing ``--prompt-families`` at the holdout family would leak it into training
and violate the unseen-eval protocol — don't do it unless you know exactly why.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from zero_hack import PROJECT_ROOT
from zero_hack.models.common import load_split_records, pick_device
from zero_hack.models.grpo import (
    GRPOConfig,
    GRPOTrainer,
    RewardConfig,
    StepPolicy,
    build_prompts,
    load_policy,
    save_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Pretrained policy checkpoint (.pt).")
    parser.add_argument("--dataset", default="valid_s005k")
    parser.add_argument("--generated-root", default=str(PROJECT_ROOT / "data" / "generated"))
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Defaults to <generated-root>/<dataset>/splits.",
    )
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), required=True)
    parser.add_argument(
        "--prompt-families",
        nargs="*",
        default=None,
        help=(
            "Families to draw prompts from. Default: the bundle's train families "
            "(excludes the holdout). Pointing this at the holdout family violates "
            "the unseen-eval protocol."
        ),
    )
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument(
        "--prompt-splits",
        nargs="*",
        default=["train", "valid"],
        help="Which splits to cut prompts from (train/valid only; never holdout test).",
    )

    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200, help="Number of GRPO updates.")
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.6, 0.8])
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)

    # Reward weights (see RewardConfig). Quality = weighted block/token/exact/
    # length/diversity vs the gold suffix; multiplied by the validity gate.
    parser.add_argument("--w-block", type=float, default=0.6)
    parser.add_argument("--w-token", type=float, default=0.4)
    parser.add_argument("--w-exact", type=float, default=0.5)
    parser.add_argument("--w-length", type=float, default=0.3)
    parser.add_argument("--w-diversity", type=float, default=0.2)
    parser.add_argument("--termination-bonus", type=float, default=0.1)
    parser.add_argument("--truncation-penalty", type=float, default=0.3)
    graded = parser.add_mutually_exclusive_group()
    graded.add_argument("--graded-validity", dest="graded_validity", action="store_true")
    graded.add_argument("--no-graded-validity", dest="graded_validity", action="store_false")
    parser.set_defaults(graded_validity=True)

    parser.add_argument(
        "--mask-sampling",
        action="store_true",
        help="Apply ViolationMask during sampling (off by default so reward is informative).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default=None,
        help="Output checkpoint. Default: best_grpo.pt next to --checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    generated_root = Path(args.generated_root)
    splits_dir = (
        Path(args.splits_dir) if args.splits_dir else generated_root / args.dataset / "splits"
    )

    bundle = load_split_records(
        splits_dir=splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )

    prompt_families = tuple(f.lower() for f in (args.prompt_families or bundle.train_families))
    if args.holdout_family in prompt_families:
        print(
            f"WARNING: --prompt-families includes the holdout family "
            f"'{args.holdout_family}'. This leaks the held-out family into "
            f"training and violates the unseen-eval protocol."
        )

    # Collect prompt source records from the requested splits, filtered to the
    # prompt families. The holdout family never appears in train/valid splits, so
    # this stays unseen by construction.
    source_records = []
    for split in args.prompt_splits:
        if split not in bundle.records:
            print(f"skip prompt split '{split}': not in bundle")
            continue
        for record in bundle.records[split]:
            if record.family.lower() in prompt_families:
                source_records.append(record)

    prompts = build_prompts(source_records, fractions=tuple(args.fractions))
    if not prompts:
        raise SystemExit(
            "No prompts built. Check --prompt-families / --prompt-splits / --fractions."
        )
    print(
        f"dataset={args.dataset} holdout={args.holdout_family} "
        f"prompt_families={prompt_families} prompts={len(prompts)} "
        f"(from {len(source_records)} records)"
    )

    device = pick_device(args.device)
    policy: StepPolicy = load_policy(args.checkpoint, device=device)
    print(
        f"loaded policy kind={policy.kind} vocab={len(policy.vocabulary.id_to_token)} "
        f"max_context={policy.max_context} device={device}"
    )

    grpo_config = GRPOConfig(
        group_size=args.group_size,
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        lr=args.lr,
        kl_coef=args.kl_coef,
        temperature=args.temperature,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        mask_sampling=args.mask_sampling,
        log_every=args.log_every,
    )
    reward_config = RewardConfig(
        w_block=args.w_block,
        w_token=args.w_token,
        w_exact=args.w_exact,
        w_length=args.w_length,
        w_diversity=args.w_diversity,
        termination_bonus=args.termination_bonus,
        truncation_penalty=args.truncation_penalty,
        graded_validity=args.graded_validity,
    )

    trainer = GRPOTrainer(
        policy,
        prompts,
        config=grpo_config,
        reward_config=reward_config,
        seed=args.seed,
    )
    history = trainer.train()

    out = Path(args.out) if args.out else Path(args.checkpoint).with_name("best_grpo.pt")
    save_policy(
        policy,
        out,
        source_checkpoint=args.checkpoint,
        extra_meta={
            "method": "grpo",
            "holdout_family": args.holdout_family,
            "prompt_families": list(prompt_families),
            "config": grpo_config.__dict__,
            "reward": reward_config.__dict__,
        },
    )
    print(f"saved finetuned policy -> {out}")

    history_path = out.with_name("grpo_history.json")
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {history_path}")

    if history:
        last = history[-1]
        print(
            f"final step: mean_reward={last['mean_reward']:.4f} "
            f"valid_rate={last['valid_rate']:.4f} "
            f"block_acc={last['block_accuracy']:.4f} "
            f"exact={last['exact_rate']:.4f} kl={last['kl']:.4f}"
        )


if __name__ == "__main__":
    main()
