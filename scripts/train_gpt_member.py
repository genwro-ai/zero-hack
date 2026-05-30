#!/usr/bin/env python3
import argparse
import random

import torch

from zero_hack.models.common import DEFAULT_SPLITS_DIR, load_split_records, pick_device
from zero_hack.models.gpt_lm import CausalLM, fit, save_member


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=6e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(None)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    bundle = load_split_records(
        args.splits_dir, holdout_family=args.holdout_family, limit_per_family=args.limit_per_family
    )
    config = {
        "d_model": args.d_model,
        "n_heads": args.heads,
        "n_layers": args.layers,
        "max_len": args.max_len,
        "dropout": 0.1,
    }
    model = CausalLM(
        len(bundle.vocabulary.id_to_token),
        config["d_model"],
        config["n_heads"],
        config["n_layers"],
        config["max_len"],
        config["dropout"],
    )
    model = fit(
        model,
        bundle.records["train"],
        bundle.vocabulary,
        device,
        args.epochs,
        args.batch_size,
        args.lr,
    )
    save_member(args.out, model, config, bundle.vocabulary)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
