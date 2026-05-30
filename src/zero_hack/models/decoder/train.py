import argparse
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from zero_hack import PROJECT_ROOT
from zero_hack.data import FAMILY_TOKENS
from zero_hack.models.common import (
    DEFAULT_SPLITS_DIR,
    count_parameters,
    load_split_records,
    pick_device,
)
from zero_hack.models.decoder.model import DecoderConfig, DecoderLM, save_checkpoint
from zero_hack.models.topk import TopKAccumulator

IGNORE = -100


class SequenceDataset(Dataset):
    def __init__(self, records, vocab, max_seq_len):
        self.rows = []
        for r in records:
            ids = vocab.encode(["<BOS>", FAMILY_TOKENS[r.family], *r.steps, "<EOS>"])[:max_seq_len]
            if len(ids) >= 2:
                self.rows.append(ids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch, pad_id):
    width = max(len(ids) for ids in batch) - 1
    inputs, labels = [], []
    for ids in batch:
        pad = width - len(ids) + 1
        inputs.append(ids[:-1] + [pad_id] * pad)
        labels.append([IGNORE] + ids[2:] + [IGNORE] * pad)
    return torch.tensor(inputs), torch.tensor(labels)


def make_loader(records, vocab, cfg, batch_size, shuffle):
    data = SequenceDataset(records, vocab, cfg.max_seq_len)
    collate_fn = partial(collate, pad_id=vocab.pad_id)
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


def run_epoch(model, batches, optimizer, scheduler, device):
    model.train()
    total = 0.0
    for inputs, labels in batches:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
    return total / max(len(batches), 1)


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    acc = TopKAccumulator(k=3)
    for inputs, labels in batches:
        inputs, labels = inputs.to(device), labels.to(device)
        top = model(inputs).topk(3, dim=-1).indices
        rows, cols = (labels != IGNORE).nonzero(as_tuple=True)
        gold = labels[rows, cols].tolist()
        preds = top[rows, cols].tolist()
        for g, p in zip(gold, preds, strict=False):
            acc.update(g, p)
    return acc.summary()["all"]


def parse_args():
    p = argparse.ArgumentParser(description="Train the decoder-only next-step LM.")
    p.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    p.add_argument("--limit-per-family", type=int, default=None)
    p.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "decoder" / "best.pt"))
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    bundle = load_split_records(
        args.splits_dir, holdout_family=args.holdout_family, limit_per_family=args.limit_per_family
    )
    print(f"counts: {bundle.counts()}")
    vocab = bundle.vocabulary
    cfg = DecoderConfig(d_model=args.d_model, n_layers=args.n_layers)

    train = make_loader(bundle.records["train"], vocab, cfg, args.batch_size, shuffle=True)
    valid = make_loader(bundle.records["valid"], vocab, cfg, args.batch_size, shuffle=False)
    model = DecoderLM(len(vocab.id_to_token), cfg).to(device)
    print(f"parameters: {count_parameters(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train) * args.epochs
    )

    best = -1.0
    for epoch in range(args.epochs):
        loss = run_epoch(model, train, optimizer, scheduler, device)
        metrics = evaluate(model, valid, device)
        print(f"epoch {epoch + 1} loss {loss:.4f} valid {metrics}")
        if metrics["top1"] > best:
            best = metrics["top1"]
            save_checkpoint(args.out, model, cfg, vocab)

    for split in bundle.test_split_names:
        loader = make_loader(bundle.records[split], vocab, cfg, args.batch_size, shuffle=False)
        role = "ood" if split.removeprefix("test_") == bundle.holdout_family else "id"
        print(f"{split} ({role}) {evaluate(model, loader, device)}")
    print(f"checkpoint: {args.out}")


if __name__ == "__main__":
    main()
