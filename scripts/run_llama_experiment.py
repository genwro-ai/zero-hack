#!/usr/bin/env python3
import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaModel

from zero_hack import PROJECT_ROOT
from zero_hack.data import SPECIAL_TOKENS, Vocabulary
from zero_hack.eval import io
from zero_hack.eval.score import score_task
from zero_hack.eval.validator import validate_sequence
from zero_hack.models.anomaly_threshold import tune_anomaly_threshold
from zero_hack.models.classic_baselines import SEQUENCE_TERMINATOR, predict_anomaly
from zero_hack.models.common import load_split_records, pick_device

FAMILIES = ("mosfet", "igbt", "ic")
SPECIAL = set(SPECIAL_TOKENS)
METRICS = {
    "next_step": ("top1", "top3", "top5", "mrr"),
    "anomaly": ("accuracy", "f1", "roc_auc"),
    "completion_off": ("block_accuracy", "token_accuracy", "norm_edit_distance"),
    "completion_on": ("block_accuracy", "token_accuracy", "norm_edit_distance"),
}


def build_step_vocab(records):
    steps = set()
    for record in records:
        steps.update(record.steps)
    tokens = list(SPECIAL_TOKENS) + sorted(steps)
    return Vocabulary(token_to_id={t: i for i, t in enumerate(tokens)}, id_to_token=tuple(tokens))


class StepEmbedder(nn.Module):
    def __init__(self, step_tokens, dim):
        super().__init__()
        self.n_special = len(SPECIAL_TOKENS)
        body = step_tokens[self.n_special :]
        words = sorted({w for t in body for w in t.split()})
        chars = sorted({c for t in body for c in t})
        word_to_id = {w: i + 1 for i, w in enumerate(words)}
        char_to_id = {c: i + 1 for i, c in enumerate(chars)}
        max_w = max((len(t.split()) for t in body), default=1)
        max_c = max((len(t) for t in body), default=1)
        n = len(step_tokens)
        word_ids = torch.zeros(n, max_w, dtype=torch.long)
        word_mask = torch.zeros(n, max_w)
        char_ids = torch.zeros(n, max_c, dtype=torch.long)
        char_mask = torch.zeros(n, max_c)
        for i, token in enumerate(step_tokens):
            if i < self.n_special:
                continue
            for j, word in enumerate(token.split()):
                word_ids[i, j] = word_to_id[word]
                word_mask[i, j] = 1.0
            for j, char in enumerate(token):
                char_ids[i, j] = char_to_id[char]
                char_mask[i, j] = 1.0
        self.register_buffer("word_ids", word_ids)
        self.register_buffer("word_mask", word_mask)
        self.register_buffer("char_ids", char_ids)
        self.register_buffer("char_mask", char_mask)
        self.word_emb = nn.Embedding(len(words) + 1, dim, padding_idx=0)
        self.char_emb = nn.Embedding(len(chars) + 1, dim, padding_idx=0)
        self.special = nn.Embedding(self.n_special, dim)
        for emb in (self.word_emb, self.char_emb, self.special):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self):
        words = self.word_emb(self.word_ids)
        word_mask = self.word_mask.unsqueeze(-1)
        word_pooled = (words * word_mask).sum(1) / word_mask.sum(1).clamp(min=1.0)
        chars = self.char_emb(self.char_ids)
        char_mask = self.char_mask.unsqueeze(-1)
        char_pooled = (chars * char_mask).sum(1) / char_mask.sum(1).clamp(min=1.0)
        composed = word_pooled + char_pooled
        special = self.special(torch.arange(self.n_special, device=composed.device))
        return torch.cat([special, composed[self.n_special :]], dim=0)


class CompositionalLlama(nn.Module):
    def __init__(self, step_tokens, hidden, layers, heads, ffn, max_context):
        super().__init__()
        config = LlamaConfig(
            vocab_size=len(step_tokens),
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            intermediate_size=ffn,
            max_position_embeddings=max_context,
            rope_theta=10000.0,
        )
        self.backbone = LlamaModel(config)
        self.embedder = StepEmbedder(step_tokens, hidden)
        self.max_context = max_context

    def sequence_logits(self, input_ids, attention_mask):
        if input_ids.size(1) > self.max_context:
            input_ids = input_ids[:, -self.max_context :]
            attention_mask = attention_mask[:, -self.max_context :]
        embeddings = self.embedder()
        hidden = self.backbone(
            inputs_embeds=embeddings[input_ids], attention_mask=attention_mask
        ).last_hidden_state
        return hidden @ embeddings.t()

    def forward(self, input_ids, attention_mask):
        logits = self.sequence_logits(input_ids, attention_mask)
        last = attention_mask.long().sum(1).clamp_min(1) - 1
        return logits[torch.arange(logits.size(0), device=logits.device), last]


class SeqDataset(Dataset):
    def __init__(self, records, vocabulary, max_len):
        self.rows = []
        for record in records:
            ids = vocabulary.encode(["<BOS>", *record.steps, "<EOS>"])[:max_len]
            if len(ids) >= 3:
                self.rows.append(ids)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch, pad_id):
    width = max(len(ids) for ids in batch)
    input_ids, attention = [], []
    for ids in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention.append([1] * len(ids) + [0] * pad)
    return torch.tensor(input_ids), torch.tensor(attention, dtype=torch.bool)


def _loss(model, input_ids, attention, pad_id):
    logits = model.sequence_logits(input_ids, attention)
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        ignore_index=pad_id,
    )


def _valid_loss(model, loader, device, pad_id):
    model.eval()
    use_cuda = device.type == "cuda"
    total, seen = 0.0, 0
    with torch.no_grad():
        for input_ids, attention in loader:
            input_ids, attention = input_ids.to(device), attention.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                total += float(_loss(model, input_ids, attention, pad_id))
            seen += 1
    return total / max(1, seen)


def train_llama(
    model,
    train_records,
    valid_records,
    vocabulary,
    device,
    epochs,
    batch_size,
    lr,
    patience,
    num_workers,
    min_delta,
):
    model = model.to(device)
    use_cuda = device.type == "cuda"
    pad_id = vocabulary.pad_id
    coll = partial(collate, pad_id=pad_id)
    train_loader = DataLoader(
        SeqDataset(train_records, vocabulary, model.max_context),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=coll,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    valid_loader = DataLoader(
        SeqDataset(valid_records, vocabulary, model.max_context),
        batch_size=batch_size,
        collate_fn=coll,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=num_workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    total = max(1, len(train_loader) * epochs)
    warmup = max(1, int(0.05 * total))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best, best_state, stale = math.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for input_ids, attention in train_loader:
            input_ids, attention = input_ids.to(device), attention.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                loss = _loss(model, input_ids, attention, pad_id)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())
        valid = _valid_loss(model, valid_loader, device, pad_id)
        train_loss = running / max(1, len(train_loader))
        print(f"  epoch {epoch}/{epochs} train={train_loss:.4f} valid={valid:.4f}", flush=True)
        if valid < best - min_delta:
            best = valid
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop epoch {epoch} best={best:.4f}", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class LlamaAdapter:
    def __init__(self, model, vocabulary, device, grammar_topk, max_new, grammar_nextstep):
        self.model = model.to(device).eval()
        self.vocabulary = vocabulary
        self.device = device
        self.grammar_topk = grammar_topk
        self.max_new = max_new
        self.grammar_nextstep = grammar_nextstep
        self.max_len = model.max_context
        self.blocked = [vocabulary.token_to_id[t] for t in SPECIAL_TOKENS]
        self.suppress = [vocabulary.token_to_id[t] for t in ("<PAD>", "<BOS>", "<UNK_STEP>")]

    def _ids(self, tokens):
        tokens = tokens[-self.max_len :]
        return torch.tensor([self.vocabulary.encode(tokens)], device=self.device)

    def _violates(self, steps, token):
        index = len(steps)
        return any(v.step_index == index for v in validate_sequence([*steps, token]))

    @torch.no_grad()
    def predict_topk(self, family, prefix_steps, k=3):
        prefix = list(prefix_steps)
        ids = self._ids(["<BOS>", *prefix])
        logits = self.model(ids, torch.ones_like(ids, dtype=torch.bool))[0]
        logits[self.blocked] = -torch.inf
        if not self.grammar_nextstep:
            idx = torch.topk(logits, min(k, logits.numel())).indices.tolist()
            return [self.vocabulary.id_to_token[i] for i in idx]
        order = torch.topk(logits, min(self.grammar_topk, logits.numel())).indices.tolist()
        legal = [i for i in order if not self._violates(prefix, self.vocabulary.id_to_token[i])]
        ranked = legal if legal else order
        return [self.vocabulary.id_to_token[i] for i in ranked[:k]]

    @torch.no_grad()
    def score_sequence(self, family, steps):
        steps = list(steps)
        if not steps:
            return 0.0
        ids = self._ids(["<BOS>", *steps, "<EOS>"])
        log_probs = F.log_softmax(
            self.model.sequence_logits(ids, torch.ones_like(ids, dtype=torch.bool))[0, :-1], dim=-1
        )
        chosen = log_probs.gather(1, ids[0, 1:, None]).squeeze(1)
        return float(chosen[: len(steps)].sum())

    @torch.no_grad()
    def complete(self, family, prefix_steps, grammar_on):
        steps = list(prefix_steps)
        produced = []
        for _ in range(self.max_new):
            ids = self._ids(["<BOS>", *steps])
            logits = self.model(ids, torch.ones_like(ids, dtype=torch.bool))[0].clone()
            logits[self.suppress] = -torch.inf
            order = torch.argsort(logits, descending=True).tolist()
            if grammar_on:
                legal = [
                    t
                    for t in order[: self.grammar_topk]
                    if self.vocabulary.id_to_token[t] == "<EOS>"
                    or not self._violates(steps, self.vocabulary.id_to_token[t])
                ]
                chosen = legal[0] if legal else order[0]
            else:
                chosen = order[0]
            token = self.vocabulary.id_to_token[chosen]
            if token == "<EOS>":
                break
            steps.append(token)
            produced.append(token)
            if token == SEQUENCE_TERMINATOR:
                break
        return produced


def evaluate(adapter, eval_dir, out_dir, threshold):
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_inputs = io.read_eval_input_valid(eval_dir / "eval_input_valid.csv")
    io.write_next_step_predictions(
        out_dir / "nextstep.csv",
        [
            {
                "example_id": r["example_id"],
                "ranks": adapter.predict_topk(r["family"], r["partial_sequence"], 5),
            }
            for r in valid_inputs
        ],
    )
    anomaly_inputs = io.read_eval_input_anomaly(eval_dir / "eval_input_anomaly.csv")
    io.write_anomaly_predictions(
        out_dir / "anomaly.csv",
        [
            {
                "example_id": r["example_id"],
                **predict_anomaly(adapter, r["family"], r["sequence"], "hybrid", threshold),
            }
            for r in anomaly_inputs
        ],
    )
    for tag, grammar_on in (("off", False), ("on", True)):
        io.write_completion_predictions(
            out_dir / f"completion_{tag}.csv",
            [
                {
                    "example_id": r["example_id"],
                    "steps": adapter.complete(r["family"], r["partial_sequence"], grammar_on),
                }
                for r in valid_inputs
            ],
        )
    results = {
        "next_step": score_task(
            "next_step",
            ground_truth=eval_dir / "nextstep_truth.csv",
            predictions=out_dir / "nextstep.csv",
            eval_input=eval_dir / "eval_input_valid.csv",
        ),
        "anomaly": score_task(
            "anomaly",
            ground_truth=eval_dir / "anomaly_truth.csv",
            predictions=out_dir / "anomaly.csv",
            eval_input=eval_dir / "eval_input_anomaly.csv",
        ),
    }
    for tag in ("off", "on"):
        results[f"completion_{tag}"] = score_task(
            "completion",
            ground_truth=eval_dir / "completion_truth.csv",
            predictions=out_dir / f"completion_{tag}.csv",
            eval_input=eval_dir / "eval_input_valid.csv",
        )
    return results


def fmt(value):
    return f"{value:16.3f}" if isinstance(value, (int, float)) else f"{'-':>16}"


def print_table(by_view):
    for task, keys in METRICS.items():
        print(f"== {task} ==")
        print(f"{'':5s}" + "".join(f"{key:>16}" for key in keys))
        for view in ("id", "ood"):
            block = by_view[view][task]
            block = block.get("all", block)
            print(f"{view:5s}" + "".join(fmt(block.get(key)) for key in keys))


def average(folds):
    avg = {}
    for view in ("id", "ood"):
        avg[view] = {}
        for task, keys in METRICS.items():
            avg[view][task] = {}
            for key in keys:
                vals = []
                for fold in folds.values():
                    block = fold[view][task]
                    block = block.get("all", block)
                    if isinstance(block.get(key), (int, float)):
                        vals.append(block[key])
                avg[view][task][key] = sum(vals) / len(vals) if vals else None
    return avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", default=str(PROJECT_ROOT / "data" / "generated"))
    parser.add_argument("--eval-root", default=str(PROJECT_ROOT / "data" / "eval"))
    parser.add_argument("--datasets", nargs="+", default=["valid_s100k"])
    parser.add_argument("--holdout-families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--limit-per-family", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--grammar-nextstep", action="store_true")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=1024)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--grammar-topk", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--val-anomaly-valid", type=int, default=200)
    parser.add_argument("--val-anomaly-invalid", type=int, default=129)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "llama"))
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(None)
    out_root = Path(args.out)
    for dataset in args.datasets:
        splits_dir = Path(args.generated_root) / dataset / "splits"
        folds = {}
        for holdout in args.holdout_families:
            print(f"\n== dataset={dataset} holdout={holdout} ==", flush=True)
            torch.manual_seed(args.seed)
            bundle = load_split_records(
                splits_dir, holdout_family=holdout, limit_per_family=args.limit_per_family
            )
            inventory = [record for records in bundle.records.values() for record in records]
            vocabulary = build_step_vocab(inventory)
            print(
                f"step_vocab={len(vocabulary.id_to_token)} train={len(bundle.records['train'])}",
                flush=True,
            )
            model = CompositionalLlama(
                vocabulary.id_to_token,
                args.hidden,
                args.layers,
                args.heads,
                args.ffn,
                args.max_context,
            )
            model = train_llama(
                model,
                bundle.records["train"],
                bundle.records["valid"],
                vocabulary,
                device,
                args.epochs,
                args.batch_size,
                args.lr,
                args.patience,
                args.num_workers,
                args.min_delta,
            )
            adapter = LlamaAdapter(
                model,
                vocabulary,
                device,
                args.grammar_topk,
                args.max_new_tokens,
                args.grammar_nextstep,
            )
            threshold = tune_anomaly_threshold(
                adapter,
                bundle.records["valid"],
                n_valid=args.val_anomaly_valid,
                n_invalid=args.val_anomaly_invalid,
                seed=args.seed,
            ).threshold
            print(f"--> anomaly threshold={threshold:.4f}", flush=True)
            by_view = {}
            for view in ("id", "ood"):
                eval_dir = Path(args.eval_root) / dataset / f"holdout_{holdout}" / view
                if not eval_dir.exists():
                    raise SystemExit(
                        f"missing eval dir {eval_dir}; run local_scripts/make_eval_sets.sh"
                    )
                print(f"--> evaluating {view}", flush=True)
                by_view[view] = evaluate(
                    adapter, eval_dir, out_root / dataset / f"holdout_{holdout}" / view, threshold
                )
            folds[holdout] = by_view
            result_dir = out_root / dataset / f"holdout_{holdout}"
            (result_dir / "results.json").write_text(
                json.dumps(by_view, indent=2) + "\n", encoding="utf-8"
            )
            print(f"\n#### {dataset} | holdout {holdout} ####")
            print_table(by_view)
        averaged = average(folds)
        (out_root / dataset / "average.json").write_text(
            json.dumps(averaged, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n#### {dataset} | AVERAGE over {len(folds)} folds ####")
        print_table(averaged)


if __name__ == "__main__":
    main()
