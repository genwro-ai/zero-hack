import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, Vocabulary

IGNORE = -100


class CausalLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len, dropout):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, d_model)
        self.positions = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_model * 4,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.max_len = max_len
        nn.init.normal_(self.tokens.weight, std=0.02)
        nn.init.normal_(self.positions.weight, std=0.02)

    def forward(self, ids, attn_mask):
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.drop(self.tokens(ids) + self.positions(pos))
        causal = nn.Transformer.generate_square_subsequent_mask(ids.size(1), device=ids.device)
        hidden = self.norm(self.blocks(x, mask=causal.bool(), src_key_padding_mask=~attn_mask))
        return hidden @ self.tokens.weight.t()


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


def collate(batch, pad_id):
    width = max(len(ids) for ids in batch) - 1
    inputs, labels, mask = [], [], []
    for ids in batch:
        prefix = ids[:-1]
        pad = width - len(prefix)
        inputs.append(prefix + [pad_id] * pad)
        labels.append([IGNORE] + ids[2:] + [IGNORE] * pad)
        mask.append([1] * len(prefix) + [0] * pad)
    return torch.tensor(inputs), torch.tensor(labels), torch.tensor(mask, dtype=torch.bool)


def cosine_lr(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def fit(model, records, vocabulary, device, epochs, batch_size, lr):
    dataset = SequenceDataset(records, vocabulary, model.max_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, vocabulary.pad_id),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    total = len(loader) * epochs
    warmup = int(0.05 * total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine_lr(s, warmup, total))
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for inputs, labels, mask in loader:
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)
            loss = F.cross_entropy(
                model(inputs, mask).flatten(0, 1), labels.flatten(), ignore_index=IGNORE
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
        print(f"epoch {epoch + 1} train_loss={running / len(loader):.4f}")
    return model


def save_member(path, model, config, vocabulary):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "token_to_id": vocabulary.token_to_id,
            "id_to_token": list(vocabulary.id_to_token),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_member(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    vocab = Vocabulary(
        token_to_id=checkpoint["token_to_id"], id_to_token=tuple(checkpoint["id_to_token"])
    )
    model = CausalLM(
        len(vocab.id_to_token),
        config["d_model"],
        config["n_heads"],
        config["n_layers"],
        config["max_len"],
        config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, vocab, config


class EnsemblePredictor:
    def __init__(self, models, vocabulary, device, max_len):
        self.models = [model.to(device).eval() for model in models]
        self.vocabulary = vocabulary
        self.device = device
        self.max_len = max_len
        specials = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
        self.blocked = [vocabulary.token_to_id[t] for t in specials if t in vocabulary.token_to_id]

    def encode(self, family, steps):
        family_token = FAMILY_TOKENS.get(family.lower(), FAMILY_TOKENS["unknown"])
        tokens = ["<BOS>", family_token, *steps][-self.max_len :]
        ids = torch.tensor([self.vocabulary.encode(tokens)], device=self.device)
        return ids, torch.ones_like(ids, dtype=torch.bool)

    @torch.no_grad()
    def mean_probs(self, ids, mask):
        total = None
        for model in self.models:
            probs = F.softmax(model(ids, mask), dim=-1)
            total = probs if total is None else total + probs
        return total / len(self.models)

    @torch.no_grad()
    def predict_topk(self, family, prefix_steps, k):
        ids, mask = self.encode(family, list(prefix_steps))
        probs = self.mean_probs(ids, mask)[0, -1]
        probs[self.blocked] = 0.0
        top = torch.topk(probs, min(k, probs.numel())).indices.tolist()
        return [self.vocabulary.id_to_token[i] for i in top]

    @torch.no_grad()
    def score_sequence(self, family, steps):
        steps = list(steps)
        if not steps:
            return 0.0
        ids, mask = self.encode(family, steps)
        probs = self.mean_probs(ids, mask)[0, :-1]
        chosen = probs.gather(1, ids[0, 1:, None]).squeeze(1)
        return float(chosen[1:].clamp_min(1e-12).log().sum())
