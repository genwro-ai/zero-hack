import random

import torch
from torch import nn

from zero_hack.data.datasets import FAMILY_TOKENS, SPECIAL_TOKENS, Vocabulary


def build_examples(records, vocab, max_len):
    examples = []
    for record in records:
        family = FAMILY_TOKENS[record.family]
        steps = list(record.steps)
        for i in range(len(steps) + 1):
            tokens = (["<BOS>", family] + steps[:i])[-max_len:]
            target = steps[i] if i < len(steps) else "<EOS>"
            examples.append((vocab.encode(tokens), vocab.token_to_id.get(target, vocab.unk_id)))
    return examples


def collate(batch, pad_id):
    width = max(len(ids) for ids, _ in batch)
    x, mask, y = [], [], []
    for ids, target in batch:
        pad = width - len(ids)
        x.append([pad_id] * pad + ids)
        mask.append([0] * pad + [1] * len(ids))
        y.append(target)
    return torch.tensor(x), torch.tensor(mask, dtype=torch.bool), torch.tensor(y)


class StepTransformer(nn.Module):
    def __init__(self, vocab, d_model=128, nhead=4, layers=2, max_len=256):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.cfg = {"d_model": d_model, "nhead": nhead, "layers": layers, "max_len": max_len}
        size = len(vocab.id_to_token)
        self.token = nn.Embedding(size, d_model, padding_idx=vocab.pad_id)
        self.position = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids, attention_mask):
        positions = torch.arange(input_ids.size(1), device=input_ids.device)
        x = self.token(input_ids) + self.position(positions)[None]
        x = self.encoder(x, src_key_padding_mask=~attention_mask)
        return self.head(x[:, -1])

    def fit(self, records, epochs=3, batch_size=128, lr=3e-4, seed=0):
        rng = random.Random(seed)
        torch.manual_seed(seed)
        data = build_examples(records, self.vocab, self.max_len)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        for epoch in range(epochs):
            rng.shuffle(data)
            running = 0.0
            for start in range(0, len(data), batch_size):
                x, mask, y = collate(data[start : start + batch_size], self.vocab.pad_id)
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self(x, mask), y)
                loss.backward()
                optimizer.step()
                running += loss.item()
            print(f"epoch {epoch + 1}  loss {running / (len(data) // batch_size + 1):.3f}")

    @torch.no_grad()
    def logprobs(self, prefix, family):
        self.eval()
        token = FAMILY_TOKENS.get(family, FAMILY_TOKENS["unknown"])
        tokens = (["<BOS>", token] + list(prefix))[-self.max_len :]
        ids = torch.tensor([self.vocab.encode(tokens)], device=self.device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        return torch.log_softmax(self(ids, mask)[0], dim=-1)

    def rank(self, prefix, family, k=5):
        banned = {self.vocab.token_to_id[t] for t in SPECIAL_TOKENS}
        banned |= {self.vocab.token_to_id[t] for t in FAMILY_TOKENS.values()}
        order = torch.argsort(self.logprobs(prefix, family), descending=True).tolist()
        return [self.vocab.id_to_token[i] for i in order if i not in banned][:k]

    def complete(self, prefix, family, max_steps=200):
        eos = self.vocab.token_to_id["<EOS>"]
        steps, produced = list(prefix), []
        for _ in range(max_steps):
            nxt = int(torch.argmax(self.logprobs(steps, family)))
            if nxt == eos:
                break
            step = self.vocab.id_to_token[nxt]
            produced.append(step)
            steps.append(step)
        return produced

    def save(self, path):
        checkpoint = {"tokens": self.vocab.id_to_token, "cfg": self.cfg, "state": self.state_dict()}
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        tokens = tuple(ckpt["tokens"])
        vocab = Vocabulary({t: i for i, t in enumerate(tokens)}, tokens)
        model = cls(vocab, **ckpt["cfg"])
        model.load_state_dict(ckpt["state"])
        return model
