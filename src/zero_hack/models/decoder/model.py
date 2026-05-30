from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, Vocabulary


@dataclass
class DecoderConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 256


class DecoderLM(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        block = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.n_heads,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device)
        x = self.drop(self.token_emb(ids) + self.pos_emb(pos))
        causal = nn.Transformer.generate_square_subsequent_mask(ids.size(1), device=ids.device)
        return self.head(self.norm(self.encoder(x, mask=causal)))


class Predictor:
    def __init__(self, model, vocab, device, max_seq_len):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.max_seq_len = max_seq_len
        specials = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
        self.blocked = [vocab.token_to_id[t] for t in specials if t in vocab.token_to_id]

    def _encode(self, family, steps):
        family_token = FAMILY_TOKENS.get(family, FAMILY_TOKENS["unknown"])
        tokens = ["<BOS>", family_token, *steps][-self.max_seq_len :]
        return torch.tensor([self.vocab.encode(tokens)], device=self.device)

    @torch.no_grad()
    def predict_topk(self, family, prefix, k=3):
        logits = self.model(self._encode(family, list(prefix)))[0, -1]
        logits[self.blocked] = -torch.inf
        idx = logits.topk(min(k, logits.numel())).indices.tolist()
        return [self.vocab.id_to_token[i] for i in idx]

    @torch.no_grad()
    def score_sequence(self, family, steps):
        steps = list(steps)
        if not steps:
            return 0.0
        ids = self._encode(family, steps)
        log_probs = F.log_softmax(self.model(ids)[0, :-1], dim=-1)
        chosen = log_probs.gather(1, ids[0, 1:, None]).squeeze(1)
        return float(chosen[1:].sum())


def save_checkpoint(path, model, cfg, vocab):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(cfg),
            "token_to_id": vocab.token_to_id,
            "id_to_token": list(vocab.id_to_token),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_predictor(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = DecoderConfig(**ckpt["config"])
    vocab = Vocabulary(ckpt["token_to_id"], tuple(ckpt["id_to_token"]))
    model = DecoderLM(len(vocab.id_to_token), cfg)
    model.load_state_dict(ckpt["state_dict"])
    return Predictor(model, vocab, device, cfg.max_seq_len)
