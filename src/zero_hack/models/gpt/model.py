from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_context: int = 192
    tie_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _alibi_slopes(nhead, device):
    start = 2.0 ** (-8.0 / nhead)
    return torch.tensor([start ** (i + 1) for i in range(nhead)], device=device)


class AlibiAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, hidden, attn_mask):
        batch, seq_len, _ = hidden.shape
        qkv = self.qkv(hidden).reshape(batch, seq_len, 3, self.nhead, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        attended = attended.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.proj(attended)


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.attn = AlibiAttention(config)
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden, attn_mask):
        hidden = hidden + self.attn(self.ln_attn(hidden), attn_mask)
        hidden = hidden + self.mlp(self.ln_mlp(hidden))
        return hidden


class GPTNextStepModel(nn.Module):
    """Decoder-only model with ALiBi attention for process-step next-token prediction."""

    def __init__(self, vocab_size: int, config: GPTConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(GPTBlock(config) for _ in range(config.num_layers))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.token_embedding.weight

    def _hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = input_ids.size(1)
        if seq_len > self.config.max_context:
            input_ids = input_ids[:, -self.config.max_context :]
            attention_mask = attention_mask[:, -self.config.max_context :]
            seq_len = input_ids.size(1)

        device = input_ids.device
        hidden = self.dropout(self.token_embedding(input_ids))

        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        keep = causal[None, None] & attention_mask[:, None, None, :]
        positions = torch.arange(seq_len, device=device)
        dist = (positions[:, None] - positions[None, :]).float()
        bias = (-_alibi_slopes(self.config.nhead, device)[:, None, None] * dist).unsqueeze(0)
        blocked = torch.zeros_like(keep, dtype=torch.float).masked_fill(~keep, -torch.inf)
        attn_mask = bias + blocked

        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        return self.ln_f(hidden), attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden, attention_mask = self._hidden(input_ids, attention_mask)
        last_valid = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return self.head(hidden[batch_idx, last_valid, :])

    def sequence_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden, _ = self._hidden(input_ids, attention_mask)
        return self.head(hidden)
