from dataclasses import asdict, dataclass

import torch
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


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.nhead,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normed = self.ln_attn(hidden)
        attended, _ = self.attn(
            normed,
            normed,
            normed,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        hidden = hidden + self.mlp(self.ln_mlp(hidden))
        return hidden


class GPTNextStepModel(nn.Module):
    """Small decoder-only model for categorical process-step next-token prediction."""

    def __init__(self, vocab_size: int, config: GPTConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(config.max_context, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(GPTBlock(config) for _ in range(config.num_layers))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size, bias=False)

        if config.tie_embeddings:
            self.head.weight = self.token_embedding.weight

    def forward_all(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for every input position."""

        _, seq_len = input_ids.shape
        if seq_len > self.config.max_context:
            input_ids = input_ids[:, -self.config.max_context :]
            attention_mask = attention_mask[:, -self.config.max_context :]
            seq_len = self.config.max_context

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        key_padding_mask = ~attention_mask

        for block in self.blocks:
            hidden = block(
                hidden,
                causal_mask=causal_mask,
                key_padding_mask=key_padding_mask,
            )

        hidden = self.ln_f(hidden)
        return self.head(hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward_all(input_ids, attention_mask)
        if input_ids.shape[1] > self.config.max_context:
            attention_mask = attention_mask[:, -self.config.max_context :]
        last_valid = attention_mask.long().sum(dim=1).clamp_min(1) - 1
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        return logits[batch_idx, last_valid, :]
