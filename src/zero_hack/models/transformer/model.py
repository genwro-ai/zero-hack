from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_context: int = 192


class TransformerModel(nn.Module):
    def __init__(self, vocab_size: int, config: TransformerConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(config.max_context, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.head = nn.Linear(config.d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).clamp_(
            max=self.config.max_context - 1
        )

        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        ).to(torch.bool)
        src_key_padding_mask = ~attention_mask

        encoded = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        last_valid = torch.where(attention_mask, positions, 0).max(dim=1).values
        batch_idx = torch.arange(encoded.size(0), device=input_ids.device)
        last_hidden = encoded[batch_idx, last_valid, :]
        return self.head(last_hidden)
