"""A small LSTM for next-step prediction.

The forward contract matches ``zero_hack.models.common``: inputs are
*left-padded* ``[B, T]`` token ids plus a boolean ``attention_mask``
(``True`` for real tokens). Because padding is on the left, the real final
step is always at index ``-1``, so we run the LSTM over the whole sequence
and take the output at the last timestep, returning next-step logits of
shape ``[B, vocab_size]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LSTMConfig:
    """Small defaults for initial experiments."""

    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1


class LSTMModel(nn.Module):
    """Recurrent encoder with a next-step linear head."""

    def __init__(self, vocab_size: int, config: LSTMConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_dim, vocab_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # attention_mask is unused: left padding puts the real final step at -1.
        embedded = self.embedding(input_ids)
        output, _ = self.lstm(embedded)

        # Inputs are left-padded, so the real final step is always at index -1.
        last_hidden = output[:, -1, :]
        return self.head(self.dropout(last_hidden))
