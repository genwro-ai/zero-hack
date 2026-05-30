from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class LSTMConfig:
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.1


class LSTMModel(nn.Module):
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
        if attention_mask is not None:
            input_ids, lengths = _right_pad_from_mask(input_ids, attention_mask, self.pad_id)
        else:
            lengths = torch.full(
                (input_ids.size(0),),
                input_ids.size(1),
                dtype=torch.long,
                device=input_ids.device,
            )

        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        return self.head(self.dropout(hidden[-1]))


def _right_pad_from_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = attention_mask.sum(dim=1).clamp_min(1)
    right_padded = input_ids.new_full(input_ids.shape, pad_id)
    for row_idx, length in enumerate(lengths.tolist()):
        right_padded[row_idx, :length] = input_ids[row_idx, attention_mask[row_idx]]
    return right_padded, lengths
