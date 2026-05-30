from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, Vocabulary
from zero_hack.models.gpt import GPTConfig, GPTNextStepModel

TERMINATOR = "SHIP LOT"


@dataclass
class GFlowNetConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_context: int = 192
    families: tuple[str, ...] = ("mosfet", "igbt", "ic")

    def to_gpt_config(self) -> GPTConfig:
        return GPTConfig(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            max_context=self.max_context,
        )

    def to_dict(self) -> dict:
        return asdict(self)


class GFlowNetPolicy(nn.Module):
    """Prefix-conditioned forward policy plus family-conditioned log partition."""

    def __init__(self, vocab_size: int, config: GFlowNetConfig, pad_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id
        self.forward_policy = GPTNextStepModel(vocab_size, config.to_gpt_config(), pad_id=pad_id)
        self.family_to_id = {family: idx for idx, family in enumerate(config.families)}
        self.log_z_global = nn.Parameter(torch.tensor(0.0))
        self.log_z_family = nn.Embedding(len(config.families), 1)
        nn.init.zeros_(self.log_z_family.weight)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.forward_policy(input_ids, attention_mask)

    def log_z(self, family_ids: torch.Tensor) -> torch.Tensor:
        return self.log_z_global + self.log_z_family(family_ids).squeeze(-1)

    def family_ids(
        self, families: list[str] | tuple[str, ...], device: torch.device
    ) -> torch.Tensor:
        ids = [self.family_to_id.get(family.lower(), 0) for family in families]
        return torch.tensor(ids, dtype=torch.long, device=device)


def invalid_action_ids(vocabulary: Vocabulary) -> list[int]:
    invalid = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
    return [vocabulary.token_to_id[token] for token in invalid if token in vocabulary.token_to_id]


def encode_prefix(
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str] | tuple[str, ...],
    *,
    max_context: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    family_token = FAMILY_TOKENS.get(family.lower(), FAMILY_TOKENS["unknown"])
    tokens = ["<BOS>", family_token, *prefix][-max_context:]
    input_ids = torch.tensor([vocabulary.encode(tokens)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    return input_ids, attention_mask


def mask_logits(
    logits: torch.Tensor,
    *,
    vocabulary: Vocabulary,
    current_length: int,
    min_length: int,
    invalid_ids: list[int] | None = None,
    terminator: str = TERMINATOR,
) -> torch.Tensor:
    masked = logits.clone()
    if invalid_ids:
        masked[..., torch.tensor(invalid_ids, device=masked.device)] = -torch.inf
    terminator_id = vocabulary.token_to_id.get(terminator)
    if terminator_id is not None and current_length < min_length:
        masked[..., terminator_id] = -torch.inf
    return masked


@torch.no_grad()
def sample_completion(
    model: GFlowNetPolicy,
    vocabulary: Vocabulary,
    family: str,
    prefix: list[str] | tuple[str, ...],
    *,
    device: torch.device,
    min_length: int = 100,
    max_length: int = 200,
    temperature: float = 1.0,
    invalid_ids: list[int] | None = None,
    terminator: str = TERMINATOR,
) -> list[str]:
    """Sample a continuation from an existing trajectory prefix."""

    model.eval()
    steps = list(prefix)
    continuation: list[str] = []
    if steps and steps[-1] == terminator:
        return continuation

    invalid_ids = invalid_ids if invalid_ids is not None else invalid_action_ids(vocabulary)
    while len(steps) < max_length:
        input_ids, attention_mask = encode_prefix(
            vocabulary,
            family,
            steps,
            max_context=model.config.max_context,
            device=device,
        )
        logits = model(input_ids, attention_mask).squeeze(0)
        logits = mask_logits(
            logits,
            vocabulary=vocabulary,
            current_length=len(steps),
            min_length=min_length,
            invalid_ids=invalid_ids,
            terminator=terminator,
        )
        action_id = int(torch.distributions.Categorical(logits=logits / temperature).sample())
        action = vocabulary.id_to_token[action_id]
        steps.append(action)
        continuation.append(action)
        if action == terminator:
            break
    return continuation
