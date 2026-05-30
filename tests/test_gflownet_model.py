from types import SimpleNamespace

import torch

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, Vocabulary
from zero_hack.models.gflownet.model import sample_completion


class FakePolicy:
    def __init__(self, ship_id: int) -> None:
        self.config = SimpleNamespace(max_context=32)
        self.ship_id = ship_id

    def eval(self) -> None:
        return None

    def __call__(self, input_ids, attention_mask):
        logits = torch.full((1, 10), -100.0, device=input_ids.device)
        logits[0, self.ship_id] = 100.0
        return logits


def test_sample_completion_continues_existing_prefix_with_suffix_only():
    tokens = list(SPECIAL_TOKENS) + list(FAMILY_TOKENS.values()) + ["STEP A", "SHIP LOT"]
    vocabulary = Vocabulary(
        token_to_id={token: idx for idx, token in enumerate(tokens)},
        id_to_token=tuple(tokens),
    )
    model = FakePolicy(vocabulary.token_to_id["SHIP LOT"])

    continuation = sample_completion(
        model,
        vocabulary,
        "ic",
        ["STEP A"],
        device=torch.device("cpu"),
        min_length=1,
        max_length=5,
    )

    assert continuation == ["SHIP LOT"]


def test_sample_completion_returns_empty_for_finished_prefix():
    tokens = list(SPECIAL_TOKENS) + list(FAMILY_TOKENS.values()) + ["SHIP LOT"]
    vocabulary = Vocabulary(
        token_to_id={token: idx for idx, token in enumerate(tokens)},
        id_to_token=tuple(tokens),
    )
    model = FakePolicy(vocabulary.token_to_id["SHIP LOT"])

    continuation = sample_completion(
        model,
        vocabulary,
        "ic",
        ["SHIP LOT"],
        device=torch.device("cpu"),
    )

    assert continuation == []
