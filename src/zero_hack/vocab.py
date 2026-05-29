"""Step vocabulary shared by every model (baselines, LSTM, Transformer)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

FAMILIES = ("mosfet", "igbt", "ic")


class Vocab:
    """Bidirectional mapping between step strings and integer ids.

    Special tokens always occupy the first four ids:
    ``<PAD>=0``, ``<BOS>=1``, ``<EOS>=2``, ``<UNK>=3``.
    """

    def __init__(self, tokens: list[str]) -> None:
        self.id_to_token = list(tokens)
        self.token_to_id = {tok: i for i, tok in enumerate(self.id_to_token)}

    # ---- construction ---------------------------------------------------
    @classmethod
    def build(cls, sequences: Iterable[list[str]]) -> Vocab:
        """Build a vocab from an iterable of step sequences (sorted for determinism)."""
        steps: set[str] = set()
        for seq in sequences:
            steps.update(seq)
        ordered = list(SPECIAL_TOKENS) + sorted(steps)
        return cls(ordered)

    # ---- lookups --------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def encode(self, steps: list[str]) -> list[int]:
        unk = self.unk_id
        return [self.token_to_id.get(s, unk) for s in steps]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids]

    # ---- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"id_to_token": self.id_to_token}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["id_to_token"])


FAMILY_TO_ID = {fam: i for i, fam in enumerate(FAMILIES)}
ID_TO_FAMILY = {i: fam for fam, i in FAMILY_TO_ID.items()}
