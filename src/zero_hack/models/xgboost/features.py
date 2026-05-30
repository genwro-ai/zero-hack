import re

import numpy as np
from zero_hack.vocab import FAMILY_TO_ID

CATEGORIES = (
    "logistics",
    "inspect",
    "measure",
    "clean",
    "deposit",
    "litho",
    "develop",
    "etch",
    "strip",
    "implant",
    "anneal",
    "cmp",
    "fill",
    "passivation",
    "test",
    "other",
)
CATEGORY_INDEX = {name: idx for idx, name in enumerate(CATEGORIES)}

UNKNOWN_FAMILY_ID = len(FAMILY_TO_ID)
SINCE_CAP = 64
_ALIGN_LEVEL = re.compile(r"LEVEL\s+(\d+)")


def categorize(step: str) -> str:
    """Map a step name to a coarse functional category."""
    s = step.upper()
    if "STRIP" in s and "RESIST" in s:
        return "strip"
    clean = ("CLEAN", "RINSE", "HF DIP", "DRY WAFER", "OXIDE STRIP", "SURFACE PREP")
    if any(token in s for token in clean):
        return "clean"
    if "ETCH" in s:
        return "etch"
    if "DEVELOP" in s:
        return "develop"
    if "IMPLANT" in s:
        return "implant"
    if "CMP" in s:
        return "cmp"
    if "FILL VIA" in s:
        return "fill"
    deposit = ("DEPOSIT", "OXIDATION", "GROWTH", "EPITAXIAL DEPOSITION", "DENSIFY")
    if any(token in s for token in deposit):
        return "deposit"
    if "ANNEAL" in s or "DRIVE IN" in s or "RAPID THERMAL" in s or "RTA" in s:
        return "anneal"
    if "SPIN COAT" in s or "SOFT BAKE" in s or "ALIGN MASK" in s or "EXPOSE" in s or "BAKE" in s:
        return "litho"
    if "INSPECT" in s:
        return "inspect"
    if "MEASURE" in s or "CHECK" in s:
        return "measure"
    if "TEST" in s or "YIELD" in s or "PARAMETRIC" in s or "SORT" in s:
        return "test"
    if "PASSIVATION" in s:
        return "passivation"
    if "RECEIVE" in s or "LOT" in s or "SHIP" in s or "RELEASE" in s or "PACKAGE" in s:
        return "logistics"
    return "other"


class _State:
    def __init__(self, family: str, lag: int) -> None:
        self.family_id = FAMILY_TO_ID.get(family.lower(), UNKNOWN_FAMILY_ID)
        self.position = 0
        self.lag = [-1] * lag
        self.last_cat = -1
        self.last_cat_pos = [-1] * len(CATEGORIES)
        self.count = [0] * len(CATEGORIES)
        self.max_align = 0
        self.has_cure_passivation = 0
        self.has_deposit_passivation = 0
        self.has_wafer_sort = 0
        self.has_pad_open = 0


class FeatureExtractor:
    """Turns (family, prefix) into a fixed-length numeric feature row."""

    def __init__(self, lag: int = 8, since_cap: int = SINCE_CAP) -> None:
        self.lag = lag
        self.since_cap = since_cap
        self.step_to_code: dict[str, int] = {}

    def set_vocab(self, steps: list[str]) -> None:
        self.step_to_code = {step: idx for idx, step in enumerate(steps)}

    @property
    def n_features(self) -> int:
        return self.lag + 3 + 2 * len(CATEGORIES) + 5

    def _code(self, step: str) -> int:
        return self.step_to_code.get(step, -1)

    def _row(self, state: _State) -> list[float]:
        row = list(state.lag)
        row.append(state.family_id)
        row.append(state.position)
        row.append(state.last_cat)
        for last_pos in state.last_cat_pos:
            if last_pos < 0:
                row.append(self.since_cap)
            else:
                row.append(min(self.since_cap, state.position - last_pos))
        row.extend(state.count)
        row.append(state.max_align)
        row.append(state.has_cure_passivation)
        row.append(state.has_deposit_passivation)
        row.append(state.has_wafer_sort)
        row.append(state.has_pad_open)
        return row

    def _advance(self, state: _State, step: str) -> None:
        category = CATEGORY_INDEX[categorize(step)]
        state.last_cat = category
        state.last_cat_pos[category] = state.position
        state.count[category] += 1
        state.lag = [self._code(step)] + state.lag[:-1]

        s = step.upper()
        if "ALIGN MASK" in s:
            match = _ALIGN_LEVEL.search(s)
            if match:
                state.max_align = max(state.max_align, int(match.group(1)))
        if s == "CURE PASSIVATION":
            state.has_cure_passivation = 1
        if "DEPOSIT" in s and "PASSIVATION" in s:
            state.has_deposit_passivation = 1
        if s == "WAFER SORT TEST":
            state.has_wafer_sort = 1
        if "PAD WINDOW" in s or "OPEN PAD WINDOW" in s or "OPEN BOND PAD WINDOW" in s:
            state.has_pad_open = 1
        state.position += 1

    def sequence_matrix(self, family: str, steps: list[str] | tuple[str, ...]) -> np.ndarray:
        state = _State(family, self.lag)
        rows = []
        for step in steps:
            rows.append(self._row(state))
            self._advance(state, step)
        if not rows:
            return np.zeros((0, self.n_features), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def prefix_row(self, family: str, prefix_steps: list[str] | tuple[str, ...]) -> np.ndarray:
        state = _State(family, self.lag)
        for step in prefix_steps:
            self._advance(state, step)
        return np.asarray(self._row(state), dtype=np.float32)
