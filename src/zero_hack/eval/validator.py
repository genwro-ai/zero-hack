"""Bridge to the canonical process-rule validator.

The 10 forbidden-pattern rules are implemented once, in the track's
``data/industrial/generate_sequences.py`` (``validate_sequence``). We load that
module by path rather than re-implementing the rules, so the validator-oracle
anomaly baseline and the rule-attribution metric stay in lock-step with the
authoritative grammar.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from types import ModuleType

from zero_hack import INDUSTRIAL_DATA_DIR

_GENERATOR_PATH = INDUSTRIAL_DATA_DIR / "generate_sequences.py"


@lru_cache(maxsize=1)
def _generator_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("zero_hack_generate_sequences", _GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load validator from {_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_sequence(steps: list[str]) -> list:
    """Return the list of :class:`Violation` objects for ``steps`` (empty = valid)."""
    return _generator_module().validate_sequence(list(steps))


def is_valid(steps: list[str]) -> bool:
    """True when the sequence triggers none of the 10 process-logic rules."""
    return not validate_sequence(steps)


def first_violated_rule(steps: list[str]) -> str | None:
    """Rule id of the earliest violation (by step index), or ``None`` if valid."""
    violations = validate_sequence(steps)
    if not violations:
        return None
    earliest = min(violations, key=lambda v: v.step_index)
    return earliest.rule
