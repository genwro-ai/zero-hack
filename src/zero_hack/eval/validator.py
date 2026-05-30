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


@lru_cache(maxsize=1)
def trigger_steps() -> frozenset[str]:
    """Steps that can *ever* trigger a process-logic rule.

    A next-step that is not one of these can never introduce a violation, so a
    masking layer only has to test these candidates. Sourced directly from the
    rule sets in ``generate_sequences.py`` (plus the ``ALIGN MASK LEVEL N``
    family, matched by prefix) so nothing about the rules is duplicated.
    """
    m = _generator_module()
    return frozenset(
        m.DEPOSITION_STEPS
        | m.ETCH_STEPS
        | m.IMPLANT_STEPS
        | m.CMP_STEPS
        | m.PAD_WINDOW_STEPS
        | m.ELECTRICAL_TEST_STEPS
        | m.BACKSIDE_METAL_STEPS
        | {"SHIP LOT"}
    )
