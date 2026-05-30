import json
import random
from pathlib import Path

from zero_hack.eval.anomaly_synth import corrupt_steps
from zero_hack.eval.validator import is_valid

_GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "corrupt_golden.json").read_text())


def test_corrupt_reproduces_golden_output():
    out = corrupt_steps(list(_GOLDEN["input_steps"]), random.Random(_GOLDEN["seed"]))
    assert out is not None
    steps, rule = out
    assert steps == _GOLDEN["expected_steps"]
    assert rule == _GOLDEN["expected_rule"]


def test_corrupt_output_is_rejected_by_validator():
    steps, _ = corrupt_steps(list(_GOLDEN["input_steps"]), random.Random(_GOLDEN["seed"]))
    assert not is_valid(steps)
    assert steps != _GOLDEN["input_steps"]


def test_corrupt_is_deterministic_for_a_fixed_seed():
    a = corrupt_steps(list(_GOLDEN["input_steps"]), random.Random(7))
    b = corrupt_steps(list(_GOLDEN["input_steps"]), random.Random(7))
    assert a == b
