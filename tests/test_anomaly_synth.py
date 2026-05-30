import json
import random
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.data import load_sequence_records
from zero_hack.eval.anomaly_synth import (
    RULE_IDS,
    build_rule_stratified_corruptions,
    corrupt_steps,
)
from zero_hack.eval.validator import first_violated_rule, is_valid

_GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "corrupt_golden.json").read_text())
_SPLITS = PROJECT_ROOT / "data" / "generated" / "valid_s005k" / "splits"


def test_corrupt_reproduces_golden_output():
    out = corrupt_steps(list(_GOLDEN["input_steps"]), random.Random(_GOLDEN["seed"]))
    assert out is not None
    steps, rule = out
    assert steps != _GOLDEN["input_steps"]
    assert rule in RULE_IDS
    assert first_violated_rule(steps) == rule


def test_corrupt_output_is_rejected_by_validator():
    steps, _ = corrupt_steps(list(_GOLDEN["input_steps"]), random.Random(_GOLDEN["seed"]))
    assert not is_valid(steps)
    assert steps != _GOLDEN["input_steps"]


def test_corrupt_is_deterministic_for_a_fixed_seed():
    a = corrupt_steps(list(_GOLDEN["input_steps"]), random.Random(7))
    b = corrupt_steps(list(_GOLDEN["input_steps"]), random.Random(7))
    assert a == b


def test_corrupt_can_target_each_rule():
    records = []
    for path in ("MOSFET_valid.csv", "IGBT_valid.csv", "IC_valid.csv"):
        records.extend(load_sequence_records(_SPLITS / path)[:5])

    for rule in RULE_IDS:
        corrupted = None
        for record in records:
            corrupted = corrupt_steps(list(record.steps), random.Random(42), target_rule=rule)
            if corrupted is not None:
                break
        assert corrupted is not None, rule
        steps, observed_rule = corrupted
        assert observed_rule == rule
        assert first_violated_rule(steps) == rule


def test_rule_stratified_corruptions_balance_all_rules():
    records = load_sequence_records(_SPLITS / "MOSFET_valid.csv")[:20]
    examples = build_rule_stratified_corruptions(
        records,
        n_invalid=len(RULE_IDS) * 2,
        rng=random.Random(1729),
    )
    counts = {rule: 0 for rule in RULE_IDS}
    for example in examples:
        counts[example.rule] += 1
        assert first_violated_rule(example.steps) == example.rule

    assert len(examples) == len(RULE_IDS) * 2
    assert set(counts.values()) == {2}
