"""Behaviour tests for the neurosymbolic process-sequence generator.

The generator must produce sequences that are (a) valid under the official
10-rule validator, (b) full-vocabulary, (c) diverse, and (d) carry decoupled
family labels. These tests pin those guarantees at the public API surface.
"""

import random

import pytest

from zero_hack.data.synth import (
    CANONICAL_VOCAB,
    SynthSequence,
    generate_dataset,
    generate_one,
    sample_family_label,
)
from zero_hack.data.synth.generator import write_dataset_csv
from zero_hack.eval.phases import PHASES, UNKNOWN_PHASE, phase_runs, step_candidate_phases
from zero_hack.eval.validator import validate_sequence

_PHASE_INDEX = {phase: idx for idx, phase in enumerate(PHASES)}


def test_canonical_vocab_is_phase_known():
    """Every token the generator can emit must be labelable by phases.py."""
    assert CANONICAL_VOCAB
    for token in CANONICAL_VOCAB:
        candidates = step_candidate_phases(token)
        assert candidates != (UNKNOWN_PHASE,), f"{token!r} is UNKNOWN to phases.py"


def test_generated_sequence_is_valid():
    """Validity is baked in: no generated sequence triggers any rule."""
    for seed in range(50):
        steps = generate_one(random.Random(seed))
        violations = validate_sequence(steps)
        assert violations == [], f"seed {seed} produced {violations[:2]}"


def test_generation_is_deterministic():
    """Same seed -> identical sequence."""
    a = generate_one(random.Random(123))
    b = generate_one(random.Random(123))
    assert a == b
    assert generate_one(random.Random(124)) != a  # different seed -> different seq


def test_sequences_are_diverse():
    """Distinct seeds should overwhelmingly yield distinct sequences."""
    seqs = {tuple(generate_one(random.Random(seed))) for seed in range(200)}
    assert len(seqs) >= 190  # allow a few collisions, but near-unique


def test_phase_runs_are_monotonic_over_backbone():
    """The collapsed phase shape must walk PHASES forward with no backtracking."""
    for seed in range(30):
        steps = generate_one(random.Random(seed))
        runs = [p for p in phase_runs(steps) if p != UNKNOWN_PHASE]
        indices = [_PHASE_INDEX[p] for p in runs]
        assert indices == sorted(indices), f"seed {seed}: non-monotonic phases {runs}"


def test_sequence_length_in_eval_band():
    """Lengths stay near the reference families (107-151 steps, generous band)."""
    for seed in range(30):
        n = len(generate_one(random.Random(seed)))
        assert 80 <= n <= 220, f"seed {seed}: length {n} out of band"


@pytest.mark.slow
def test_dataset_covers_full_documented_vocab():
    """The headline requirement: every documented token appears in valid context."""
    dataset = generate_dataset(count=1500, seed=7)
    emitted = {step for seq in dataset for step in seq.steps}
    missing = CANONICAL_VOCAB - emitted
    assert not missing, f"{len(missing)} tokens never emitted: {sorted(missing)[:25]}"


def test_family_labels_are_decoupled_and_include_unk_and_synthetic():
    """Labels span known families, synthetic families, and UNK."""
    labels = [
        sample_family_label(random.Random(seed), synthetic_n=12, unk_prob=0.25)
        for seed in range(400)
    ]
    label_set = set(labels)
    assert {"mosfet", "igbt", "ic"} & label_set, "no known families sampled"
    assert any(lbl.startswith("synthetic_") for lbl in labels), "no synthetic families"
    assert "UNK" in label_set, "UNK never sampled"


def test_generate_dataset_returns_labeled_records():
    dataset = generate_dataset(count=20, seed=1)
    assert len(dataset) == 20
    assert all(isinstance(rec, SynthSequence) for rec in dataset)
    assert all(rec.family_label for rec in dataset)
    assert all(rec.steps for rec in dataset)
    # validity backstop holds for the dataset path too
    assert all(validate_sequence(list(rec.steps)) == [] for rec in dataset)


def test_write_dataset_csv_roundtrip(tmp_path):
    import csv

    dataset = generate_dataset(count=5, seed=3)
    out = tmp_path / "raw.csv"
    write_dataset_csv(out, dataset)

    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["FAMILY", "SEQUENCE_ID", "STEP"]
    body = rows[1:]
    assert len(body) == sum(len(rec.steps) for rec in dataset)
    # first data row matches the first sequence's first step
    assert body[0] == [dataset[0].family_label, dataset[0].sequence_id, dataset[0].steps[0]]
