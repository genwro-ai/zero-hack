from zero_hack import INDUSTRIAL_DATA_DIR
from zero_hack.data import load_sequence_records
from zero_hack.eval.phases import (
    PHASES,
    UNKNOWN_PHASE,
    build_step_phase_rows,
    phase_runs,
    step_candidate_phases,
    steps_to_phases,
)


def _industrial_records():
    records = []
    for family, name in (
        ("mosfet", "MOSFET_variants.csv"),
        ("igbt", "IGBT_variants.csv"),
        ("ic", "IC_variants.csv"),
    ):
        records.extend(load_sequence_records(INDUSTRIAL_DATA_DIR / name, family=family))
    return records


def test_all_industrial_vocab_steps_have_phase_candidates():
    steps = {step for record in _industrial_records()[:30] for step in record.steps}

    rows = build_step_phase_rows(steps)

    assert rows
    assert all(row.primary_phase != UNKNOWN_PHASE for row in rows)
    assert all(UNKNOWN_PHASE not in row.candidate_phases for row in rows)


def test_sequence_phase_labels_are_position_aligned_and_monotonic():
    phase_index = {phase: idx for idx, phase in enumerate(PHASES)}

    for record in _industrial_records()[:30]:
        labels = steps_to_phases(record.steps)
        assert len(labels) == len(record.steps)
        assert all(label in phase_index for label in labels)
        label_indices = [phase_index[label] for label in labels]
        assert label_indices == sorted(label_indices)


def test_phase_runs_capture_expected_backbone_shape():
    record = load_sequence_records(INDUSTRIAL_DATA_DIR / "MOSFET_variants.csv", family="mosfet")[0]

    runs = phase_runs(record.steps)

    assert runs[0] == "PREFIX"
    assert "PROCESS_CYCLE" in runs
    assert "PASSIVATION_BLOCK" in runs
    assert "TEST_SUITE" in runs
    assert runs[-1] == "SUFFIX"


def test_ambiguous_steps_expose_multiple_candidate_phases():
    assert len(step_candidate_phases("THERMAL OXIDATION")) > 1
    assert "PROCESS_CYCLE" in step_candidate_phases("SPIN COAT PHOTORESIST")
