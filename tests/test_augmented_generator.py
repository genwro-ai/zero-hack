import random

import pytest

from zero_hack.data.augmented_generator import (
    AugmentationOptions,
    generate_augmented_dataset,
    generate_augmented_sequence,
)
from zero_hack.eval.validator import validate_sequence


def assert_valid(steps: list[str]) -> None:
    violations = validate_sequence(steps)
    assert violations == []


def core_litho_levels(steps: list[str]) -> list[int]:
    stop = min(
        idx
        for idx, step in enumerate(steps)
        if step in {"DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC"}
    )
    levels = []
    for step in steps[:stop]:
        if step.startswith("ALIGN MASK LEVEL "):
            levels.append(int(step.removeprefix("ALIGN MASK LEVEL ")))
    return levels


REFERENCE_LENGTH_LIMITS = {
    "mosfet": int(126 * 1.2),
    "igbt": int(151 * 1.2),
    "ic": int(107 * 1.2),
}


@pytest.mark.parametrize("family", ["mosfet", "igbt", "ic"])
def test_augmented_sequence_uses_organizer_validator(family):
    steps = generate_augmented_sequence(family, random.Random(7))
    assert_valid(steps)


@pytest.mark.parametrize("family", ["mosfet", "igbt", "ic"])
@pytest.mark.parametrize("seed", range(20))
def test_default_augmented_sequences_stay_near_reference_length(family, seed):
    steps = generate_augmented_sequence(family, random.Random(seed))
    assert len(steps) <= REFERENCE_LENGTH_LIMITS[family]


@pytest.mark.parametrize("family", ["mosfet", "igbt", "ic"])
@pytest.mark.parametrize("cycles", [3, 4, 5, 6])
def test_can_force_litho_cycle_count(family, cycles):
    steps = generate_augmented_sequence(
        family,
        random.Random(11),
        AugmentationOptions(litho_cycles=cycles),
    )
    assert_valid(steps)
    assert core_litho_levels(steps) == list(range(1, cycles + 1))


def test_can_force_post_expose_bake_present_or_absent():
    present = generate_augmented_sequence(
        "ic",
        random.Random(1),
        AugmentationOptions(post_expose_bake=True),
    )
    absent = generate_augmented_sequence(
        "ic",
        random.Random(1),
        AugmentationOptions(post_expose_bake=False),
    )
    assert_valid(present)
    assert_valid(absent)
    assert "POST EXPOSE BAKE" in present
    assert "POST EXPOSE BAKE" not in absent


def test_can_force_hard_bake_present_or_absent():
    present = generate_augmented_sequence(
        "mosfet",
        random.Random(2),
        AugmentationOptions(hard_bake=True),
    )
    absent = generate_augmented_sequence(
        "mosfet",
        random.Random(2),
        AugmentationOptions(hard_bake=False),
    )
    assert_valid(present)
    assert_valid(absent)
    assert "HARD BAKE" in present
    assert "HARD BAKE" not in absent


def test_can_insert_intermediate_clean_cycle():
    steps = generate_augmented_sequence(
        "igbt",
        random.Random(3),
        AugmentationOptions(intermediate_clean=True, synonym_style="canonical"),
    )
    assert_valid(steps)
    assert steps.count("HF DIP") >= 2


def test_can_omit_or_add_extra_measurements():
    with_measurements = generate_augmented_sequence(
        "ic",
        random.Random(4),
        AugmentationOptions(extra_measurements=True),
    )
    without_measurements = generate_augmented_sequence(
        "ic",
        random.Random(4),
        AugmentationOptions(extra_measurements=False),
    )
    assert_valid(with_measurements)
    assert_valid(without_measurements)
    assert "MEASURE LINE WIDTH" in with_measurements
    assert "MEASURE LINE WIDTH" not in without_measurements


def test_can_force_dry_wafer_after_hf_dip():
    present = generate_augmented_sequence(
        "mosfet",
        random.Random(5),
        AugmentationOptions(dry_wafer=True),
    )
    absent = generate_augmented_sequence(
        "mosfet",
        random.Random(5),
        AugmentationOptions(dry_wafer=False),
    )
    assert_valid(present)
    assert_valid(absent)
    assert "DRY WAFER" in present
    assert "DRY WAFER" not in absent


def test_can_force_igbt_epitaxial_rework_check():
    present = generate_augmented_sequence(
        "igbt",
        random.Random(6),
        AugmentationOptions(epitaxial_rework_check=True),
    )
    absent = generate_augmented_sequence(
        "igbt",
        random.Random(6),
        AugmentationOptions(epitaxial_rework_check=False),
    )
    assert_valid(present)
    assert_valid(absent)
    assert "EPITAXIAL REWORK CHECK" in present
    assert "EPITAXIAL REWORK CHECK" not in absent


def test_can_force_pre_anneal_check():
    present = generate_augmented_sequence(
        "mosfet",
        random.Random(7),
        AugmentationOptions(pre_anneal_check=True),
    )
    absent = generate_augmented_sequence(
        "mosfet",
        random.Random(7),
        AugmentationOptions(pre_anneal_check=False),
    )
    assert_valid(present)
    assert_valid(absent)
    assert "PRE ANNEAL CHECK" in present
    assert "PRE ANNEAL CHECK" not in absent


def test_can_force_second_metal_layer():
    present = generate_augmented_sequence(
        "ic",
        random.Random(8),
        AugmentationOptions(second_metal_layer=True),
    )
    absent = generate_augmented_sequence(
        "ic",
        random.Random(8),
        AugmentationOptions(second_metal_layer=False),
    )
    assert_valid(present)
    assert_valid(absent)
    assert present.count("METAL PATTERN INSPECTION") == absent.count("METAL PATTERN INSPECTION") + 1


@pytest.mark.parametrize("family", ["mosfet", "igbt", "ic"])
@pytest.mark.parametrize("seed", range(20))
def test_test_suite_keeps_fixed_sort_before_yield_order(family, seed):
    steps = generate_augmented_sequence(family, random.Random(seed))
    assert steps.index("WAFER SORT TEST") < steps.index("YIELD ANALYSIS")


def test_can_make_via_cmp_optional():
    present = generate_augmented_sequence(
        "mosfet",
        random.Random(9),
        AugmentationOptions(cmp_after_via_fill=True),
    )
    absent = generate_augmented_sequence(
        "mosfet",
        random.Random(9),
        AugmentationOptions(cmp_after_via_fill=False),
    )
    assert_valid(present)
    assert_valid(absent)
    assert any(step in {"CMP VIA FILL", "CMP METAL"} for step in present)
    assert not any(step in {"CMP VIA FILL", "CMP METAL"} for step in absent)


def test_can_force_step_synonym_style():
    steps = generate_augmented_sequence(
        "mosfet",
        random.Random(10),
        AugmentationOptions(synonym_style="alternate"),
    )
    assert_valid(steps)
    assert "WET CLEAN RCA1" in steps
    assert "WET CLEAN RCA2" in steps
    assert "STRIP RESIST" in steps


def test_augmented_dataset_is_unique_and_valid():
    dataset = generate_augmented_dataset(
        "ic",
        count=5,
        seed=13,
        options=AugmentationOptions(litho_cycles=5),
    )
    assert len(dataset) == 5
    assert len({tuple(seq) for seq in dataset}) == 5
    for seq in dataset:
        assert_valid(seq)
