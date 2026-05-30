import random
import warnings
from collections import Counter
from dataclasses import dataclass

from zero_hack.data import SequenceRecord
from zero_hack.eval.validator import _generator_module, first_violated_rule

RULE_IDS = (
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
)

_CLEAN_HINTS = ("CLEAN", "RCA", "HF DIP", "RINSE", "DRY WAFER")
_DEVELOP_STEPS = {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}


@dataclass(frozen=True)
class ValExample:
    family: str
    steps: list[str]
    label: int  # 1 = anomaly (invalid), 0 = valid


@dataclass(frozen=True)
class CorruptedExample:
    family: str
    sequence_id: str
    steps: list[str]
    rule: str


def corrupt_steps(
    steps: list[str],
    rng: random.Random,
    max_tries: int = 12,
    target_rule: str | None = None,
) -> tuple[list[str], str] | None:
    if target_rule is not None:
        if target_rule not in RULE_IDS:
            raise ValueError(f"Unknown target rule: {target_rule}")
        for _ in range(max_tries):
            candidate = _candidate_for_rule(steps, rng, target_rule)
            if candidate is not None and first_violated_rule(candidate) == target_rule:
                return candidate, target_rule
        return None

    for _ in range(max_tries):
        rule = rng.choice(RULE_IDS)
        corrupted = corrupt_steps(steps, rng, max_tries=1, target_rule=rule)
        if corrupted is not None:
            return corrupted
    return None


def build_rule_stratified_corruptions(
    records: list[SequenceRecord],
    *,
    n_invalid: int,
    rng: random.Random,
) -> list[CorruptedExample]:
    if n_invalid <= 0 or not records:
        return []

    records = list(records)
    rng.shuffle(records)
    targets = _balanced_rule_targets(n_invalid, rng)
    examples: list[CorruptedExample] = []
    unused_indices = list(range(len(records)))
    unrealized_targets: Counter[str] = Counter()

    for rule in targets:
        found = _corrupt_first_available(records, unused_indices, rng, rule)
        if found is None:
            wrap_indices = list(range(len(records)))
            rng.shuffle(wrap_indices)
            found = _corrupt_first_available(records, wrap_indices, rng, rule)
        if found is None:
            unrealized_targets[rule] += 1
            continue

        record_index, example = found
        examples.append(example)
        if record_index in unused_indices:
            unused_indices.remove(record_index)
    if unrealized_targets:
        missing = ", ".join(f"{rule}={count}" for rule, count in sorted(unrealized_targets.items()))
        warnings.warn(
            f"Only generated {len(examples)}/{n_invalid} invalid corruptions; "
            f"unrealized target rules: {missing}",
            RuntimeWarning,
            stacklevel=2,
        )
    return examples


def _corrupt_first_available(
    records: list[SequenceRecord],
    indices: list[int],
    rng: random.Random,
    rule: str,
) -> tuple[int, CorruptedExample] | None:
    for record_index in list(indices):
        record = records[record_index]
        corrupted = corrupt_steps(list(record.steps), rng, target_rule=rule)
        if corrupted is None:
            continue
        steps, observed_rule = corrupted
        return (
            record_index,
            CorruptedExample(
                family=record.family,
                sequence_id=record.sequence_id,
                steps=steps,
                rule=observed_rule,
            ),
        )
    return None


def _balanced_rule_targets(n_invalid: int, rng: random.Random) -> list[str]:
    targets = [RULE_IDS[i % len(RULE_IDS)] for i in range(n_invalid)]
    rng.shuffle(targets)
    return targets


def _candidate_for_rule(steps: list[str], rng: random.Random, rule: str) -> list[str] | None:
    if rule == "RULE_DEP_NO_CLEAN":
        return _drop_clean_before_deposition(steps, rng)
    if rule == "RULE_METAL_ETCH_NO_LITHO":
        return _drop_litho_before_metal_etch(steps, rng)
    if rule == "RULE_ETCH_NO_MASK":
        return _drop_develop_before_nonmetal_etch(steps, rng)
    if rule == "RULE_LITHO_LEVEL_SKIP":
        return _skip_litho_level(steps, rng)
    if rule == "RULE_IMPLANT_NO_MASK":
        return _move_step_early(steps, rng, _validator_set("IMPLANT_STEPS"))
    if rule == "RULE_CMP_NO_DEP":
        return _move_step_early(steps, rng, _validator_set("CMP_STEPS"))
    if rule == "RULE_PAD_OPEN_BEFORE_DEP":
        return _move_step_early(steps, rng, _validator_set("PAD_WINDOW_STEPS"))
    if rule == "RULE_TEST_BEFORE_PASSIVATION":
        return _move_step_early(steps, rng, _validator_set("ELECTRICAL_TEST_STEPS"))
    if rule == "RULE_SHIP_BEFORE_TEST":
        return _move_named_step_early(steps, "SHIP LOT")
    if rule == "RULE_BACKSIDE_BEFORE_PASSIVATION":
        return _place_backside_metal_before_passivation(steps, rng)
    return None


def _drop_clean_before_deposition(steps: list[str], rng: random.Random) -> list[str] | None:
    dep_steps = _validator_set("DEPOSITION_STEPS")
    clean_steps = _validator_set("CLEAN_STEPS")
    idxs = _shuffled_indices(steps, rng, dep_steps)
    for idx in idxs:
        clean_idxs = [i for i in range(max(0, idx - 12), idx) if steps[i] in clean_steps]
        if clean_idxs:
            return _delete_indices(steps, clean_idxs)
    return None


def _drop_litho_before_metal_etch(steps: list[str], rng: random.Random) -> list[str] | None:
    idxs = _shuffled_indices(steps, rng, _validator_set("METAL_ETCH_STEPS"))
    for idx in idxs:
        window = range(max(0, idx - 15), idx)
        drop_idxs = [
            i
            for i in window
            if steps[i].startswith("EXPOSE LITHO LEVEL") or steps[i] in _DEVELOP_STEPS
        ]
        if drop_idxs:
            return _delete_indices(steps, drop_idxs)
    return None


def _drop_develop_before_nonmetal_etch(steps: list[str], rng: random.Random) -> list[str] | None:
    etch_steps = set(_validator_set("ETCH_STEPS")) - set(_validator_set("METAL_ETCH_STEPS"))
    idxs = _shuffled_indices(steps, rng, etch_steps)
    for idx in idxs:
        develop_idxs = [i for i in range(max(0, idx - 12), idx) if steps[i] in _DEVELOP_STEPS]
        if develop_idxs:
            return _delete_indices(steps, develop_idxs)
    return None


def _skip_litho_level(steps: list[str], rng: random.Random) -> list[str] | None:
    align = [
        (idx, int(step.removeprefix("ALIGN MASK LEVEL ")))
        for idx, step in enumerate(steps)
        if step.startswith("ALIGN MASK LEVEL ") and step.removeprefix("ALIGN MASK LEVEL ").isdigit()
    ]
    if len(align) < 2:
        return None
    positions = list(range(1, len(align)))
    rng.shuffle(positions)
    pos = positions[0]
    prev_level = align[pos - 1][1]
    idx = align[pos][0]
    seq = list(steps)
    seq[idx] = f"ALIGN MASK LEVEL {prev_level + 2}"
    return seq


def _move_step_early(
    steps: list[str],
    rng: random.Random,
    targets: frozenset[str],
) -> list[str] | None:
    idxs = _shuffled_indices(steps, rng, targets)
    if not idxs:
        return None
    return _move_index(steps, idxs[0], 1)


def _move_named_step_early(steps: list[str], step_name: str) -> list[str] | None:
    if step_name not in steps:
        return None
    return _move_index(steps, steps.index(step_name), 1)


def _place_backside_metal_before_passivation(
    steps: list[str], rng: random.Random
) -> list[str] | None:
    clean_steps = _validator_set("CLEAN_STEPS")
    cure_idx = steps.index("CURE PASSIVATION") if "CURE PASSIVATION" in steps else len(steps)
    clean_idxs = [idx for idx, step in enumerate(steps[:cure_idx]) if step in clean_steps]
    if not clean_idxs:
        return None
    insert_at = rng.choice(clean_idxs) + 1
    seq = list(steps)
    if "DEPOSIT BACKSIDE METAL" in seq:
        old_idx = seq.index("DEPOSIT BACKSIDE METAL")
        step = seq.pop(old_idx)
        if old_idx < insert_at:
            insert_at -= 1
    else:
        step = "DEPOSIT BACKSIDE METAL"
    seq.insert(insert_at, step)
    return seq


def _move_index(steps: list[str], old_idx: int, insert_at: int) -> list[str]:
    seq = list(steps)
    step = seq.pop(old_idx)
    if old_idx < insert_at:
        insert_at -= 1
    seq.insert(insert_at, step)
    return seq


def _delete_indices(steps: list[str], idxs: list[int]) -> list[str]:
    seq = list(steps)
    for idx in sorted(idxs, reverse=True):
        del seq[idx]
    return seq


def _shuffled_indices(
    steps: list[str],
    rng: random.Random,
    targets: set[str] | frozenset[str],
) -> list[int]:
    idxs = [idx for idx, step in enumerate(steps) if step in targets]
    rng.shuffle(idxs)
    return idxs


def _validator_set(name: str) -> frozenset[str]:
    return getattr(_generator_module(), name)


def build_validation_anomaly_set(
    records: list[SequenceRecord],
    *,
    n_valid: int,
    n_invalid: int,
    seed: int,
) -> list[ValExample]:
    rng = random.Random(seed)

    by_family: dict[str, list[SequenceRecord]] = {}
    for record in records:
        by_family.setdefault(record.family, []).append(record)

    examples: list[ValExample] = []
    for family, recs in sorted(by_family.items()):
        recs = list(recs)
        rng.shuffle(recs)
        invalid_examples = build_rule_stratified_corruptions(
            recs,
            n_invalid=n_invalid,
            rng=rng,
        )
        examples.extend(
            ValExample(family=example.family, steps=example.steps, label=1)
            for example in invalid_examples
        )

        for rec in recs[:n_valid]:
            steps = list(rec.steps)
            examples.append(ValExample(family=family, steps=steps, label=0))
    return examples
