import random
from dataclasses import dataclass

from zero_hack.data import SequenceRecord
from zero_hack.eval.validator import first_violated_rule, is_valid

_CLEAN_HINTS = ("CLEAN", "RCA", "HF DIP", "RINSE", "DRY WAFER")


@dataclass(frozen=True)
class ValExample:
    family: str
    steps: list[str]
    label: int  # 1 = anomaly (invalid), 0 = valid


def corrupt_steps(
    steps: list[str], rng: random.Random, max_tries: int = 12
) -> tuple[list[str], str] | None:
    n = len(steps)
    for _ in range(max_tries):
        op = rng.choice(("drop_clean", "drop_develop", "ship_early", "test_early", "swap"))
        seq = list(steps)

        if op == "drop_clean":
            idxs = [i for i, s in enumerate(seq) if any(h in s for h in _CLEAN_HINTS)]
            if idxs:
                del seq[rng.choice(idxs)]
        elif op == "drop_develop":
            idxs = [i for i, s in enumerate(seq) if s.startswith("DEVELOP")]
            if idxs:
                del seq[rng.choice(idxs)]
        elif op == "ship_early" and "SHIP LOT" in seq:
            seq.remove("SHIP LOT")
            seq.insert(rng.randint(0, max(0, len(seq) // 3)), "SHIP LOT")
        elif op == "test_early":
            idxs = [i for i, s in enumerate(seq) if s.endswith("TEST") and "WAFER SORT" not in s]
            if idxs:
                step = seq.pop(rng.choice(idxs))
                seq.insert(rng.randint(0, max(0, len(seq) // 4)), step)
        elif op == "swap" and n > 6:
            i = rng.randint(0, n - 2)
            j = rng.randint(0, n - 2)
            seq[i], seq[j] = seq[j], seq[i]

        if seq != steps and not is_valid(seq):
            return seq, first_violated_rule(seq) or "UNKNOWN"
    return None


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
        kept_valid = kept_invalid = 0
        for rec in recs:
            if kept_valid >= n_valid and kept_invalid >= n_invalid:
                break
            steps = list(rec.steps)
            if kept_invalid < n_invalid:
                corrupted = corrupt_steps(steps, rng)
                if corrupted is None:
                    continue
                seq, _ = corrupted
                examples.append(ValExample(family=family, steps=seq, label=1))
                kept_invalid += 1
            elif kept_valid < n_valid:
                examples.append(ValExample(family=family, steps=steps, label=0))
                kept_valid += 1
    return examples
