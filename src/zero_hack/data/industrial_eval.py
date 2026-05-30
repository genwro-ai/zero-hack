import csv
import json
import random
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.data.datasets import (
    FAMILY_REFERENCE_FILE_NAMES,
    SequenceRecord,
    load_industrial_family_records,
)
from zero_hack.eval import io
from zero_hack.eval.anomaly_synth import CorruptedExample, build_rule_stratified_corruptions

DEFAULT_INDUSTRIAL_DIR = PROJECT_ROOT / "data" / "industrial"
FAMILIES = tuple(FAMILY_REFERENCE_FILE_NAMES)


def load_industrial_variant_records(
    industrial_dir: str | Path = DEFAULT_INDUSTRIAL_DIR,
    *,
    families: tuple[str, ...] = FAMILIES,
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for family in families:
        records.extend(load_industrial_family_records(industrial_dir, family))
    return records


def records_by_family(records: list[SequenceRecord]) -> dict[str, list[SequenceRecord]]:
    by_family: dict[str, list[SequenceRecord]] = {}
    for record in records:
        by_family.setdefault(record.family, []).append(record)
    return by_family


def write_industrial_eval_set(
    out_dir: str | Path,
    *,
    valid_records: list[SequenceRecord],
    fractions: tuple[float, ...] = (0.6, 0.8),
    n_valid_per_family: int | None = None,
    n_anomaly_healthy_per_family: int | None = None,
    n_anomaly_unhealthy_per_family: int | None = None,
    seed: int = 1729,
    metadata: dict | None = None,
) -> None:
    out_dir = Path(out_dir)
    rng = random.Random(seed)

    valid_by_family = records_by_family(valid_records)
    for rows in valid_by_family.values():
        rng.shuffle(rows)

    valid_rows: list[list] = []
    nextstep_truth: list[list] = []
    completion_truth: list[list] = []
    anomaly_rows: list[list] = []
    anomaly_truth: list[list] = []

    for family, records in sorted(valid_by_family.items()):
        selected = records[:n_valid_per_family] if n_valid_per_family is not None else records
        for record in selected:
            steps = list(record.steps)
            for frac in fractions:
                cut = int(len(steps) * frac)
                cut = max(1, min(cut, len(steps) - 1))
                example_id = f"{family}_{record.sequence_id}_f{int(frac * 100)}"
                valid_rows.append([example_id, family, frac, io.join_steps(steps[:cut])])
                nextstep_truth.append([example_id, steps[cut]])
                completion_truth.append([example_id, io.join_steps(steps[cut:])])

    for family, records in sorted(valid_by_family.items()):
        selected = (
            records[:n_anomaly_healthy_per_family]
            if n_anomaly_healthy_per_family is not None
            else records
        )
        for record in selected:
            example_id = f"{family}_{record.sequence_id}_ok"
            anomaly_rows.append([example_id, family, io.join_steps(record.steps)])
            anomaly_truth.append([example_id, 1, ""])

    for family, records in sorted(valid_by_family.items()):
        target = (
            n_anomaly_unhealthy_per_family
            if n_anomaly_unhealthy_per_family is not None
            else len(records)
        )
        corruptions = build_rule_stratified_corruptions(
            records,
            n_invalid=target,
            rng=rng,
        )
        for example in corruptions:
            example_id = f"{family}_{example.sequence_id}_{example.rule}"
            anomaly_rows.append([example_id, family, io.join_steps(example.steps)])
            anomaly_truth.append([example_id, 0, example.rule])

    order = list(range(len(anomaly_rows)))
    rng.shuffle(order)
    anomaly_rows = [anomaly_rows[idx] for idx in order]
    anomaly_truth = [anomaly_truth[idx] for idx in order]

    _write_csv(
        out_dir / "eval_input_valid.csv",
        ["EXAMPLE_ID", "FAMILY", "COMPLETION_FRACTION", "PARTIAL_SEQUENCE"],
        valid_rows,
    )
    _write_csv(
        out_dir / "nextstep_truth.csv",
        ["EXAMPLE_ID", "NEXT_STEP"],
        nextstep_truth,
    )
    _write_csv(
        out_dir / "completion_truth.csv",
        ["EXAMPLE_ID", "TRUE_SEQUENCE"],
        completion_truth,
    )
    _write_csv(
        out_dir / "eval_input_anomaly.csv",
        ["EXAMPLE_ID", "FAMILY", "SEQUENCE"],
        anomaly_rows,
    )
    _write_csv(
        out_dir / "anomaly_truth.csv",
        ["EXAMPLE_ID", "IS_VALID", "RULE"],
        anomaly_truth,
    )

    payload = {
        "source": "industrial_variants_with_generated_rule_anomalies",
        "families": sorted(valid_by_family),
        "n_valid_rows": len(valid_rows),
        "n_anomaly_rows": len(anomaly_rows),
        "n_anomaly_healthy": sum(1 for row in anomaly_truth if row[1] == 1),
        "n_anomaly_unhealthy": sum(1 for row in anomaly_truth if row[1] == 0),
        "completion_fractions": list(fractions),
        "seed": seed,
        **(metadata or {}),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def write_threshold_calibration_set(
    out_dir: str | Path,
    *,
    valid_records: list[SequenceRecord],
    n_valid: int = 650,
    n_invalid: int = 350,
    seed: int = 1729,
    metadata: dict | None = None,
) -> None:
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    valid_by_family = records_by_family(valid_records)
    for rows in valid_by_family.values():
        rng.shuffle(rows)

    selected_valid = _balanced_take(valid_by_family, n_valid)
    selected_invalid = _balanced_corruptions(valid_by_family, n_invalid, rng)

    anomaly_rows: list[list] = []
    anomaly_truth: list[list] = []
    for record in selected_valid:
        example_id = f"calib_{record.family}_{record.sequence_id}_ok"
        anomaly_rows.append([example_id, record.family, io.join_steps(record.steps)])
        anomaly_truth.append([example_id, 1, ""])
    for record in selected_invalid:
        example_id = f"calib_{record.family}_{record.sequence_id}_{record.rule}"
        anomaly_rows.append([example_id, record.family, io.join_steps(record.steps)])
        anomaly_truth.append([example_id, 0, record.rule])

    order = list(range(len(anomaly_rows)))
    rng.shuffle(order)
    anomaly_rows = [anomaly_rows[idx] for idx in order]
    anomaly_truth = [anomaly_truth[idx] for idx in order]

    _write_csv(
        out_dir / "eval_input_anomaly.csv",
        ["EXAMPLE_ID", "FAMILY", "SEQUENCE"],
        anomaly_rows,
    )
    _write_csv(
        out_dir / "anomaly_truth.csv",
        ["EXAMPLE_ID", "IS_VALID", "RULE"],
        anomaly_truth,
    )

    payload = {
        "source": "industrial_variants_with_generated_rule_anomalies",
        "purpose": "threshold_calibration",
        "families": sorted(valid_by_family),
        "n_anomaly_rows": len(anomaly_rows),
        "n_anomaly_healthy": n_valid,
        "n_anomaly_unhealthy": n_invalid,
        "seed": seed,
        **(metadata or {}),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _balanced_corruptions(
    records_by_key: dict[str, list[SequenceRecord]],
    total: int,
    rng: random.Random,
) -> list[CorruptedExample]:
    examples: list[CorruptedExample] = []
    for key, count in _balanced_counts(records_by_key, total).items():
        examples.extend(
            build_rule_stratified_corruptions(
                records_by_key[key],
                n_invalid=count,
                rng=rng,
            )
        )
    return examples


def _balanced_counts(records_by_key: dict[str, list], total: int) -> dict[str, int]:
    if total < 0:
        raise ValueError("sample count must be non-negative")
    keys = sorted(records_by_key)
    if not keys:
        raise ValueError("cannot sample from an empty record set")
    base = total // len(keys)
    remainder = total % len(keys)
    return {key: base + (1 if index < remainder else 0) for index, key in enumerate(keys)}


def _balanced_take(records_by_key: dict[str, list], total: int) -> list:
    if total < 0:
        raise ValueError("sample count must be non-negative")
    if total == 0:
        return []
    keys = sorted(records_by_key)
    if not keys:
        raise ValueError("cannot sample from an empty record set")

    selected = []
    cursors = {key: 0 for key in keys}
    base = total // len(keys)
    remainder = total % len(keys)
    for index, key in enumerate(keys):
        target = base + (1 if index < remainder else 0)
        rows = records_by_key[key]
        take = min(target, len(rows))
        selected.extend(rows[:take])
        cursors[key] = take

    while len(selected) < total:
        progressed = False
        for key in keys:
            cursor = cursors[key]
            rows = records_by_key[key]
            if cursor >= len(rows):
                continue
            selected.append(rows[cursor])
            cursors[key] = cursor + 1
            progressed = True
            if len(selected) == total:
                break
        if not progressed:
            available = sum(len(rows) for rows in records_by_key.values())
            raise ValueError(f"requested {total} samples but only {available} are available")
    return selected


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
