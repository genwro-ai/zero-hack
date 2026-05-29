"""Readers/writers for the shared eval-protocol CSV formats.

Covers the organizer eval *inputs* (``eval_input_valid.csv``,
``eval_input_anomaly.csv``), the three *submission* files (Tasks 1-3), and the
*ground-truth* files used for local self-scoring. Column lookups are
case-insensitive and accept the common name variants, since the organizer fixes
submission columns but not ground-truth ones.

Multi-step fields (``PARTIAL_SEQUENCE``, ``SEQUENCE``, ``PREDICTED_SEQUENCE``,
ground-truth sequences) are pipe-separated, per ``generation_rules.md`` §5.
"""

from __future__ import annotations

import csv
from pathlib import Path

PIPE = "|"


def split_steps(value: str) -> list[str]:
    """Split a pipe-separated step field into a list, dropping blanks."""
    if value is None:
        return []
    return [step.strip() for step in value.split(PIPE) if step.strip()]


def join_steps(steps: list[str] | tuple[str, ...]) -> str:
    return PIPE.join(steps)


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _resolve(row: dict[str, str], *candidates: str) -> str | None:
    """Case-insensitively fetch the first present candidate column."""
    lower = {k.lower(): v for k, v in row.items()}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# --------------------------------------------------------------------------- #
# Eval inputs                                                                  #
# --------------------------------------------------------------------------- #
def read_eval_input_valid(path: str | Path) -> list[dict]:
    """Read Tasks 1/2 input: EXAMPLE_ID, FAMILY, COMPLETION_FRACTION, PARTIAL_SEQUENCE."""
    rows = []
    for row in _read_rows(path):
        frac = _resolve(row, "COMPLETION_FRACTION", "FRACTION")
        rows.append(
            {
                "example_id": _resolve(row, "EXAMPLE_ID", "ID"),
                "family": (_resolve(row, "FAMILY") or "").lower(),
                "completion_fraction": float(frac) if frac else None,
                "partial_sequence": split_steps(_resolve(row, "PARTIAL_SEQUENCE", "PARTIAL") or ""),
            }
        )
    return rows


def read_eval_input_anomaly(path: str | Path) -> list[dict]:
    """Read Task 3 input: EXAMPLE_ID, FAMILY, SEQUENCE."""
    rows = []
    for row in _read_rows(path):
        rows.append(
            {
                "example_id": _resolve(row, "EXAMPLE_ID", "ID"),
                "family": (_resolve(row, "FAMILY") or "").lower(),
                "sequence": split_steps(_resolve(row, "SEQUENCE", "FULL_SEQUENCE") or ""),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Predictions (submission files)                                              #
# --------------------------------------------------------------------------- #
def read_next_step_predictions(path: str | Path) -> dict[str, list[str]]:
    """EXAMPLE_ID -> [RANK_1 ... RANK_5] (blanks dropped)."""
    out: dict[str, list[str]] = {}
    for row in _read_rows(path):
        example_id = _resolve(row, "EXAMPLE_ID", "ID")
        ranks = []
        for i in range(1, 6):
            step = _resolve(row, f"RANK_{i}")
            if step and step.strip():
                ranks.append(step.strip())
        out[example_id] = ranks
    return out


def read_completion_predictions(path: str | Path) -> dict[str, list[str]]:
    """EXAMPLE_ID -> predicted suffix steps (after the cut point)."""
    out: dict[str, list[str]] = {}
    for row in _read_rows(path):
        example_id = _resolve(row, "EXAMPLE_ID", "ID")
        seq = _resolve(row, "PREDICTED_SEQUENCE", "PREDICTION", "SEQUENCE") or ""
        out[example_id] = split_steps(seq)
    return out


def read_anomaly_predictions(path: str | Path) -> dict[str, dict]:
    """EXAMPLE_ID -> {is_valid:int, score:float|None, predicted_rule:str|None}."""
    out: dict[str, dict] = {}
    for row in _read_rows(path):
        example_id = _resolve(row, "EXAMPLE_ID", "ID")
        score = _resolve(row, "SCORE")
        rule = _resolve(row, "PREDICTED_RULE", "RULE", "RULE_ID")
        out[example_id] = {
            "is_valid": int(float(_resolve(row, "IS_VALID", "VALID"))),
            "score": float(score) if score not in (None, "") else None,
            "predicted_rule": rule.strip() if rule and rule.strip() else None,
        }
    return out


# --------------------------------------------------------------------------- #
# Ground truth (local self-scoring)                                           #
# --------------------------------------------------------------------------- #
def read_next_step_truth(path: str | Path) -> dict[str, str]:
    """EXAMPLE_ID -> true next step."""
    out: dict[str, str] = {}
    for row in _read_rows(path):
        example_id = _resolve(row, "EXAMPLE_ID", "ID")
        step = _resolve(row, "NEXT_STEP", "TRUE_NEXT_STEP", "TARGET", "STEP")
        out[example_id] = (step or "").strip()
    return out


def read_completion_truth(path: str | Path) -> dict[str, list[str]]:
    """EXAMPLE_ID -> true suffix steps (after the cut point)."""
    out: dict[str, list[str]] = {}
    for row in _read_rows(path):
        example_id = _resolve(row, "EXAMPLE_ID", "ID")
        seq = _resolve(row, "TRUE_SEQUENCE", "GROUND_TRUTH", "REMAINING_SEQUENCE", "SEQUENCE") or ""
        out[example_id] = split_steps(seq)
    return out


def read_anomaly_truth(path: str | Path) -> dict[str, dict]:
    """EXAMPLE_ID -> {is_valid:int, rule:str|None}."""
    out: dict[str, dict] = {}
    for row in _read_rows(path):
        example_id = _resolve(row, "EXAMPLE_ID", "ID")
        rule = _resolve(row, "RULE", "TRUE_RULE", "PREDICTED_RULE", "RULE_ID")
        out[example_id] = {
            "is_valid": int(float(_resolve(row, "IS_VALID", "VALID"))),
            "rule": rule.strip() if rule and rule.strip() else None,
        }
    return out


# --------------------------------------------------------------------------- #
# Submission writers                                                          #
# --------------------------------------------------------------------------- #
def write_next_step_predictions(path: str | Path, rows: list[dict]) -> None:
    """Rows: {example_id, ranks: list[str]} -> EXAMPLE_ID, RANK_1..RANK_5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["EXAMPLE_ID", "RANK_1", "RANK_2", "RANK_3", "RANK_4", "RANK_5"])
        for row in rows:
            ranks = list(row["ranks"])[:5]
            ranks += [""] * (5 - len(ranks))
            writer.writerow([row["example_id"], *ranks])


def write_completion_predictions(path: str | Path, rows: list[dict]) -> None:
    """Rows: {example_id, steps: list[str]} -> EXAMPLE_ID, PREDICTED_SEQUENCE."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["EXAMPLE_ID", "PREDICTED_SEQUENCE"])
        for row in rows:
            writer.writerow([row["example_id"], join_steps(row["steps"])])


def write_anomaly_predictions(path: str | Path, rows: list[dict]) -> None:
    """Rows: {example_id, is_valid, score?, predicted_rule?}."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["EXAMPLE_ID", "IS_VALID", "SCORE", "PREDICTED_RULE"])
        for row in rows:
            score = row.get("score")
            writer.writerow(
                [
                    row["example_id"],
                    int(row["is_valid"]),
                    "" if score is None else f"{float(score):.6f}",
                    row.get("predicted_rule") or "",
                ]
            )
