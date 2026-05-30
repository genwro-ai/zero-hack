import csv
from collections import defaultdict
from pathlib import Path

from zero_hack.eval import io

TASKS = ("next_step", "completion", "anomaly")


def _levenshtein(seq1: list, seq2: list) -> int:
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev_row = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev_row[j - 1]
            else:
                dp[j] = 1 + min(prev_row[j], dp[j - 1], prev_row[j - 1])
    return dp[n]


def normalized_edit_distance(pred: list[str], ref: list[str]) -> float:
    if not pred and not ref:
        return 0.0
    return _levenshtein(pred, ref) / max(len(pred), len(ref))


def token_accuracy(pred: list[str], ref: list[str]) -> float:
    """Fraction of positions up to min length where pred matches ref."""
    n = min(len(pred), len(ref))
    if n == 0:
        return 0.0
    return sum(p == r for p, r in zip(pred, ref, strict=False)) / n


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    """Pairwise ROC-AUC from the organizer scorer.

    labels: 1 = positive (valid), 0 = negative (invalid).
    scores: higher score = more likely positive.
    """
    pos_scores = [s for s, label in zip(scores, labels, strict=False) if label == 1]
    neg_scores = [s for s, label in zip(scores, labels, strict=False) if label == 0]
    if not pos_scores or not neg_scores:
        return float("nan")
    concordant = sum(p > n for p in pos_scores for n in neg_scores)
    tied = sum(p == n for p in pos_scores for n in neg_scores)
    total = len(pos_scores) * len(neg_scores)
    return (concordant + 0.5 * tied) / total


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _major_block(step: str) -> str:
    """Map a process step to a coarse major process block."""
    s = step.upper()
    if "LITHO" in s or s.startswith("SPIN COAT PHOTORESIST") or "MASK LEVEL" in s:
        return "LITHO"
    if "ETCH" in s or s.startswith("OPEN PAD WINDOW"):
        return "ETCH"
    if "IMPLANT" in s or "ANNEAL" in s or "DIFFUSION" in s:
        return "DOPING_THERMAL"
    if s.startswith("DEPOSIT") or "OXIDATION" in s or "GROWTH" in s:
        return "DEPOSITION"
    if s.startswith("CMP") or "PLANAR" in s:
        return "PLANARIZATION"
    if "VIA" in s:
        return "VIA"
    if "PASSIVATION" in s:
        return "PASSIVATION"
    if "BACKSIDE" in s or "GRIND" in s:
        return "BACKSIDE"
    if "TEST" in s or "MEASURE" in s or "INSPECT" in s or "ANALYSIS" in s:
        return "METROLOGY_TEST"
    if "LOT" in s or "RELEASE" in s or "SHIP" in s:
        return "LOGISTICS"
    return "OTHER"


def _block_signature(seq: list[str]) -> list[str]:
    sig: list[str] = []
    prev: str | None = None
    for step in seq:
        block = _major_block(step)
        if block != prev:
            sig.append(block)
            prev = block
    return sig


def block_level_accuracy(pred: list[str], ref: list[str]) -> float:
    return token_accuracy(_block_signature(pred), _block_signature(ref))


def _norm_key(header: str) -> str:
    return header.strip().lstrip("\ufeff").strip('"')


def _read_csv_norm(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [
            {_norm_key(key): (value or "").strip().strip('"') for key, value in row.items()}
            for row in reader
        ]


def _mean(values: list[float | bool]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _round4(value: float) -> float | None:
    if value != value:
        return None
    return round(value, 4)


def _group_ids(
    ids: list[str],
    families: dict[str, str] | None = None,
    fractions: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"all": ids}
    if families:
        for example_id in ids:
            groups.setdefault(families.get(example_id, "unknown"), []).append(example_id)
    if fractions:
        for example_id in ids:
            groups.setdefault(f"fraction:{fractions.get(example_id, 'unknown')}", []).append(
                example_id
            )
    return groups


def score_next_step(
    truth: dict[str, str],
    predictions: dict[str, list[str]],
    families: dict[str, str] | None = None,
    fractions: dict[str, str] | None = None,
) -> dict:
    """Score next-step predictions with organizer Top-1/3/5 and MRR definitions."""
    ids = sorted(example_id for example_id in truth if example_id in predictions)
    if not ids:
        raise ValueError("No matching EXAMPLE_IDs between ground truth and predictions.")

    out: dict[str, dict] = {}
    for group, group_ids in _group_ids(ids, families, fractions).items():
        top1: list[bool] = []
        top3: list[bool] = []
        top5: list[bool] = []
        mrr: list[float] = []
        for example_id in group_ids:
            truth_step = truth[example_id]
            ranks = predictions[example_id]
            top1.append(bool(ranks) and ranks[0] == truth_step)
            top3.append(truth_step in ranks[:3])
            top5.append(truth_step in ranks)
            mrr.append(1.0 / (ranks.index(truth_step) + 1) if truth_step in ranks else 0.0)
        out[group] = {
            "n": len(group_ids),
            "top1": _round4(_mean(top1)),
            "top3": _round4(_mean(top3)),
            "top5": _round4(_mean(top5)),
            "mrr": _round4(_mean(mrr)),
        }
    return out


def score_completion(
    truth: dict[str, list[str]],
    predictions: dict[str, list[str]],
    families: dict[str, str] | None = None,
    fractions: dict[str, str] | None = None,
) -> dict:
    """Score sequence completion with organizer NED/exact/token/block metrics."""
    ids = sorted(example_id for example_id in truth if example_id in predictions)
    if not ids:
        raise ValueError("No matching EXAMPLE_IDs between ground truth and predictions.")

    out: dict[str, dict] = {}
    for group, group_ids in _group_ids(ids, families, fractions).items():
        ned: list[float] = []
        exact: list[bool] = []
        tacc: list[float] = []
        block: list[float] = []
        for example_id in group_ids:
            ref = truth[example_id]
            pred = predictions[example_id]
            ned.append(normalized_edit_distance(pred, ref))
            exact.append(pred == ref)
            tacc.append(token_accuracy(pred, ref))
            block.append(block_level_accuracy(pred, ref))
        out[group] = {
            "n": len(group_ids),
            "exact_match": _round4(_mean(exact)),
            "norm_edit_distance": _round4(_mean(ned)),
            "token_accuracy": _round4(_mean(tacc)),
            "block_accuracy": _round4(_mean(block)),
        }
    return out


def score_anomaly(
    truth: dict[str, dict],
    predictions: dict[str, dict],
    families: dict[str, str] | None = None,
) -> dict:
    ids = sorted(example_id for example_id in truth if example_id in predictions)
    if not ids:
        raise ValueError("No matching EXAMPLE_IDs between ground truth and predictions.")

    return {
        group: _score_anomaly_ids(group_ids, truth, predictions)
        for group, group_ids in _group_ids(ids, families).items()
    }


def _score_anomaly_ids(
    ids: list[str],
    truth: dict[str, dict],
    predictions: dict[str, dict],
) -> dict:
    labels: list[int] = []
    scores: list[float] = []
    preds_bin: list[int] = []
    rule_pairs: list[tuple[str, str]] = []

    for example_id in ids:
        gold = truth[example_id]
        pred = predictions[example_id]
        labels.append(int(gold["is_valid"]))
        raw_pred = pred.get("is_valid", -1)
        pred_valid = int(raw_pred) if raw_pred is not None else -1
        preds_bin.append(pred_valid)
        score = pred.get("score")
        scores.append(
            float(score) if score is not None else float(pred_valid) if pred_valid >= 0 else 0.5
        )
        rule = gold.get("rule") or gold.get("violation_rule") or ""
        if rule:
            rule_pairs.append((example_id, rule))

    n_pos = sum(label == 1 for label in labels)
    n_neg = sum(label == 0 for label in labels)
    accuracy = _safe_div(
        sum(pred == label for pred, label in zip(preds_bin, labels, strict=False)),
        len(labels),
    )
    auc = _roc_auc(labels, scores)

    tp = sum((label == 0) and (pred == 0) for label, pred in zip(labels, preds_bin, strict=False))
    tn = sum((label == 1) and (pred == 1) for label, pred in zip(labels, preds_bin, strict=False))
    fp = sum((label == 1) and (pred == 0) for label, pred in zip(labels, preds_bin, strict=False))
    fn = sum((label == 0) and (pred != 0) for label, pred in zip(labels, preds_bin, strict=False))
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)

    correct_detection = [
        (rule, predictions[example_id].get("predicted_rule", "") or "")
        for example_id, rule in rule_pairs
        if truth[example_id]["is_valid"] == 0 and predictions[example_id]["is_valid"] == 0
    ]
    rule_attr = (
        sum(rule == pred_rule for rule, pred_rule in correct_detection) / len(correct_detection)
        if correct_detection
        else float("nan")
    )

    rule_detection = defaultdict(list)
    for example_id in ids:
        if truth[example_id]["is_valid"] == 0:
            rule = truth[example_id].get("rule") or truth[example_id].get("violation_rule") or ""
            rule_detection[rule].append(predictions[example_id]["is_valid"] == 0)

    return {
        "n": len(ids),
        "n_valid": n_pos,
        "n_invalid": n_neg,
        "positive_class": "anomaly (IS_VALID=0)",
        "accuracy": _round4(accuracy),
        "precision": _round4(precision),
        "recall": _round4(recall),
        "f1": _round4(f1),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "roc_auc": _round4(auc),
        "rule_attribution_accuracy": _round4(rule_attr),
        "n_detected_violations": len(correct_detection),
        "detection_rate_by_rule": {
            rule: _round4(_mean(hits)) for rule, hits in sorted(rule_detection.items()) if rule
        },
    }


def _families_and_fractions(eval_input: str | Path | None) -> tuple[dict[str, str], dict[str, str]]:
    if eval_input is None:
        return {}, {}
    rows = io.read_eval_input_valid(eval_input)
    return (
        {row["example_id"]: row["family"] for row in rows},
        {
            row["example_id"]: str(row["completion_fraction"])
            for row in rows
            if row.get("completion_fraction") is not None
        },
    )


def _anomaly_families(eval_input: str | Path | None) -> dict[str, str]:
    if eval_input is None:
        return {}
    return {row["example_id"]: row["family"] for row in io.read_eval_input_anomaly(eval_input)}


def score_next_step_files(
    ground_truth: str | Path,
    predictions: str | Path,
    eval_input: str | Path | None = None,
) -> dict:
    truth = io.read_next_step_truth(ground_truth)
    pred = io.read_next_step_predictions(predictions)
    families, fractions = _families_and_fractions(eval_input)

    for row in _read_csv_norm(ground_truth):
        example_id = row.get("EXAMPLE_ID")
        if not example_id:
            continue
        if row.get("FAMILY"):
            families[example_id] = row["FAMILY"].lower()
        if row.get("COMPLETION_FRACTION"):
            fractions[example_id] = row["COMPLETION_FRACTION"]

    return score_next_step(truth, pred, families=families or None, fractions=fractions or None)


def score_completion_files(
    ground_truth: str | Path,
    predictions: str | Path,
    eval_input: str | Path | None = None,
) -> dict:
    pred = io.read_completion_predictions(predictions)
    families, fractions = _families_and_fractions(eval_input)
    truth: dict[str, list[str]] = {}

    rows = _read_csv_norm(ground_truth)
    if rows and "FULL_SEQUENCE" in rows[0] and "PARTIAL_SEQUENCE" in rows[0]:
        for row in rows:
            partial = io.split_steps(row["PARTIAL_SEQUENCE"])
            full = io.split_steps(row["FULL_SEQUENCE"])
            example_id = row["EXAMPLE_ID"]
            truth[example_id] = full[len(partial) :]
            if row.get("FAMILY"):
                families[example_id] = row["FAMILY"].lower()
            if row.get("COMPLETION_FRACTION"):
                fractions[example_id] = row["COMPLETION_FRACTION"]
    else:
        truth = io.read_completion_truth(ground_truth)

    return score_completion(truth, pred, families=families or None, fractions=fractions or None)


def score_anomaly_files(
    ground_truth: str | Path,
    predictions: str | Path,
    eval_input: str | Path | None = None,
    valid_supplement: str | Path | None = None,
) -> dict:
    truth: dict[str, dict] = {}
    rows = _read_csv_norm(ground_truth)
    for row in rows:
        example_id = row["EXAMPLE_ID"]
        if "IS_VALID" in row and row["IS_VALID"] != "":
            truth[example_id] = {
                "is_valid": int(float(row["IS_VALID"])),
                "rule": row.get("RULE") or row.get("VIOLATION_RULE") or "",
            }
        else:
            truth[example_id] = {
                "is_valid": 0,
                "rule": row.get("VIOLATION_RULE") or row.get("RULE") or "",
            }

    if valid_supplement:
        for row in _read_csv_norm(valid_supplement):
            truth.setdefault(row["EXAMPLE_ID"], {"is_valid": 1, "rule": ""})

    return score_anomaly(
        truth,
        io.read_anomaly_predictions(predictions),
        families=_anomaly_families(eval_input) or None,
    )


def score_task(
    task: str,
    ground_truth: str | Path,
    predictions: str | Path,
    eval_input: str | Path | None = None,
    valid_supplement: str | Path | None = None,
) -> dict:
    dispatch = {
        "next_step": score_next_step_files,
        "next-step": score_next_step_files,
        "completion": score_completion_files,
        "anomaly": score_anomaly_files,
    }
    if task not in dispatch:
        raise ValueError(f"Unknown task {task!r}. Expected one of: {', '.join(TASKS)}")
    if task == "anomaly":
        return dispatch[task](ground_truth, predictions, eval_input, valid_supplement)
    return dispatch[task](ground_truth, predictions, eval_input)
