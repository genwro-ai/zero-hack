"""Task 3 — anomaly-detection metrics.

The **positive class is an anomaly** (an invalid sequence, ``IS_VALID = 0``):
the task is to *detect* process-rule violations. Per ``generation_rules.md``
§5.2 we report Binary Accuracy, Precision, Recall, F1, the confusion matrix,
ROC-AUC, and Rule Attribution Accuracy.

``SCORE`` is the probability the sequence is *valid* (per the submission
format), so the anomaly score used for ROC-AUC is ``1 - SCORE``. AUC is computed
with the rank (Mann-Whitney) estimator and is ``None`` when scores are absent.

Rule Attribution Accuracy is measured *among detected violations* — examples
that are truly invalid and were flagged invalid (true positives) — as the
fraction whose ``PREDICTED_RULE`` matches the ground-truth rule.
"""

from __future__ import annotations


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    """AUC for ``score`` predicting ``label==1`` via the rank estimator.

    Returns ``None`` if either class is empty. Ties get averaged ranks.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # ranks are 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(auc, 4)


def score_anomaly(
    truth: dict[str, dict],
    predictions: dict[str, dict],
) -> dict:
    """Compute anomaly metrics over the shared example ids.

    ``truth``: example_id -> {is_valid:int, rule:str|None}.
    ``predictions``: example_id -> {is_valid:int, score:float|None, predicted_rule}.
    A missing prediction defaults to "valid" (is_valid=1, the negative class).
    """
    ids = sorted(truth)
    tp = fp = tn = fn = 0  # positive class = anomaly (invalid)
    auc_scores: list[float] = []
    auc_labels: list[int] = []
    attributable = 0
    attributed_correct = 0

    for example_id in ids:
        gold = truth[example_id]
        pred = predictions.get(example_id, {"is_valid": 1, "score": None, "predicted_rule": None})
        gold_anomaly = gold["is_valid"] == 0
        pred_anomaly = pred["is_valid"] == 0

        if gold_anomaly and pred_anomaly:
            tp += 1
        elif not gold_anomaly and pred_anomaly:
            fp += 1
        elif not gold_anomaly and not pred_anomaly:
            tn += 1
        else:
            fn += 1

        if pred.get("score") is not None:
            auc_scores.append(1.0 - float(pred["score"]))  # P(anomaly)
            auc_labels.append(1 if gold_anomaly else 0)

        # Rule attribution: among detected violations (true positives).
        if gold_anomaly and pred_anomaly:
            attributable += 1
            if pred.get("predicted_rule") and gold.get("rule"):
                attributed_correct += int(pred["predicted_rule"] == gold["rule"])

    n = len(ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n": n,
        "positive_class": "anomaly (IS_VALID=0)",
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "roc_auc": _roc_auc(auc_scores, auc_labels) if auc_scores else None,
        "rule_attribution_accuracy": (
            round(attributed_correct / attributable, 4) if attributable else None
        ),
        "n_detected_violations": attributable,
    }
