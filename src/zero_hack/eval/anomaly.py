def _roc_auc(labels: list[int], scores: list[float]) -> float | None:
    """Official pairwise ROC-AUC.

    ``labels`` use 1 = valid, 0 = invalid; ``scores`` are higher for valid.
    """

    pos_scores = [score for score, label in zip(scores, labels, strict=False) if label == 1]
    neg_scores = [score for score, label in zip(scores, labels, strict=False) if label == 0]
    if not pos_scores or not neg_scores:
        return None
    concordant = sum(pos > neg for pos in pos_scores for neg in neg_scores)
    tied = sum(pos == neg for pos in pos_scores for neg in neg_scores)
    auc = (concordant + 0.5 * tied) / (len(pos_scores) * len(neg_scores))
    return round(auc, 4)


def score_anomaly(
    truth: dict[str, dict],
    predictions: dict[str, dict],
    families: dict[str, str] | None = None,
) -> dict:
    """Compute anomaly metrics over the shared example ids.

    ``truth``: example_id -> {is_valid:int, rule:str|None}.
    ``predictions``: example_id -> {is_valid:int, score:float|None, predicted_rule}.
    A missing prediction defaults to "valid" (is_valid=1, the negative class).
    """
    ids = sorted(truth)
    groups: dict[str, list[str]] = {"all": ids}
    if families:
        for example_id in ids:
            groups.setdefault(families.get(example_id, "unknown"), []).append(example_id)

    return {
        group: _score_anomaly_ids(group_ids, truth, predictions)
        for group, group_ids in groups.items()
    }


def _score_anomaly_ids(
    ids: list[str],
    truth: dict[str, dict],
    predictions: dict[str, dict],
) -> dict:
    tp = fp = tn = fn = 0  # positive class = anomaly (invalid)
    auc_labels: list[int] = []
    auc_scores: list[float] = []
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
            auc_labels.append(int(gold["is_valid"]))
            auc_scores.append(float(pred["score"]))

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
        "roc_auc": _roc_auc(auc_labels, auc_scores) if auc_scores else None,
        "rule_attribution_accuracy": (
            round(attributed_correct / attributable, 4) if attributable else None
        ),
        "n_detected_violations": attributable,
    }
