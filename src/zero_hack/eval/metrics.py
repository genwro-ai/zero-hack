def next_step(pairs):
    top1 = top3 = top5 = mrr = 0.0
    for ranked, truth in pairs:
        if truth in ranked:
            rank = ranked.index(truth) + 1
            mrr += 1 / rank
            top1 += rank == 1
            top3 += rank <= 3
            top5 += rank <= 5
    n = len(pairs) or 1
    return {"top1": top1 / n, "top3": top3 / n, "top5": top5 / n, "mrr": mrr / n}


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def completion(pairs):
    exact = edit = token = 0.0
    for pred, truth in pairs:
        exact += pred == truth
        edit += edit_distance(pred, truth) / max(len(pred), len(truth), 1)
        hits = sum(p == t for p, t in zip(pred, truth, strict=False))
        token += hits / max(len(truth), 1)
    n = len(pairs) or 1
    return {"exact_match": exact / n, "edit_distance": edit / n, "token_acc": token / n}


def roc_auc(labels, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return 0.5
    rank_sum = sum(r for r, y in zip(ranks, labels, strict=False) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def anomaly(rows):
    tp = fp = tn = fn = hits = 0
    labels, scores = [], []
    for valid_pred, score, rule_pred, valid_true, rule_true in rows:
        bad_pred = valid_pred == 0
        bad_true = valid_true == 0
        tp += bad_pred and bad_true
        fp += bad_pred and not bad_true
        tn += not bad_pred and not bad_true
        fn += not bad_pred and bad_true
        hits += bad_pred and bad_true and rule_pred == rule_true
        labels.append(1 - valid_true)
        scores.append(1 - score)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / (len(rows) or 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(labels, scores),
        "rule_acc": hits / tp if tp else 0.0,
    }
