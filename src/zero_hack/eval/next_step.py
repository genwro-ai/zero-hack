def _reciprocal_rank(gold: str, ranked: list[str]) -> float:
    for idx, pred in enumerate(ranked, start=1):
        if pred == gold:
            return 1.0 / idx
    return 0.0


def score_next_step(
    truth: dict[str, str],
    predictions: dict[str, list[str]],
    families: dict[str, str] | None = None,
) -> dict:
    """Compute next-step metrics over the shared example ids.

    ``truth``: example_id -> gold next step.
    ``predictions``: example_id -> ranked predictions (best first).
    ``families``: optional example_id -> family for per-group breakdown.
    Missing predictions for an evaluated id count as a miss (empty ranking).
    """
    ids = sorted(truth)
    groups: dict[str, list[str]] = {"all": ids}
    if families:
        for example_id in ids:
            groups.setdefault(families.get(example_id, "unknown"), []).append(example_id)

    out: dict[str, dict] = {}
    for group, group_ids in groups.items():
        n = len(group_ids)
        top1 = top3 = top5 = 0
        mrr = 0.0
        for example_id in group_ids:
            gold = truth[example_id]
            ranked = predictions.get(example_id, [])
            if ranked[:1] == [gold]:
                top1 += 1
            if gold in ranked[:3]:
                top3 += 1
            if gold in ranked[:5]:
                top5 += 1
            mrr += _reciprocal_rank(gold, ranked)
        out[group] = {
            "n": n,
            "top1": round(top1 / n, 4) if n else 0.0,
            "top3": round(top3 / n, 4) if n else 0.0,
            "top5": round(top5 / n, 4) if n else 0.0,
            "mrr": round(mrr / n, 4) if n else 0.0,
        }
    return out
