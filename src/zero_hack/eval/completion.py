"""Task 2 — sequence-completion metrics.

Each example provides a true suffix (the steps *after* the cut point) and a
predicted suffix. Per ``generation_rules.md`` §5.2 we report:

- **Exact Match Rate** — fraction with prediction identical to truth.
- **Normalized Edit Distance** — token-level Levenshtein / ``max(len)`` (lower
  is better). The EDA flags exact-match as a trap on held-out data, so edit
  distance and block accuracy are the informative signals.
- **Token Accuracy** — position-wise matches / ``len(truth)``.
- **Block-level Accuracy** — LCS of the block-run shapes / ``len(truth blocks)``;
  rewards getting the process *shape* right even when exact steps differ.
"""

from __future__ import annotations

from zero_hack.eval.blocks import block_runs


def levenshtein(a: list[str], b: list[str]) -> int:
    """Token-level edit distance (insert/delete/substitute = cost 1)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, tok_a in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, tok_b in enumerate(b, start=1):
            cost = 0 if tok_a == tok_b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if tok_a == tok_b else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _token_accuracy(pred: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0 if not pred else 0.0
    matches = sum(1 for p, g in zip(pred, gold, strict=False) if p == g)
    return matches / len(gold)


def _normalized_edit_distance(pred: list[str], gold: list[str]) -> float:
    denom = max(len(pred), len(gold))
    if denom == 0:
        return 0.0
    return levenshtein(pred, gold) / denom


def _block_accuracy(pred: list[str], gold: list[str]) -> float:
    gold_runs = block_runs(gold)
    if not gold_runs:
        return 1.0 if not pred else 0.0
    return lcs_length(block_runs(pred), gold_runs) / len(gold_runs)


def score_completion(
    truth: dict[str, list[str]],
    predictions: dict[str, list[str]],
    families: dict[str, str] | None = None,
) -> dict:
    """Compute completion metrics over the shared example ids.

    A missing prediction is treated as an empty completion (worst case).
    """
    ids = sorted(truth)
    groups: dict[str, list[str]] = {"all": ids}
    if families:
        for example_id in ids:
            groups.setdefault(families.get(example_id, "unknown"), []).append(example_id)

    out: dict[str, dict] = {}
    for group, group_ids in groups.items():
        n = len(group_ids)
        exact = 0
        ned_sum = tok_sum = block_sum = 0.0
        for example_id in group_ids:
            gold = truth[example_id]
            pred = predictions.get(example_id, [])
            exact += int(pred == gold)
            ned_sum += _normalized_edit_distance(pred, gold)
            tok_sum += _token_accuracy(pred, gold)
            block_sum += _block_accuracy(pred, gold)
        out[group] = {
            "n": n,
            "exact_match": round(exact / n, 4) if n else 0.0,
            "norm_edit_distance": round(ned_sum / n, 4) if n else 0.0,
            "token_accuracy": round(tok_sum / n, 4) if n else 0.0,
            "block_accuracy": round(block_sum / n, 4) if n else 0.0,
        }
    return out
