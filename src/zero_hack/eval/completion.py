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


def _token_accuracy(pred: list[str], gold: list[str]) -> float:
    n = min(len(pred), len(gold))
    if n == 0:
        return 0.0
    return sum(p == g for p, g in zip(pred, gold, strict=False)) / n


def _normalized_edit_distance(pred: list[str], gold: list[str]) -> float:
    denom = max(len(pred), len(gold))
    if denom == 0:
        return 0.0
    return levenshtein(pred, gold) / denom


def _block_accuracy(pred: list[str], gold: list[str]) -> float:
    return _token_accuracy(_block_signature(pred), _block_signature(gold))


def _major_block(step: str) -> str:
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
