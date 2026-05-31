import argparse
from collections import defaultdict

from zero_hack.data import SequenceRecord, Vocabulary
from zero_hack.models.common import DEFAULT_SPLITS_DIR, load_split_records
from zero_hack.models.ngram.model import NGramModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Classic n-gram next-step baseline.")
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS_DIR))
    parser.add_argument("--limit-per-family", type=int, default=None)
    parser.add_argument("--holdout-family", choices=("mosfet", "igbt", "ic"), default=None)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    bundle = load_split_records(
        splits_dir=args.splits_dir,
        holdout_family=args.holdout_family,
        limit_per_family=args.limit_per_family,
    )
    print(f"counts: {bundle.counts()}")

    model = NGramModel(n=args.n, backoff_alpha=args.alpha).fit(bundle.records["train"])

    for split in bundle.test_split_names:
        summary = _evaluate_topk(model, bundle.records[split], bundle.vocabulary, k=args.k)
        label = split.removeprefix("test_")
        role = "ood" if label == bundle.holdout_family else "id"
        print(f"{split} ({role}) summary: {summary}")


def _evaluate_topk(
    model: NGramModel,
    records: list[SequenceRecord],
    vocabulary: Vocabulary,
    *,
    k: int,
) -> dict:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for record in records:
        for position, gold_step in enumerate(record.steps):
            preds = model.predict_topk(record.family, record.steps[:position], k=k)
            gold_id = vocabulary.token_to_id.get(gold_step, vocabulary.unk_id)
            pred_ids = [vocabulary.token_to_id.get(step, vocabulary.unk_id) for step in preds]
            is_top1 = bool(pred_ids) and pred_ids[0] == gold_id
            is_topk = gold_id in pred_ids[:k]
            for group in ("all", record.family):
                counts = totals[group]
                counts[0] += 1
                counts[1] += int(is_top1)
                counts[2] += int(is_topk)
    return {
        group: _topk_rates(total, top1, topk_count, k)
        for group, (total, top1, topk_count) in sorted(totals.items())
    }


def _topk_rates(total: int, top1: int, topk: int, k: int) -> dict[str, float]:
    if total == 0:
        return {"n": 0, "top1": 0.0, f"top{k}": 0.0}
    return {
        "n": total,
        "top1": round(top1 / total, 4),
        f"top{k}": round(topk / total, 4),
    }


if __name__ == "__main__":
    main()
