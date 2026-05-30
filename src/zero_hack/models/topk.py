from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TopKAccumulator:
    k: int = 3
    total: int = 0
    top1: int = 0
    topk: int = 0
    by_group: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0]))

    def update(self, gold: int, ranked_preds: list[int], group: str = "all") -> None:
        self.total += 1
        is_top1 = bool(ranked_preds) and ranked_preds[0] == gold
        is_topk = gold in ranked_preds[: self.k]
        self.top1 += int(is_top1)
        self.topk += int(is_topk)
        counts = self.by_group[group]
        counts[0] += 1
        counts[1] += int(is_top1)
        counts[2] += int(is_topk)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {"all": self._rates(self.total, self.top1, self.topk)}
        for group, (total, top1, topk) in sorted(self.by_group.items()):
            out[group] = self._rates(total, top1, topk)
        return out

    def _rates(self, total: int, top1: int, topk: int) -> dict[str, float]:
        if total == 0:
            return {"n": 0, "top1": 0.0, f"top{self.k}": 0.0}
        return {
            "n": total,
            "top1": round(top1 / total, 4),
            f"top{self.k}": round(topk / total, 4),
        }
