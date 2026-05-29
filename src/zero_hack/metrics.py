"""Shared evaluation metrics for next-step prediction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TopKAccumulator:
    """Streaming top-1 / top-k accuracy, broken down by group (e.g. family)."""

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
        g = self.by_group[group]
        g[0] += 1
        g[1] += int(is_top1)
        g[2] += int(is_topk)

    def summary(self) -> dict[str, dict[str, float]]:
        def rates(total: int, t1: int, tk: int) -> dict[str, float]:
            if total == 0:
                return {"n": 0, "top1": 0.0, f"top{self.k}": 0.0}
            return {
                "n": total,
                "top1": round(t1 / total, 4),
                f"top{self.k}": round(tk / total, 4),
            }

        out = {"all": rates(self.total, self.top1, self.topk)}
        for group, (n, t1, tk) in sorted(self.by_group.items()):
            out[group] = rates(n, t1, tk)
        return out
