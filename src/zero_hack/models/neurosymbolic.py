from zero_hack.eval.validator import validate_sequence


class SymbolicMaskAdapter:
    def __init__(self, base, vocabulary, pool=30):
        self.base = base
        self.vocabulary = vocabulary
        self.pool = pool

    def _legal(self, prefix, step):
        index = len(prefix)
        return not any(v.step_index == index for v in validate_sequence([*prefix, step]))

    def predict_topk(self, family, prefix_steps, k=3):
        prefix = list(prefix_steps)
        candidates = self.base.predict_topk(family, prefix, max(k, self.pool))
        legal = [c for c in candidates if self._legal(prefix, c)]
        ranked = legal if legal else candidates
        return ranked[:k]

    def score_sequence(self, family, steps):
        return self.base.score_sequence(family, steps)
