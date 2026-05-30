import math

import numpy as np
import xgboost as xgb

from zero_hack.data import SequenceRecord
from zero_hack.models.topk import TopKAccumulator
from zero_hack.models.xgboost.features import FeatureExtractor

_SCORE_FLOOR = 1e-9


class XGBoostNextStep:
    """Multiclass next-step classifier over engineered process-state features."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.3,
        lag: int = 8,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        n_jobs: int = -1,
        seed: int = 1729,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.seed = seed
        self.extractor = FeatureExtractor(lag=lag)
        self.label_to_step: list[str] = []
        self.step_to_label: dict[str, int] = {}
        self.booster: xgb.Booster | None = None

    def _fit_labels(self, records: list[SequenceRecord]) -> None:
        steps = sorted({step for record in records for step in record.steps})
        self.label_to_step = steps
        self.step_to_label = {step: idx for idx, step in enumerate(steps)}
        self.extractor.set_vocab(steps)

    def fit(self, records: list[SequenceRecord]) -> "XGBoostNextStep":
        self._fit_labels(records)
        matrices = []
        labels: list[int] = []
        for record in records:
            matrices.append(self.extractor.sequence_matrix(record.family, record.steps))
            labels.extend(self.step_to_label[step] for step in record.steps)
        features = np.concatenate(matrices, axis=0)
        targets = np.asarray(labels, dtype=np.int32)

        dtrain = xgb.DMatrix(features, label=targets)
        params = {
            "objective": "multi:softprob",
            "num_class": len(self.label_to_step),
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": "hist",
            "seed": self.seed,
        }
        if self.n_jobs > 0:
            params["nthread"] = self.n_jobs
        self.booster = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        return self

    def _proba(self, features: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model is not fitted")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self.booster.predict(xgb.DMatrix(features))

    def predict_topk(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        k: int = 3,
    ) -> list[str]:
        row = self.extractor.prefix_row(family, prefix_steps)
        proba = self._proba(row)[0]
        ranked = np.argsort(-proba)[:k]
        return [self.label_to_step[idx] for idx in ranked]

    def score_sequence(
        self,
        family: str,
        steps: list[str] | tuple[str, ...],
    ) -> float:
        features = self.extractor.sequence_matrix(family, steps)
        if features.shape[0] == 0:
            return 0.0
        proba = self._proba(features)
        total = 0.0
        for position, step in enumerate(steps):
            label = self.step_to_label.get(step, -1)
            prob = proba[position, label] if 0 <= label < proba.shape[1] else 0.0
            total += math.log(max(prob, _SCORE_FLOOR))
        return total

    def evaluate(
        self,
        records: list[SequenceRecord],
        vocabulary,
        k: int = 3,
    ) -> dict:
        acc = TopKAccumulator(k=k)
        for record in records:
            features = self.extractor.sequence_matrix(record.family, record.steps)
            if features.shape[0] == 0:
                continue
            proba = self._proba(features)
            ranked = np.argsort(-proba, axis=1)[:, :k]
            for position, step in enumerate(record.steps):
                gold_id = vocabulary.token_to_id.get(step, vocabulary.unk_id)
                pred_ids = [
                    vocabulary.token_to_id.get(self.label_to_step[idx], vocabulary.unk_id)
                    for idx in ranked[position]
                ]
                acc.update(gold_id, pred_ids, group=record.family)
        return acc.summary()
