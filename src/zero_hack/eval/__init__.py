"""Shared evaluation-protocol scoring for the three Industrial AI tasks.

Implements the metrics in ``data/industrial/generation_rules.md`` §5 against the
documented submission formats:

- Task 1 (next step): Top-1/3/5 accuracy, MRR  -> :mod:`zero_hack.eval.next_step`
- Task 2 (completion): exact match, normalized edit distance, token accuracy,
  block-level accuracy                          -> :mod:`zero_hack.eval.completion`
- Task 3 (anomaly): accuracy, P/R/F1, confusion, ROC-AUC, rule attribution
                                                 -> :mod:`zero_hack.eval.anomaly`

The canonical 10-rule validator is reused (not re-implemented) via
:mod:`zero_hack.eval.validator`. These scorers are dependency-free and mirror
the organizer ``eval_metrics.py`` interface; when that script ships, prefer it.
"""

from zero_hack.eval.anomaly import score_anomaly
from zero_hack.eval.completion import score_completion
from zero_hack.eval.next_step import score_next_step
from zero_hack.eval.score import TASKS, score_task

__all__ = ["TASKS", "score_anomaly", "score_completion", "score_next_step", "score_task"]
