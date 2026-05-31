"""A process-rule masking head for autoregressive step prediction.

``ViolationMask`` is a parameter-free, training-free layer placed *after* a
model's logits. Given the next-step logits and the steps decoded so far, it
sends the logit of every step that would trigger one of the 10 process-logic
rules to ``-inf``, so ``argmax`` / sampling over the result can only ever pick a
rule-compliant continuation.

It is architecture-agnostic: the LSTM, the Transformer, or any other next-step
model can share the same instance - it only needs the vocabulary's
``id_to_token`` order and the prefix of real steps emitted so far. All rule
knowledge is sourced from ``generate_sequences.py`` through
``zero_hack.eval.validator``; nothing about the rules is reimplemented here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from zero_hack.eval.validator import trigger_steps, validate_sequence

_ALIGN_PREFIX = "ALIGN MASK LEVEL "


def _introduces_violation(prefix: list[str], candidate: str) -> bool:
    """True if appending ``candidate`` to ``prefix`` triggers a rule at its position.

    Every rule attributes its violation to the offending trigger step, so a
    candidate is illegal exactly when a new violation appears at index
    ``len(prefix)``. Earlier violations (a prefix that is already invalid) are
    ignored - masking only prevents *new* ones.
    """
    position = len(prefix)
    return any(v.step_index == position for v in validate_sequence([*prefix, candidate]))


class ViolationMask(nn.Module):
    """Mask next-step logits that would break a process-logic rule.

    Parameters
    ----------
    id_to_token:
        The vocabulary in id order (e.g. ``vocabulary.id_to_token``).
    masked_value:
        Value written into illegal positions (``-inf`` by default, so they drop
        out of ``argmax``, ``topk`` and ``softmax``).
    is_legal:
        Optional ``(prefix, candidate) -> bool`` predicate. Defaults to the
        validator-backed check. Inject a faster incremental predicate here
        without touching the layer if profiling ever demands it.
    """

    def __init__(
        self,
        id_to_token: Sequence[str],
        *,
        masked_value: float = float("-inf"),
        is_legal: Callable[[list[str], str], bool] | None = None,
    ) -> None:
        super().__init__()
        self._id_to_token = tuple(id_to_token)
        self._masked_value = masked_value
        self._is_legal = is_legal or (lambda prefix, cand: not _introduces_violation(prefix, cand))
        # Only steps that can ever trigger a rule are worth testing; every other
        # token is legal in any context and passes straight through.
        triggers = trigger_steps()
        self._maskable: tuple[tuple[int, str], ...] = tuple(
            (i, tok)
            for i, tok in enumerate(self._id_to_token)
            if tok in triggers or tok.startswith(_ALIGN_PREFIX)
        )

    def legal_mask(self, prefix: Sequence[str]) -> torch.Tensor:
        """Boolean vector over the vocab: ``True`` where the step is rule-compliant."""
        mask = torch.ones(len(self._id_to_token), dtype=torch.bool)
        prefix_list = list(prefix)
        for i, tok in self._maskable:
            if not self._is_legal(prefix_list, tok):
                mask[i] = False
        return mask

    def forward(
        self, logits: torch.Tensor, prefix: Sequence[str] | Sequence[Sequence[str]]
    ) -> torch.Tensor:
        """Return ``logits`` with rule-violating positions set to ``masked_value``.

        ``logits`` is either ``[vocab]`` with a single ``prefix`` (list of steps),
        or ``[batch, vocab]`` with one prefix per row.
        """
        if logits.dim() == 1:
            mask = self.legal_mask(prefix).to(logits.device)  # type: ignore[arg-type]
            return logits.masked_fill(~mask, self._masked_value)
        masks = torch.stack([self.legal_mask(p) for p in prefix]).to(logits.device)
        return logits.masked_fill(~masks, self._masked_value)
