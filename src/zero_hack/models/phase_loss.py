"""Auxiliary next-phase loss utilities for next-step sequence models.

Typical GPT training integration:

    token_loss = token_criterion(token_logits, target_ids)
    phase_loss = next_phase_loss(token_logits, target_ids)
    loss = token_loss + phase_loss

Use ``PhaseTargetLookup`` to pass contextual phase labels instead of deriving
targets from the target token's primary vocabulary phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, SequenceRecord, Vocabulary
from zero_hack.eval.phases import PHASES, UNKNOWN_PHASE, primary_phase_for_step, steps_to_phases

DEFAULT_PHASE_IGNORE_INDEX = -100

__all__ = [
    "DEFAULT_PHASE_IGNORE_INDEX",
    "NextPhaseLoss",
    "PhaseLogitProjector",
    "PhaseLossOutput",
    "PhaseTargetLookup",
    "build_token_phase_ids",
    "phase_id_lookup",
    "phase_targets_from_token_ids",
]


def phase_id_lookup(phases: tuple[str, ...] = PHASES) -> dict[str, int]:
    """Return stable phase-to-id labels matching ``eval.phases.PHASES`` order."""

    return {phase: idx for idx, phase in enumerate(phases)}


def build_token_phase_ids(
    vocabulary: Vocabulary,
    *,
    phases: tuple[str, ...] = PHASES,
    ignore_index: int = DEFAULT_PHASE_IGNORE_INDEX,
) -> torch.Tensor:
    """Map each vocabulary token id to a phase id.

    Special tokens, family tokens, and steps unknown to the phase labeler are
    mapped to ``ignore_index``. For ambiguous process steps this uses the
    vocabulary-level primary phase. Use ``PhaseTargetLookup`` when exact
    sequence-position context is available.
    """

    phase_to_id = phase_id_lookup(phases)
    ignored_tokens = set(SPECIAL_TOKENS) | set(FAMILY_TOKENS.values())
    ids = torch.full((len(vocabulary.id_to_token),), ignore_index, dtype=torch.long)

    for token_id, token in enumerate(vocabulary.id_to_token):
        if token in ignored_tokens:
            continue

        phase = primary_phase_for_step(token)
        if phase == UNKNOWN_PHASE:
            continue
        ids[token_id] = phase_to_id[phase]

    return ids


def phase_targets_from_token_ids(
    target_ids: torch.Tensor,
    token_phase_ids: torch.Tensor,
    *,
    ignore_index: int = DEFAULT_PHASE_IGNORE_INDEX,
) -> torch.Tensor:
    """Convert next-token ids to next-phase ids with ignored invalid targets."""

    token_phase_ids = token_phase_ids.to(device=target_ids.device)
    safe_target_ids = target_ids.clamp(min=0, max=token_phase_ids.numel() - 1)
    phase_targets = token_phase_ids[safe_target_ids]
    return torch.where(target_ids == safe_target_ids, phase_targets, ignore_index)


@dataclass(frozen=True)
class PhaseTargetLookup:
    """Contextual next-phase labels keyed by dataset batch metadata."""

    targets_by_position: dict[tuple[str, str, int], int]
    id_to_phase: tuple[str, ...] = PHASES
    ignore_index: int = DEFAULT_PHASE_IGNORE_INDEX

    @classmethod
    def from_records(
        cls,
        records: list[SequenceRecord] | tuple[SequenceRecord, ...],
        *,
        phases: tuple[str, ...] = PHASES,
        ignore_index: int = DEFAULT_PHASE_IGNORE_INDEX,
    ) -> PhaseTargetLookup:
        phase_to_id = phase_id_lookup(phases)
        targets: dict[tuple[str, str, int], int] = {}

        for record in records:
            for position, phase in enumerate(steps_to_phases(record.steps)):
                targets[(record.family, record.sequence_id, position)] = phase_to_id.get(
                    phase,
                    ignore_index,
                )

        return cls(
            targets_by_position=targets,
            id_to_phase=phases,
            ignore_index=ignore_index,
        )

    def target_for(self, family: str, sequence_id: str, position: int) -> int:
        return self.targets_by_position.get(
            (family, sequence_id, int(position)),
            self.ignore_index,
        )

    def targets_for_batch(
        self,
        batch: dict[str, Any],
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build a tensor of phase targets from a standard next-step batch."""

        positions = batch["position"]
        if isinstance(positions, torch.Tensor):
            position_values = positions.detach().cpu().tolist()
            if device is None:
                device = positions.device
        else:
            position_values = list(positions)

        targets = [
            self.target_for(family, sequence_id, position)
            for family, sequence_id, position in zip(
                batch["family"],
                batch["sequence_id"],
                position_values,
                strict=True,
            )
        ]
        return torch.tensor(targets, dtype=torch.long, device=device)


class PhaseLogitProjector(nn.Module):
    """Project next-token logits into next-phase logits by log-sum-exp pooling."""

    def __init__(
        self,
        token_phase_ids: torch.Tensor,
        *,
        num_phases: int = len(PHASES),
        ignore_index: int = DEFAULT_PHASE_IGNORE_INDEX,
    ) -> None:
        super().__init__()
        if token_phase_ids.ndim != 1:
            raise ValueError("token_phase_ids must be a 1D tensor")

        token_phase_ids = token_phase_ids.detach().to(dtype=torch.long)
        phase_mask = torch.zeros(num_phases, token_phase_ids.numel(), dtype=torch.bool)
        for phase_id in range(num_phases):
            phase_mask[phase_id] = token_phase_ids == phase_id

        self.num_phases = num_phases
        self.ignore_index = ignore_index
        self.register_buffer("token_phase_ids", token_phase_ids, persistent=False)
        self.register_buffer("phase_mask", phase_mask, persistent=False)

    @classmethod
    def from_vocabulary(
        cls,
        vocabulary: Vocabulary,
        *,
        phases: tuple[str, ...] = PHASES,
        ignore_index: int = DEFAULT_PHASE_IGNORE_INDEX,
    ) -> PhaseLogitProjector:
        return cls(
            build_token_phase_ids(
                vocabulary,
                phases=phases,
                ignore_index=ignore_index,
            ),
            num_phases=len(phases),
            ignore_index=ignore_index,
        )

    def forward(self, token_logits: torch.Tensor) -> torch.Tensor:
        """Return phase logits with shape ``(..., num_phases)``."""

        if token_logits.shape[-1] != self.token_phase_ids.numel():
            raise ValueError(
                "token_logits last dimension must match token_phase_ids length: "
                f"{token_logits.shape[-1]} != {self.token_phase_ids.numel()}"
            )

        mask = self.phase_mask.to(device=token_logits.device)
        fill_value = torch.finfo(token_logits.dtype).min
        expanded = token_logits.unsqueeze(-2).masked_fill(~mask, fill_value)
        return torch.logsumexp(expanded, dim=-1)


@dataclass(frozen=True)
class PhaseLossOutput:
    loss: torch.Tensor
    phase_loss: torch.Tensor
    phase_logits: torch.Tensor
    phase_targets: torch.Tensor


class NextPhaseLoss(nn.Module):
    """Weighted next-phase auxiliary loss derived from next-token logits."""

    def __init__(
        self,
        token_phase_ids: torch.Tensor,
        *,
        weight: float = 0.1,
        num_phases: int = len(PHASES),
        ignore_index: int = DEFAULT_PHASE_IGNORE_INDEX,
    ) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError("weight must be non-negative")
        self.weight = weight
        self.ignore_index = ignore_index
        self.projector = PhaseLogitProjector(
            token_phase_ids,
            num_phases=num_phases,
            ignore_index=ignore_index,
        )

    @classmethod
    def from_vocabulary(
        cls,
        vocabulary: Vocabulary,
        *,
        phases: tuple[str, ...] = PHASES,
        weight: float = 0.1,
        ignore_index: int = DEFAULT_PHASE_IGNORE_INDEX,
    ) -> NextPhaseLoss:
        return cls(
            build_token_phase_ids(
                vocabulary,
                phases=phases,
                ignore_index=ignore_index,
            ),
            weight=weight,
            num_phases=len(phases),
            ignore_index=ignore_index,
        )

    def forward(
        self,
        token_logits: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        *,
        phase_targets: torch.Tensor | None = None,
        return_output: bool = False,
    ) -> torch.Tensor | PhaseLossOutput:
        """Compute weighted auxiliary loss.

        Pass ``target_ids`` for a drop-in vocabulary-derived target, or pass
        contextual ``phase_targets`` from ``PhaseTargetLookup`` when available.
        """

        if phase_targets is None:
            if target_ids is None:
                raise ValueError("target_ids or phase_targets must be provided")
            phase_targets = phase_targets_from_token_ids(
                target_ids,
                self.projector.token_phase_ids,
                ignore_index=self.ignore_index,
            )
        else:
            phase_targets = phase_targets.to(device=token_logits.device)

        phase_logits = self.projector(token_logits)
        valid = phase_targets != self.ignore_index
        if valid.any():
            phase_loss = F.cross_entropy(
                phase_logits,
                phase_targets,
                ignore_index=self.ignore_index,
            )
        else:
            phase_loss = phase_logits.sum() * 0.0

        loss = phase_loss * self.weight
        if return_output:
            return PhaseLossOutput(
                loss=loss,
                phase_loss=phase_loss,
                phase_logits=phase_logits,
                phase_targets=phase_targets,
            )
        return loss
