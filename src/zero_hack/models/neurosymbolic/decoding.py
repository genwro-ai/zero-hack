"""Family-agnostic neurosymbolic decoding heads for process-step generation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import torch

from zero_hack.data import FAMILY_TOKENS, SPECIAL_TOKENS, Vocabulary
from zero_hack.eval.validator import _generator_module, validate_sequence

HeadMode = Literal["none", "hard", "shaped"]

_ALIGN_LEVEL = re.compile(r"^ALIGN MASK LEVEL (\d+)$")
_SPECIAL_OUTPUT_TOKENS = frozenset(SPECIAL_TOKENS) | frozenset(FAMILY_TOKENS.values())

_PHASE_ORDER = (
    "start",
    "initial_measure",
    "clean_prep",
    "frontside",
    "via",
    "metal",
    "passivation",
    "backside",
    "final_inspection",
    "test",
    "ship",
)
_PHASE_INDEX = {phase: idx for idx, phase in enumerate(_PHASE_ORDER)}


@dataclass(frozen=True)
class ProcessShapingWeights:
    """Soft shaping coefficients applied after hard rule filtering."""

    obligation: float = 1.2
    phase: float = 0.8
    block_stack: float = 0.9
    progress: float = 0.5
    repair: float = 0.7


@dataclass(frozen=True)
class ProcessState:
    phase: str
    phase_index: int
    litho_stage: str | None
    current_litho_level: int
    resist_on: bool
    etch_debt: bool
    implant_debt: bool
    deposition_debt: bool
    via_debt: bool
    passivation_deposited: bool
    passivation_cured: bool
    wafer_sorted: bool
    terminal_started: bool
    last_step: str | None


class NeurosymbolicHardHead:
    """Mask next-step candidates that introduce one of the 10 process-rule violations."""

    def shape_logits(
        self,
        prefix: list[str] | tuple[str, ...],
        logits: torch.Tensor,
        vocabulary: Vocabulary,
    ) -> torch.Tensor:
        shaped = logits.clone()
        for token_id, step in enumerate(vocabulary.id_to_token):
            if not _is_process_step(step) or _introduces_rule_violation(prefix, step):
                shaped[token_id] = -torch.inf
        return shaped


class NeurosymbolicShapedHead(NeurosymbolicHardHead):
    """Apply hard rule filtering plus family-agnostic soft process-route shaping."""

    def __init__(self, weights: ProcessShapingWeights | None = None) -> None:
        self.weights = weights or ProcessShapingWeights()

    def shape_logits(
        self,
        prefix: list[str] | tuple[str, ...],
        logits: torch.Tensor,
        vocabulary: Vocabulary,
    ) -> torch.Tensor:
        shaped = super().shape_logits(prefix, logits, vocabulary)
        state = infer_process_state(prefix)
        for token_id, step in enumerate(vocabulary.id_to_token):
            if not torch.isfinite(shaped[token_id]):
                continue
            shaped[token_id] = shaped[token_id] + _soft_score(step, state, self.weights)
        return shaped


def shape_logits(
    prefix: list[str] | tuple[str, ...],
    logits: torch.Tensor,
    vocabulary: Vocabulary,
    *,
    mode: HeadMode = "hard",
    weights: ProcessShapingWeights | None = None,
) -> torch.Tensor:
    """Shape next-token logits with a selected neurosymbolic head."""
    if mode == "none":
        shaped = logits.clone()
        for token_id, step in enumerate(vocabulary.id_to_token):
            if not _is_process_step(step):
                shaped[token_id] = -torch.inf
        return shaped
    if mode == "hard":
        return NeurosymbolicHardHead().shape_logits(prefix, logits, vocabulary)
    if mode == "shaped":
        return NeurosymbolicShapedHead(weights).shape_logits(prefix, logits, vocabulary)
    raise ValueError(f"Unknown neurosymbolic head mode: {mode}")


def infer_process_state(prefix: list[str] | tuple[str, ...]) -> ProcessState:
    phase = "start"
    litho_stage: str | None = None
    current_litho_level = 0
    resist_on = False
    etch_debt = False
    implant_debt = False
    deposition_debt = False
    via_debt = False
    passivation_deposited = False
    passivation_cured = False
    wafer_sorted = False
    terminal_started = False
    last_step = None

    for step in prefix:
        last_step = step
        phase = _advance_phase(phase, _phase_for_step(step))
        category = _category_for_step(step)

        align_match = _ALIGN_LEVEL.match(step)
        if align_match:
            current_litho_level = max(current_litho_level, int(align_match.group(1)))

        if category == "litho_start":
            resist_on = True
            litho_stage = "spin"
        elif step == "SOFT BAKE":
            litho_stage = "soft_bake"
        elif category == "litho_align":
            litho_stage = "aligned"
        elif category == "litho_expose":
            litho_stage = "exposed"
        elif step == "POST EXPOSE BAKE":
            litho_stage = "post_expose_bake"
        elif category == "litho_develop":
            litho_stage = "developed"
        elif category in {"litho_inspect", "hard_bake"}:
            litho_stage = "inspected"
        elif category in {"etch", "implant"} and resist_on:
            litho_stage = "patterned"
        elif category == "strip":
            resist_on = False
            litho_stage = None

        if category == "etch":
            etch_debt = True
            if _is_via_step(step):
                via_debt = True
        elif category == "strip":
            etch_debt = True
        elif category == "clean":
            etch_debt = False
            via_debt = False
            deposition_debt = False

        if category == "implant":
            implant_debt = True
        elif category == "anneal" or _is_implant_measurement(step):
            implant_debt = False

        if category == "deposition":
            deposition_debt = True
        elif category in {"measure", "anneal", "cmp", "litho_start"}:
            deposition_debt = False

        if category == "via_fill" or _is_via_measurement(step):
            via_debt = False

        if step in {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"}:
            passivation_deposited = True
        elif step == "CURE PASSIVATION":
            passivation_cured = True

        if step == "WAFER SORT TEST":
            wafer_sorted = True
        if _phase_for_step(step) in {"final_inspection", "test", "ship"}:
            terminal_started = True

    return ProcessState(
        phase=phase,
        phase_index=_PHASE_INDEX[phase],
        litho_stage=litho_stage,
        current_litho_level=current_litho_level,
        resist_on=resist_on,
        etch_debt=etch_debt,
        implant_debt=implant_debt,
        deposition_debt=deposition_debt,
        via_debt=via_debt,
        passivation_deposited=passivation_deposited,
        passivation_cured=passivation_cured,
        wafer_sorted=wafer_sorted,
        terminal_started=terminal_started,
        last_step=last_step,
    )


def _soft_score(step: str, state: ProcessState, weights: ProcessShapingWeights) -> float:
    return (
        weights.obligation * _obligation_score(step, state)
        + weights.phase * _phase_score(step, state)
        + weights.block_stack * _block_stack_score(step, state)
        + weights.progress * _progress_score(step, state)
        + weights.repair * _repair_score(step, state)
    )


def _obligation_score(step: str, state: ProcessState) -> float:
    category = _category_for_step(step)
    score = 0.0
    if state.passivation_deposited and not state.passivation_cured:
        score += 1.4 if step == "CURE PASSIVATION" else -0.35
    if state.etch_debt:
        if category == "strip":
            score += 0.8
        elif category == "clean":
            score += 1.0
        elif category in {"deposition", "test", "ship"}:
            score -= 0.8
    if state.via_debt:
        score += 1.0 if category in {"clean", "via_fill", "measure"} else -0.3
    if state.implant_debt:
        score += 0.8 if category in {"anneal", "measure"} else -0.15
    if state.deposition_debt:
        score += 0.4 if category in {"measure", "anneal", "cmp", "litho_start"} else 0.0
    return score


def _phase_score(step: str, state: ProcessState) -> float:
    candidate_phase = _phase_for_step(step)
    candidate_index = _PHASE_INDEX[candidate_phase]
    if candidate_phase == "start":
        return 0.0
    if state.terminal_started and candidate_index < _PHASE_INDEX["final_inspection"]:
        return -2.0
    if candidate_index in {state.phase_index, state.phase_index + 1}:
        return 0.4
    if candidate_index > state.phase_index + 2:
        return -0.8
    if candidate_index < state.phase_index - 2:
        return -0.5
    return 0.0


def _block_stack_score(step: str, state: ProcessState) -> float:
    category = _category_for_step(step)
    if not state.resist_on:
        return 0.0
    expected_by_stage = {
        "spin": {"soft_bake"},
        "soft_bake": {"litho_align"},
        "aligned": {"litho_expose"},
        "exposed": {"litho_develop", "post_expose_bake"},
        "post_expose_bake": {"litho_develop"},
        "developed": {"litho_inspect", "etch", "implant", "hard_bake"},
        "inspected": {"etch", "implant", "hard_bake"},
        "patterned": {"strip"},
    }
    expected = expected_by_stage.get(state.litho_stage or "", set())
    if category in expected or step in expected:
        return 1.0
    if category in {"deposition", "test", "ship", "backside"}:
        return -0.75
    return 0.0


def _progress_score(step: str, state: ProcessState) -> float:
    category = _category_for_step(step)
    if step == state.last_step and category != "measure":
        return -0.6
    if category in {"test", "ship"} and _has_open_debt(state):
        return -1.0
    if category in {"strip", "clean", "anneal", "via_fill"} and _has_open_debt(state):
        return 0.5
    if step == "SHIP LOT" and state.wafer_sorted:
        return 0.8
    return 0.0


def _repair_score(step: str, state: ProcessState) -> float:
    category = _category_for_step(step)
    score = 0.0
    if state.etch_debt and category in {"strip", "clean"}:
        score += 1.0
    if state.implant_debt and category in {"anneal", "measure"}:
        score += 0.7
    if state.deposition_debt and category in {"measure", "anneal", "cmp"}:
        score += 0.5
    if state.via_debt and category in {"clean", "via_fill", "measure"}:
        score += 0.8
    if state.passivation_deposited and not state.passivation_cured and step == "CURE PASSIVATION":
        score += 1.0
    return score


def _has_open_debt(state: ProcessState) -> bool:
    return (
        state.resist_on
        or state.etch_debt
        or state.implant_debt
        or state.via_debt
        or (state.passivation_deposited and not state.passivation_cured)
    )


def _is_process_step(step: str) -> bool:
    return step not in _SPECIAL_OUTPUT_TOKENS and step != "<UNK_STEP>"


def _introduces_rule_violation(prefix: list[str] | tuple[str, ...], candidate: str) -> bool:
    candidate_index = len(prefix)
    return any(
        violation.step_index == candidate_index
        for violation in validate_sequence([*prefix, candidate])
    )


def _category_for_step(step: str) -> str:
    generator = _generator_module()
    if step in generator.CLEAN_STEPS:
        return "clean"
    if step in generator.CMP_STEPS:
        return "cmp"
    if step in generator.FILL_STEPS and step not in generator.DEPOSITION_STEPS:
        return "via_fill"
    if step in generator.ETCH_STEPS or step == "ANISOTROPIC ETCH SPACER":
        return "etch"
    if step in generator.IMPLANT_STEPS:
        return "implant"
    if step in generator.ELECTRICAL_TEST_STEPS or step in {"WAFER SORT TEST", "YIELD ANALYSIS"}:
        return "test"
    if step in generator.DEPOSITION_STEPS:
        return "deposition"
    if step == "SPIN COAT PHOTORESIST":
        return "litho_start"
    if step == "SOFT BAKE":
        return "soft_bake"
    if step.startswith("ALIGN MASK LEVEL"):
        return "litho_align"
    if step.startswith("EXPOSE LITHO LEVEL"):
        return "litho_expose"
    if step == "POST EXPOSE BAKE":
        return "post_expose_bake"
    if step in {"DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"}:
        return "litho_develop"
    if step == "HARD BAKE":
        return "hard_bake"
    if "PATTERN INSPECTION" in step or step.startswith("INSPECT PATTERN"):
        return "litho_inspect"
    if step.startswith("STRIP ") or "STRIP RESIST" in step:
        return "strip"
    if "ANNEAL" in step or step in {"DRIVE IN DIFFUSION", "LIGHT ANNEAL"}:
        return "anneal"
    if step.startswith("MEASURE ") or "INSPECTION" in step or step.endswith("CHECK"):
        return "measure"
    if "PASSIVATION" in step or "PAD WINDOW" in step:
        return "passivation"
    if "BACKSIDE" in step or step in {"BACKSIDE GRIND", "BACKSIDE DRY"}:
        return "backside"
    if step.startswith("FINAL "):
        return "final_inspection"
    if step in {"SHIP LOT", "LOT RELEASE", "FINAL LOT RELEASE", "PACKAGE PREPARATION"}:
        return "ship"
    return "other"


def _phase_for_step(step: str) -> str:
    category = _category_for_step(step)
    if step in {"RECEIVE WAFER LOT", "LOT IDENTIFICATION"}:
        return "start"
    if step.startswith("INITIAL ") or step.startswith("MEASURE INITIAL"):
        return "initial_measure"
    if category == "backside":
        return "backside"
    if category == "final_inspection" or step.startswith("FINAL "):
        return "final_inspection"
    if category == "test":
        return "test"
    if step in {"SHIP LOT", "LOT RELEASE", "FINAL LOT RELEASE", "PACKAGE PREPARATION"}:
        return "ship"
    if "PASSIVATION" in step or "PAD WINDOW" in step:
        return "passivation"
    if _is_via_step(step):
        return "via"
    if "METAL" in step and "BACKSIDE" not in step:
        return "metal"
    if category == "clean" and step not in {"CLEAN AFTER ETCH", "CLEAN PAD OPENING"}:
        return "clean_prep"
    if category in {
        "deposition",
        "etch",
        "implant",
        "anneal",
        "cmp",
        "litho_start",
        "litho_align",
        "litho_expose",
        "litho_develop",
        "litho_inspect",
        "strip",
        "measure",
        "other",
    }:
        return "frontside"
    return "start"


def _advance_phase(current: str, candidate: str) -> str:
    return candidate if _PHASE_INDEX[candidate] > _PHASE_INDEX[current] else current


def _is_via_step(step: str) -> bool:
    return "VIA" in step or "TUNGSTEN" in step


def _is_via_measurement(step: str) -> bool:
    return step in {"MEASURE VIA CD", "MEASURE VIA RESISTANCE"}


def _is_implant_measurement(step: str) -> bool:
    return step in {
        "MEASURE SHEET RESISTANCE",
        "MEASURE JUNCTION DEPTH",
        "MEASURE JUNCTION PROFILE",
        "MEASURE DEVICE PARAMETER",
    }


def _topk_from_logits(logits: torch.Tensor, k: int) -> list[int]:
    finite = torch.isfinite(logits)
    if not bool(finite.any()):
        return []
    safe_k = min(k, int(finite.sum().item()))
    return torch.topk(logits, k=safe_k).indices.tolist()


def topk_steps(logits: torch.Tensor, vocabulary: Vocabulary, *, k: int = 5) -> list[str]:
    """Return top-k step strings from already-shaped logits."""
    return [vocabulary.id_to_token[token_id] for token_id in _topk_from_logits(logits, k)]


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Stable softmax over finite logits; all-masked rows return zeros."""
    finite = torch.isfinite(logits)
    if not bool(finite.any()):
        return torch.zeros_like(logits)
    shifted = torch.where(finite, logits, torch.tensor(-math.inf, device=logits.device))
    return torch.softmax(shifted, dim=-1)
