from __future__ import annotations

import math
from dataclasses import dataclass, field

from zero_hack.data import SequenceRecord
from zero_hack.eval.validator import validate_sequence
from zero_hack.models.ngram import NGramModel

TERMINATOR = "SHIP LOT"
DEFAULT_TARGET_LENGTHS = {
    "ic": 115,
    "mosfet": 125,
    "igbt": 148,
}

_MILESTONE_PREDICATES = (
    lambda step: step == "RECEIVE WAFER LOT",
    lambda step: step in {"PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS"},
    lambda step: step in {"HF DIP", "OXIDE STRIP"},
    lambda step: step.startswith("THERMAL OXIDATION") or step.startswith("GATE OXIDE"),
    lambda step: step.startswith("ALIGN MASK LEVEL"),
    lambda step: step == "DEVELOP PHOTORESIST",
    lambda step: step in {"DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC"},
    lambda step: "VIA" in step,
    lambda step: "METAL" in step,
    lambda step: step in {"DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"},
    lambda step: step == "CURE PASSIVATION",
    lambda step: step == "WAFER SORT TEST",
    lambda step: step == TERMINATOR,
)

_FAMILY_MANDATORY = {
    "mosfet": (
        {"EPITAXIAL DEPOSITION"},
        {"GATE OXIDE PREP", "GATE OXIDE GROWTH"},
        {"THRESHOLD VOLTAGE TEST"},
    ),
    "igbt": (
        {"IMPLANT P BODY"},
        {"IMPLANT N BUFFER"},
        {"BREAKDOWN VOLTAGE TEST"},
    ),
    "ic": (
        {"GRINDING WAFER BACKSIDE"},
        {"DEPOSIT PAD OXIDE"},
        {"PACKAGE PREPARATION"},
    ),
}


@dataclass(frozen=True)
class RewardConfig:
    min_length: int = 100
    max_length: int = 200
    hard_invalid_log_reward: float = -50.0
    hard_prefix_log_reward: float = -80.0
    hard_terminal_log_reward: float = -60.0
    hard_length_log_reward: float = -35.0
    valid_bonus: float = 20.0
    terminal_bonus: float = 5.0
    phase_bonus: float = 8.0
    family_bonus: float = 4.0
    novelty_bonus: float = 1.0
    memorization_penalty: float = 2.0
    length_weight: float = 2.0
    length_sigma: float = 18.0
    style_weight: float = 0.15
    style_floor: float = -12.0
    ngram_order: int = 5
    log_reward_min: float = -80.0
    log_reward_max: float = 80.0
    family_target_lengths: dict[str, int] = field(default_factory=lambda: DEFAULT_TARGET_LENGTHS)


@dataclass(frozen=True)
class RewardBreakdown:
    log_reward: float
    reward: float
    is_valid: bool
    violations: tuple[str, ...]
    components: dict[str, float]


class ProcessReward:
    """Non-differentiable neurosymbolic reward for completed process sequences."""

    def __init__(
        self,
        train_records: list[SequenceRecord] | None = None,
        *,
        config: RewardConfig | None = None,
        known_steps: set[str] | None = None,
    ) -> None:
        self.config = config or RewardConfig()
        self.known_steps = known_steps or {
            step for record in train_records or [] for step in record.steps
        }
        self.train_sequences = {
            (record.family, tuple(record.steps)) for record in train_records or []
        }
        self.style_model = None
        if train_records and self.config.style_weight:
            self.style_model = NGramModel(n=self.config.ngram_order).fit(train_records)

    def evaluate(
        self,
        family: str,
        prefix_steps: list[str] | tuple[str, ...],
        steps: list[str] | tuple[str, ...],
    ) -> RewardBreakdown:
        family = family.lower()
        prefix = tuple(prefix_steps)
        sequence = tuple(steps)

        hard_gate = self._hard_gate(family, prefix, sequence)
        if hard_gate is not None:
            reason, log_reward, violations = hard_gate
            return _breakdown(
                log_reward,
                is_valid=False,
                violations=violations,
                components={reason: log_reward},
                config=self.config,
            )

        violations = tuple(violation.rule for violation in validate_sequence(list(sequence)))
        if violations:
            log_reward = self.config.hard_invalid_log_reward - 10.0 * len(violations)
            return _breakdown(
                log_reward,
                is_valid=False,
                violations=violations,
                components={"rule_violation": log_reward},
                config=self.config,
            )

        components = {
            "validity": self.config.valid_bonus,
            "termination": self.config.terminal_bonus,
            "length": self._length_score(family, len(sequence)),
            "phase": self._phase_score(sequence),
            "family": self._family_score(family, sequence),
            "style": self._style_score(family, sequence),
            "novelty": self._novelty_score(family, sequence),
        }
        return _breakdown(
            sum(components.values()),
            is_valid=True,
            violations=(),
            components=components,
            config=self.config,
        )

    def _hard_gate(
        self,
        family: str,
        prefix: tuple[str, ...],
        sequence: tuple[str, ...],
    ) -> tuple[str, float, tuple[str, ...]] | None:
        if sequence[: len(prefix)] != prefix:
            return "prefix_mismatch", self.config.hard_prefix_log_reward, ("PREFIX_MISMATCH",)
        if self.known_steps:
            unknown = sorted(set(sequence) - self.known_steps)
            if unknown:
                return "unknown_step", self.config.hard_prefix_log_reward, ("UNKNOWN_STEP",)
        if not sequence or sequence[-1] != TERMINATOR:
            return "missing_terminal", self.config.hard_terminal_log_reward, ("MISSING_SHIP_LOT",)
        if TERMINATOR in sequence[:-1]:
            return "early_terminal", self.config.hard_terminal_log_reward, ("EARLY_SHIP_LOT",)
        if len(sequence) < self.config.min_length or len(sequence) > self.config.max_length:
            distance = min(
                abs(len(sequence) - self.config.min_length),
                abs(len(sequence) - self.config.max_length),
            )
            return (
                "length_gate",
                self.config.hard_length_log_reward - 0.25 * distance,
                ("LENGTH_OUT_OF_RANGE",),
            )
        if family not in self.config.family_target_lengths:
            return "unknown_family", self.config.hard_prefix_log_reward, ("UNKNOWN_FAMILY",)
        return None

    def _length_score(self, family: str, length: int) -> float:
        target = self.config.family_target_lengths[family]
        z = (length - target) / self.config.length_sigma
        return -self.config.length_weight * z * z

    def _phase_score(self, sequence: tuple[str, ...]) -> float:
        cursor = 0
        matched = 0
        for predicate in _MILESTONE_PREDICATES:
            while cursor < len(sequence) and not predicate(sequence[cursor]):
                cursor += 1
            if cursor >= len(sequence):
                break
            matched += 1
            cursor += 1
        return self.config.phase_bonus * matched / len(_MILESTONE_PREDICATES)

    def _family_score(self, family: str, sequence: tuple[str, ...]) -> float:
        required_groups = _FAMILY_MANDATORY.get(family, ())
        if not required_groups:
            return 0.0
        present = set(sequence)
        matched = sum(any(step in present for step in group) for group in required_groups)
        return self.config.family_bonus * matched / len(required_groups)

    def _style_score(self, family: str, sequence: tuple[str, ...]) -> float:
        if self.style_model is None:
            return 0.0
        avg_logprob = self.style_model.score_sequence(family, sequence) / max(1, len(sequence))
        return self.config.style_weight * max(self.config.style_floor, avg_logprob)

    def _novelty_score(self, family: str, sequence: tuple[str, ...]) -> float:
        if not self.train_sequences:
            return 0.0
        if (family, sequence) in self.train_sequences:
            return -self.config.memorization_penalty
        return self.config.novelty_bonus


def _breakdown(
    log_reward: float,
    *,
    is_valid: bool,
    violations: tuple[str, ...],
    components: dict[str, float],
    config: RewardConfig,
) -> RewardBreakdown:
    clipped = min(config.log_reward_max, max(config.log_reward_min, log_reward))
    return RewardBreakdown(
        log_reward=clipped,
        reward=math.exp(clipped),
        is_valid=is_valid,
        violations=violations,
        components=components,
    )
