from dataclasses import dataclass

PHASES: tuple[str, ...] = (
    "PREFIX",
    "INITIAL_MEASUREMENTS",
    "PRE_PROCESS_CLEAN",
    "FAMILY_SPECIFIC_PREP",
    "FIRST_OXIDATION",
    "PROCESS_CYCLE",
    "ILD_BLOCK",
    "VIA_BLOCK",
    "METAL_BLOCK",
    "PASSIVATION_BLOCK",
    "BACKSIDE_BLOCK",
    "FINAL_INSPECTION",
    "TEST_SUITE",
    "SUFFIX",
)

UNKNOWN_PHASE = "UNKNOWN"

_PHASE_INDEX = {phase: idx for idx, phase in enumerate(PHASES)}


@dataclass(frozen=True)
class StepPhaseRow:
    step: str
    primary_phase: str
    candidate_phases: tuple[str, ...]


_RULES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    # (candidate phases, needle, match mode) where mode is exact/prefix/substr.
    (("PREFIX",), "RECEIVE WAFER LOT", "exact"),
    (("PREFIX",), "LOT IDENTIFICATION", "exact"),
    (("PREFIX",), "INITIAL WAFER INSPECTION", "exact"),
    (("PREFIX",), "PRE CLEAN INSPECTION", "exact"),
    (("INITIAL_MEASUREMENTS", "BACKSIDE_BLOCK"), "MEASURE THICKNESS", "exact"),
    (("INITIAL_MEASUREMENTS",), "MEASURE INITIAL", "prefix"),
    (("INITIAL_MEASUREMENTS",), "MEASURE SURFACE PARTICLES", "exact"),
    (("INITIAL_MEASUREMENTS",), "MEASURE SURFACE DEFECTS", "exact"),
    (
        ("PRE_PROCESS_CLEAN", "FIRST_OXIDATION", "PROCESS_CYCLE", "ILD_BLOCK", "METAL_BLOCK"),
        "RCA CLEAN",
        "substr",
    ),
    (
        ("PRE_PROCESS_CLEAN", "FIRST_OXIDATION", "PROCESS_CYCLE", "ILD_BLOCK", "METAL_BLOCK"),
        "WET CLEAN RCA",
        "prefix",
    ),
    (
        ("PRE_PROCESS_CLEAN", "FIRST_OXIDATION", "PROCESS_CYCLE", "ILD_BLOCK", "METAL_BLOCK"),
        "HF DIP",
        "exact",
    ),
    (("PRE_PROCESS_CLEAN",), "PRE CLEAN WAFER", "exact"),
    (("PRE_PROCESS_CLEAN",), "WAFER CLEAN PRE PROCESS", "exact"),
    (("PRE_PROCESS_CLEAN", "FAMILY_SPECIFIC_PREP", "BACKSIDE_BLOCK"), "BACKSIDE CLEAN", "exact"),
    (("PRE_PROCESS_CLEAN", "FINAL_INSPECTION"), "FRONTSIDE CLEAN", "exact"),
    (("FINAL_INSPECTION",), "FRONTSIDE CLEAN FINAL", "exact"),
    (("PRE_PROCESS_CLEAN", "FAMILY_SPECIFIC_PREP"), "DRY WAFER", "prefix"),
    (
        ("FAMILY_SPECIFIC_PREP", "PROCESS_CYCLE", "ILD_BLOCK", "METAL_BLOCK"),
        "WAFER SURFACE CLEAN",
        "exact",
    ),
    (("FAMILY_SPECIFIC_PREP",), "SUBSTRATE CHECK", "exact"),
    (("FAMILY_SPECIFIC_PREP",), "EPITAXY", "substr"),
    (("FAMILY_SPECIFIC_PREP",), "EPITAXIAL", "substr"),
    (("FAMILY_SPECIFIC_PREP",), "WAFER CLEAN PRE-GRIND", "exact"),
    (("FAMILY_SPECIFIC_PREP",), "GRINDING WAFER BACKSIDE", "exact"),
    (("FAMILY_SPECIFIC_PREP",), "ETCH WET BACKSIDE", "exact"),
    (("FAMILY_SPECIFIC_PREP",), "RINSE WET WAFER_EDGE", "exact"),
    (("FAMILY_SPECIFIC_PREP",), "MEASURE GEOMETRY", "exact"),
    (("FAMILY_SPECIFIC_PREP",), "MEASURE BACKSIDE ROUGHNESS", "exact"),
    (("FAMILY_SPECIFIC_PREP",), "MEASURE RESISTIVITY", "exact"),
    (("FIRST_OXIDATION", "PROCESS_CYCLE"), "THERMAL OXIDATION", "exact"),
    (("FIRST_OXIDATION", "PROCESS_CYCLE"), "GATE OXIDE", "prefix"),
    (("FIRST_OXIDATION",), "SURFACE PREP FOR DEPOSITION", "exact"),
    (("FIRST_OXIDATION",), "DEPOSIT PAD OXIDE", "exact"),
    (("FIRST_OXIDATION",), "ANNEAL OXIDE", "exact"),
    (("FIRST_OXIDATION", "PROCESS_CYCLE"), "MEASURE OXIDE", "prefix"),
    (("FIRST_OXIDATION",), "OXIDE STRIP", "exact"),
    (("PROCESS_CYCLE", "VIA_BLOCK", "METAL_BLOCK", "PASSIVATION_BLOCK"), "SPIN COAT", "prefix"),
    (("PROCESS_CYCLE", "VIA_BLOCK", "METAL_BLOCK", "PASSIVATION_BLOCK"), "SOFT BAKE", "exact"),
    (("PROCESS_CYCLE", "VIA_BLOCK", "METAL_BLOCK", "PASSIVATION_BLOCK"), "ALIGN MASK", "prefix"),
    (("PROCESS_CYCLE", "VIA_BLOCK", "METAL_BLOCK", "PASSIVATION_BLOCK"), "EXPOSE LITHO", "prefix"),
    (
        ("PROCESS_CYCLE", "VIA_BLOCK", "METAL_BLOCK", "PASSIVATION_BLOCK"),
        "POST EXPOSE BAKE",
        "exact",
    ),
    (("PROCESS_CYCLE", "VIA_BLOCK", "METAL_BLOCK", "PASSIVATION_BLOCK"), "DEVELOP", "prefix"),
    (("PROCESS_CYCLE", "VIA_BLOCK", "METAL_BLOCK", "PASSIVATION_BLOCK"), "HARD BAKE", "exact"),
    (("PROCESS_CYCLE",), "INSPECT PATTERN", "prefix"),
    (("PROCESS_CYCLE",), "PATTERN INSPECTION", "prefix"),
    (("PROCESS_CYCLE",), "P BODY WINDOW INSPECTION", "exact"),
    (("PROCESS_CYCLE",), "FIELD PATTERN INSPECTION", "exact"),
    (("PROCESS_CYCLE",), "POLY PATTERN INSPECTION", "exact"),
    (("PROCESS_CYCLE",), "DEPOSIT POLYSILICON", "exact"),
    (("PROCESS_CYCLE",), "POLYSILICON", "prefix"),
    (("PROCESS_CYCLE",), "ANNEAL POLYSILICON", "exact"),
    (("PROCESS_CYCLE",), "DEPOSIT SPACER DIELECTRIC", "exact"),
    (("PROCESS_CYCLE",), "DEPOSIT FIELD OXIDE", "exact"),
    (("PROCESS_CYCLE",), "DEPOSIT GATE OXIDE OR DIELECTRIC", "exact"),
    (("PROCESS_CYCLE",), "OXIDE ETCH", "prefix"),
    (("PROCESS_CYCLE",), "POLYSILICON ETCH", "prefix"),
    (("PROCESS_CYCLE",), "ETCH SILICON OR OXIDE WINDOW", "exact"),
    (("PROCESS_CYCLE",), "FIELD OXIDE ETCH", "exact"),
    (("PROCESS_CYCLE",), "ANISOTROPIC ETCH SPACER", "exact"),
    (("PROCESS_CYCLE",), "CLEAN AFTER ETCH", "exact"),
    (("PROCESS_CYCLE",), "CLEAN AFTER OXIDE ETCH", "exact"),
    (("PROCESS_CYCLE",), "CLEAN AFTER POLY ETCH", "exact"),
    (("PROCESS_CYCLE",), "CLEAN AFTER WINDOW ETCH", "exact"),
    (("PROCESS_CYCLE",), "CLEAN AFTER FIELD ETCH", "exact"),
    (("PROCESS_CYCLE", "VIA_BLOCK", "METAL_BLOCK", "PASSIVATION_BLOCK"), "STRIP", "prefix"),
    (("PROCESS_CYCLE",), "IMPLANT", "prefix"),
    (("PROCESS_CYCLE",), "PRE ANNEAL CHECK", "exact"),
    (("PROCESS_CYCLE",), "RAPID THERMAL ANNEAL", "exact"),
    (("PROCESS_CYCLE",), "DRIVE IN DIFFUSION", "exact"),
    (("PROCESS_CYCLE",), "LIGHT ANNEAL", "exact"),
    (("PROCESS_CYCLE",), "ANNEAL DIELECTRIC", "exact"),
    (("PROCESS_CYCLE",), "MEASURE CD LEVEL", "prefix"),
    (("PROCESS_CYCLE",), "MEASURE OPENING CD", "exact"),
    (("PROCESS_CYCLE",), "MEASURE WINDOW CD", "exact"),
    (("PROCESS_CYCLE", "METAL_BLOCK"), "MEASURE LINE WIDTH", "exact"),
    (("PROCESS_CYCLE",), "MEASURE GATE", "prefix"),
    (("PROCESS_CYCLE",), "MEASURE POLY THICKNESS", "exact"),
    (("PROCESS_CYCLE",), "MEASURE SPACER WIDTH", "exact"),
    (("PROCESS_CYCLE",), "MEASURE SHEET RESISTANCE", "exact"),
    (("PROCESS_CYCLE",), "MEASURE JUNCTION", "prefix"),
    (("PROCESS_CYCLE",), "MEASURE DEVICE PARAMETER", "exact"),
    (("PROCESS_CYCLE",), "MEASURE SURFACE UNIFORMITY", "exact"),
    (("ILD_BLOCK", "PROCESS_CYCLE"), "DEPOSIT INTERLAYER DIELECTRIC", "exact"),
    (("ILD_BLOCK", "PROCESS_CYCLE"), "DEPOSIT INTERLEVEL DIELECTRIC", "exact"),
    (("ILD_BLOCK", "PROCESS_CYCLE"), "DENSIFY", "prefix"),
    (("ILD_BLOCK", "PROCESS_CYCLE"), "MEASURE FILM THICKNESS", "exact"),
    (("ILD_BLOCK",), "MEASURE DIELECTRIC THICKNESS", "exact"),
    (("ILD_BLOCK",), "CMP DIELECTRIC", "exact"),
    (("ILD_BLOCK",), "CMP INTERLAYER DIELECTRIC", "exact"),
    (("ILD_BLOCK",), "MEASURE PLANARITY", "exact"),
    (("ILD_BLOCK",), "MEASURE SURFACE PLANARITY", "exact"),
    (("VIA_BLOCK",), "VIA", "substr"),
    (("VIA_BLOCK",), "DEPOSIT BARRIER METAL", "exact"),
    (("VIA_BLOCK",), "DEPOSIT METAL SEED", "exact"),
    (("VIA_BLOCK",), "DEPOSIT TUNGSTEN SEED", "exact"),
    (("VIA_BLOCK",), "FILL VIA", "prefix"),
    (("VIA_BLOCK",), "CMP METAL", "exact"),
    (("VIA_BLOCK",), "CMP VIA FILL", "exact"),
    (("VIA_BLOCK",), "MEASURE CONTACT RESISTANCE", "exact"),
    (("METAL_BLOCK",), "DEPOSIT METAL 1", "exact"),
    (("METAL_BLOCK",), "DEPOSIT TOP METAL", "exact"),
    (("METAL_BLOCK",), "ANNEAL METAL", "prefix"),
    (("METAL_BLOCK",), "MEASURE METAL THICKNESS", "exact"),
    (("METAL_BLOCK",), "METAL PATTERN INSPECTION", "exact"),
    (("METAL_BLOCK",), "METAL ETCH", "prefix"),
    (("METAL_BLOCK",), "CLEAN AFTER METAL ETCH", "exact"),
    (("PASSIVATION_BLOCK",), "PASSIVATION", "substr"),
    (("PASSIVATION_BLOCK",), "OPEN PAD WINDOW", "substr"),
    (("PASSIVATION_BLOCK",), "OPEN BOND PAD WINDOW", "substr"),
    (("PASSIVATION_BLOCK",), "PAD WINDOW LITHO", "exact"),
    (("PASSIVATION_BLOCK",), "DEVELOP PAD WINDOW", "exact"),
    (("PASSIVATION_BLOCK",), "CLEAN PAD OPENING", "exact"),
    (("PASSIVATION_BLOCK",), "MEASURE PAD OPENING", "exact"),
    (("BACKSIDE_BLOCK",), "BACKSIDE", "substr"),
    (("BACKSIDE_BLOCK",), "MEASURE WAFER THICKNESS", "exact"),
    (("FINAL_INSPECTION",), "FINAL", "prefix"),
    (("TEST_SUITE",), "PARAMETRIC TEST", "exact"),
    (("TEST_SUITE",), "ELECTRICAL PARAMETRIC TEST", "exact"),
    (("TEST_SUITE",), "LEAKAGE TEST", "exact"),
    (("TEST_SUITE",), "THRESHOLD VOLTAGE TEST", "exact"),
    (("TEST_SUITE",), "BREAKDOWN VOLTAGE TEST", "exact"),
    (("TEST_SUITE",), "SWITCHING TEST", "exact"),
    (("TEST_SUITE",), "WAFER SORT TEST", "exact"),
    (("TEST_SUITE",), "YIELD ANALYSIS", "exact"),
    (("SUFFIX",), "LOT RELEASE", "exact"),
    (("SUFFIX",), "FINAL LOT RELEASE", "exact"),
    (("SUFFIX",), "PACKAGE PREPARATION", "exact"),
    (("SUFFIX",), "SHIP LOT", "exact"),
)

_LITHO_STEPS = {
    "SPIN COAT PHOTORESIST",
    "SOFT BAKE",
    "POST EXPOSE BAKE",
    "DEVELOP PHOTORESIST",
    "HARD BAKE",
}


def step_candidate_phases(step: str) -> tuple[str, ...]:
    """Return all plausible ordered process phases for one vocabulary step."""

    s = step.strip().upper()
    out: list[str] = []
    for phases, needle, mode in _RULES:
        if _matches(s, needle, mode):
            out.extend(phase for phase in phases if phase not in out)
    return tuple(out) if out else (UNKNOWN_PHASE,)


def primary_phase_for_step(step: str) -> str:
    """Return the first candidate phase for a step.

    This is useful for vocabulary-level metadata. Use ``steps_to_phases`` for
    per-position labels in a sequence, because some steps are context-dependent.
    """

    return step_candidate_phases(step)[0]


def build_step_phase_rows(steps: list[str] | tuple[str, ...] | set[str]) -> list[StepPhaseRow]:
    """Build a sorted step-to-phase table for CSV export or inspection."""

    rows: list[StepPhaseRow] = []
    for step in sorted(set(steps)):
        phases = step_candidate_phases(step)
        rows.append(
            StepPhaseRow(
                step=step,
                primary_phase=phases[0],
                candidate_phases=phases,
            )
        )
    return rows


def steps_to_phases(steps: list[str] | tuple[str, ...]) -> list[str]:
    """Map sequence positions to ordered process phases.

    The labeler is monotonic over the documented backbone and resolves generic
    lithography/strip steps by looking at the upcoming patterned etch.
    """

    labels: list[str] = []
    current = "PREFIX"

    for idx, step in enumerate(steps):
        contextual = _contextual_phase(steps, idx, current)
        if contextual is None:
            contextual = _monotonic_candidate(step, current)
        current = contextual
        labels.append(current)

    return labels


def phase_runs(steps: list[str] | tuple[str, ...]) -> list[str]:
    """Phase labels with consecutive duplicates collapsed."""

    runs: list[str] = []
    for phase in steps_to_phases(steps):
        if not runs or runs[-1] != phase:
            runs.append(phase)
    return runs


def _contextual_phase(
    steps: list[str] | tuple[str, ...],
    idx: int,
    current: str,
) -> str | None:
    s = steps[idx].strip().upper()

    if s.startswith("ALIGN MASK") or s.startswith("EXPOSE LITHO") or s in _LITHO_STEPS:
        return _phase_from_next_patterned_etch(steps, idx) or current

    if s.startswith("STRIP"):
        return _phase_from_previous_patterned_etch(steps, idx) or current

    if s == "DEVELOP PAD WINDOW":
        return "PASSIVATION_BLOCK"

    if s in {"DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC"} and _next_in(
        steps,
        idx,
        {"CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC"},
        window=8,
    ):
        return "ILD_BLOCK"

    if (s.startswith("DENSIFY") or s == "MEASURE FILM THICKNESS") and current in {
        "ILD_BLOCK",
        "PROCESS_CYCLE",
        "FIRST_OXIDATION",
    }:
        return current

    if s == "MEASURE LINE WIDTH" and current in {"PROCESS_CYCLE", "METAL_BLOCK"}:
        return current

    if s == "MEASURE THICKNESS" and current in {"BACKSIDE_BLOCK", "INITIAL_MEASUREMENTS"}:
        return current

    if (
        s in {"BACKSIDE CLEAN", "BACKSIDE CLEAN FINAL"}
        and _PHASE_INDEX[current] >= _PHASE_INDEX["PASSIVATION_BLOCK"]
    ):
        return "BACKSIDE_BLOCK"

    if s == "BACKSIDE CLEAN" and current == "FAMILY_SPECIFIC_PREP":
        return "FAMILY_SPECIFIC_PREP"

    return None


def _monotonic_candidate(step: str, current: str) -> str:
    candidates = step_candidate_phases(step)
    if candidates == (UNKNOWN_PHASE,):
        return UNKNOWN_PHASE

    current_idx = _PHASE_INDEX.get(current, 0)
    forward = [phase for phase in candidates if _PHASE_INDEX[phase] >= current_idx]
    if forward:
        return min(forward, key=lambda phase: _PHASE_INDEX[phase])
    return current


def _phase_from_next_patterned_etch(
    steps: list[str] | tuple[str, ...],
    idx: int,
    window: int = 12,
) -> str | None:
    for step in steps[idx + 1 : idx + 1 + window]:
        s = step.strip().upper()
        if "VIA ETCH" in s or s == "DIELECTRIC ETCH VIA":
            return "VIA_BLOCK"
        if s.startswith("METAL ETCH"):
            return "METAL_BLOCK"
        if s.startswith("PASSIVATION ETCH"):
            return "PASSIVATION_BLOCK"
        if (
            s.startswith("OXIDE ETCH")
            or s.startswith("POLYSILICON ETCH")
            or s == "ETCH SILICON OR OXIDE WINDOW"
            or s == "FIELD OXIDE ETCH"
        ):
            return "PROCESS_CYCLE"
    return None


def _phase_from_previous_patterned_etch(
    steps: list[str] | tuple[str, ...],
    idx: int,
    window: int = 4,
) -> str | None:
    start = max(0, idx - window)
    for step in reversed(steps[start:idx]):
        s = step.strip().upper()
        if "VIA ETCH" in s or s == "DIELECTRIC ETCH VIA":
            return "VIA_BLOCK"
        if s.startswith("METAL ETCH"):
            return "METAL_BLOCK"
        if s.startswith("PASSIVATION ETCH"):
            return "PASSIVATION_BLOCK"
        if (
            s.startswith("OXIDE ETCH")
            or s.startswith("POLYSILICON ETCH")
            or s == "ETCH SILICON OR OXIDE WINDOW"
            or s == "FIELD OXIDE ETCH"
        ):
            return "PROCESS_CYCLE"
    return None


def _next_in(
    steps: list[str] | tuple[str, ...],
    idx: int,
    targets: set[str],
    *,
    window: int,
) -> bool:
    return any(step.strip().upper() in targets for step in steps[idx + 1 : idx + 1 + window])


def _matches(step: str, needle: str, mode: str) -> bool:
    if mode == "exact":
        return step == needle
    if mode == "prefix":
        return step.startswith(needle)
    if mode == "substr":
        return needle in step
    raise ValueError(f"Unknown match mode: {mode}")
