from __future__ import annotations

import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from zero_hack.data.datasets import FAMILY_FILE_NAMES, normalize_family
from zero_hack.eval.validator import validate_sequence

SynonymStyle = Literal["random", "canonical", "alternate"]


@dataclass(frozen=True)
class AugmentationOptions:
    """Controls for the variation axes documented in generation_rules.md section 4."""

    litho_cycles: int | None = None
    post_expose_bake: bool | None = None
    hard_bake: bool | None = None
    intermediate_clean: bool | None = None
    extra_measurements: bool | None = None
    dry_wafer: bool | None = None
    epitaxial_rework_check: bool | None = None
    pre_anneal_check: bool | None = None
    second_metal_layer: bool | None = False
    cmp_after_via_fill: bool | None = None
    synonym_style: SynonymStyle = "random"


def _flag(rng: random.Random, value: bool | None, prob: float = 0.5) -> bool:
    return rng.random() < prob if value is None else value


def _choice(
    rng: random.Random,
    options: tuple[str, ...],
    style: SynonymStyle,
) -> str:
    if style == "canonical":
        return options[0]
    if style == "alternate":
        return options[-1]
    return rng.choice(options)


def _optional_measure(
    rng: random.Random,
    options: str | tuple[str, ...],
    cfg: AugmentationOptions,
    prob: float = 0.75,
) -> list[str]:
    if not _flag(rng, cfg.extra_measurements, prob):
        return []
    if isinstance(options, str):
        return [options]
    return [_choice(rng, options, cfg.synonym_style)]


def _pre_anneal(rng: random.Random, cfg: AugmentationOptions) -> list[str]:
    return ["PRE ANNEAL CHECK"] if _flag(rng, cfg.pre_anneal_check, 0.6) else []


def _strip(rng: random.Random, cfg: AugmentationOptions) -> str:
    return _choice(rng, ("STRIP PHOTORESIST", "STRIP RESIST"), cfg.synonym_style)


def _rca1(rng: random.Random, cfg: AugmentationOptions) -> str:
    return _choice(rng, ("RCA CLEAN 1", "WET CLEAN RCA1"), cfg.synonym_style)


def _rca2(rng: random.Random, cfg: AugmentationOptions) -> str:
    return _choice(rng, ("RCA CLEAN 2", "WET CLEAN RCA2"), cfg.synonym_style)


def _litho(
    rng: random.Random,
    cfg: AugmentationOptions,
    level: int,
    inspection: str | None = None,
) -> list[str]:
    steps = [
        "SPIN COAT PHOTORESIST",
        "SOFT BAKE",
        f"ALIGN MASK LEVEL {level}",
        f"EXPOSE LITHO LEVEL {level}",
    ]
    if _flag(rng, cfg.post_expose_bake, 0.3):
        steps.append("POST EXPOSE BAKE")
    steps += ["DEVELOP PHOTORESIST", inspection or f"INSPECT PATTERN LEVEL {level}"]
    if _flag(rng, cfg.hard_bake, 0.3):
        steps.append("HARD BAKE")
    return steps


def _intermediate_clean(rng: random.Random, cfg: AugmentationOptions) -> list[str]:
    if not _flag(rng, cfg.intermediate_clean, 0.25):
        return []
    return [_rca1(rng, cfg), _rca2(rng, cfg), "HF DIP"]


def _gen_prefix(rng: random.Random) -> list[str]:
    return [
        "RECEIVE WAFER LOT",
        "LOT IDENTIFICATION",
        rng.choice(["INITIAL WAFER INSPECTION", "PRE CLEAN INSPECTION"]),
    ]


def _gen_initial_measurements(
    rng: random.Random,
    family: str,
    cfg: AugmentationOptions,
) -> list[str]:
    thickness = {
        "mosfet": ("MEASURE THICKNESS",),
        "igbt": ("MEASURE INITIAL THICKNESS",),
        "ic": ("MEASURE INITIAL GEOMETRY", "MEASURE INITIAL THICKNESS"),
    }[family]
    surface = {
        "mosfet": ("MEASURE SURFACE PARTICLES",),
        "igbt": ("MEASURE SURFACE PARTICLES",),
        "ic": ("MEASURE SURFACE DEFECTS", "MEASURE SURFACE PARTICLES"),
    }[family]
    return [
        *_optional_measure(rng, thickness, cfg, 0.85),
        *_optional_measure(rng, surface, cfg, 0.85),
    ]


def _gen_pre_process_clean(
    rng: random.Random,
    family: str,
    cfg: AugmentationOptions,
) -> list[str]:
    steps = ["WAFER CLEAN PRE PROCESS" if family == "ic" else "PRE CLEAN WAFER"]
    if family == "igbt" or rng.random() > 0.5:
        steps.append("BACKSIDE CLEAN")
    if family == "igbt" or rng.random() > 0.6:
        steps.append("FRONTSIDE CLEAN")
    steps += [_rca1(rng, cfg), _rca2(rng, cfg), "HF DIP"]
    if _flag(rng, cfg.dry_wafer, 0.6):
        steps.append("DRY WAFER")
    return steps


def _gen_family_prep(
    rng: random.Random,
    family: str,
    cfg: AugmentationOptions,
) -> list[str]:
    if family == "mosfet":
        return [
            "SUBSTRATE CHECK",
            "EPITAXY PREP",
            "EPITAXIAL DEPOSITION",
            *_optional_measure(rng, "MEASURE EPITAXY THICKNESS", cfg),
            *_optional_measure(rng, "MEASURE RESISTIVITY", cfg),
            "EPITAXY ANNEAL",
            "WAFER SURFACE CLEAN",
        ]
    if family == "igbt":
        steps = [
            "EPITAXIAL WAFER CHECK",
            *_optional_measure(rng, "MEASURE EPITAXY THICKNESS", cfg),
            *_optional_measure(rng, "MEASURE RESISTIVITY", cfg),
        ]
        if _flag(rng, cfg.epitaxial_rework_check, 0.5):
            steps.append("EPITAXIAL REWORK CHECK")
        steps.append("EPITAXIAL LAYER PREP")
        return steps
    return [
        "WAFER CLEAN PRE-GRIND",
        "GRINDING WAFER BACKSIDE",
        *_optional_measure(rng, ("MEASURE GEOMETRY", "MEASURE INITIAL GEOMETRY"), cfg),
        "ETCH WET BACKSIDE",
        "RINSE WET WAFER_EDGE",
        "DRY WAFER BACKSIDE",
        "BACKSIDE CLEAN",
        *_optional_measure(rng, "MEASURE BACKSIDE ROUGHNESS", cfg),
    ]


def _gen_first_oxidation(
    rng: random.Random,
    family: str,
    cfg: AugmentationOptions,
) -> list[str]:
    steps = ["THERMAL OXIDATION", *_optional_measure(rng, "MEASURE OXIDE THICKNESS", cfg)]
    if family == "ic":
        steps += [
            _rca1(rng, cfg),
            _rca2(rng, cfg),
            "HF DIP",
            "OXIDE STRIP",
            "SURFACE PREP FOR DEPOSITION",
            "DEPOSIT PAD OXIDE",
            "ANNEAL OXIDE",
            *_optional_measure(rng, ("MEASURE FILM THICKNESS", "MEASURE OXIDE THICKNESS"), cfg),
        ]
    return steps


def _implant_tail(
    rng: random.Random,
    cfg: AugmentationOptions,
    implant_steps: list[str],
    anneal: str = "RAPID THERMAL ANNEAL",
) -> list[str]:
    return [*implant_steps, *_pre_anneal(rng, cfg), anneal]


def _mosfet_base_cycles(rng: random.Random, cfg: AugmentationOptions) -> list[str]:
    steps = []
    steps += _litho(rng, cfg, 1, "PATTERN INSPECTION LEVEL 1")
    steps += ["OXIDE ETCH", _strip(rng, cfg), "CLEAN AFTER ETCH"]
    steps += _optional_measure(rng, "MEASURE OPENING CD", cfg)
    steps += _implant_tail(rng, cfg, ["IMPLANT WELL", "DRIVE IN DIFFUSION"])
    steps += _optional_measure(rng, "MEASURE JUNCTION DEPTH", cfg)

    steps += ["THERMAL OXIDATION", "GATE OXIDE PREP", "GATE OXIDE GROWTH"]
    steps += _optional_measure(rng, "MEASURE GATE OXIDE THICKNESS", cfg)
    steps += ["DEPOSIT POLYSILICON", "POLYSILICON ANNEAL"]
    steps += _optional_measure(rng, "MEASURE POLY THICKNESS", cfg)
    steps += _litho(rng, cfg, 2, "POLY PATTERN INSPECTION")
    steps += ["POLYSILICON ETCH", _strip(rng, cfg), "CLEAN AFTER POLY ETCH"]
    steps += _optional_measure(rng, "MEASURE GATE CD", cfg)
    steps += ["IMPLANT SOURCE DRAIN", *_pre_anneal(rng, cfg), "LIGHT ANNEAL"]
    steps += _optional_measure(rng, "MEASURE SHEET RESISTANCE", cfg)
    steps += ["DEPOSIT SPACER DIELECTRIC", "ANISOTROPIC ETCH SPACER"]
    steps += _optional_measure(rng, "MEASURE SPACER WIDTH", cfg)
    steps += _implant_tail(rng, cfg, ["IMPLANT LDD"])
    steps += _optional_measure(rng, "MEASURE JUNCTION PROFILE", cfg)
    return steps


def _igbt_base_cycles(
    rng: random.Random,
    cfg: AugmentationOptions,
    *,
    include_poly_gate_cycle: bool,
) -> list[str]:
    steps = []
    steps += _litho(rng, cfg, 1, "INSPECT PATTERN LEVEL 1")
    steps += ["OXIDE ETCH DRY", _strip(rng, cfg), "CLEAN AFTER OXIDE ETCH"]
    steps += _optional_measure(rng, "MEASURE OPENING CD", cfg)
    steps += _implant_tail(rng, cfg, ["IMPLANT P BODY", "DRIVE IN DIFFUSION"])
    steps += _optional_measure(rng, "MEASURE JUNCTION DEPTH", cfg)

    steps += ["THERMAL OXIDATION"]
    steps += _litho(rng, cfg, 2, "P BODY WINDOW INSPECTION")
    steps += ["ETCH SILICON OR OXIDE WINDOW", _strip(rng, cfg), "CLEAN AFTER WINDOW ETCH"]
    steps += _optional_measure(rng, "MEASURE WINDOW CD", cfg)
    steps += _implant_tail(rng, cfg, ["IMPLANT N BUFFER"])
    steps += _optional_measure(rng, "MEASURE SHEET RESISTANCE", cfg)
    if _flag(rng, cfg.epitaxial_rework_check, 0.4):
        steps.append("EPITAXIAL REWORK CHECK")

    steps += ["DEPOSIT FIELD OXIDE", "DENSIFY OXIDE"]
    steps += _optional_measure(rng, "MEASURE FILM THICKNESS", cfg)
    steps += _litho(rng, cfg, 3, "FIELD PATTERN INSPECTION")
    steps += ["FIELD OXIDE ETCH", _strip(rng, cfg), "CLEAN AFTER FIELD ETCH"]
    steps += _optional_measure(rng, "MEASURE SURFACE UNIFORMITY", cfg)
    steps += _implant_tail(rng, cfg, ["IMPLANT SOURCE REGION", "IMPLANT DRAIN / CATHODE REGION"])
    steps += _optional_measure(rng, "MEASURE SHEET RESISTANCE", cfg)

    if include_poly_gate_cycle:
        steps += ["DEPOSIT GATE OXIDE OR DIELECTRIC", "ANNEAL DIELECTRIC"]
        steps += _optional_measure(rng, "MEASURE OXIDE QUALITY", cfg)
        steps += ["DEPOSIT POLYSILICON", "POLYSILICON ANNEAL"]
        steps += _optional_measure(rng, "MEASURE POLY THICKNESS", cfg)
        steps += _litho(rng, cfg, 4, "POLY PATTERN INSPECTION")
        steps += ["POLYSILICON ETCH DRY", _strip(rng, cfg), "CLEAN AFTER POLY ETCH"]
        steps += _optional_measure(rng, "MEASURE GATE CD", cfg)
        steps += _implant_tail(rng, cfg, ["IMPLANT CHANNEL STOP"])
        steps += _optional_measure(rng, "MEASURE DEVICE PARAMETER", cfg)
    return steps


def _ic_base_cycles(rng: random.Random, cfg: AugmentationOptions) -> list[str]:
    steps = []
    steps += _litho(rng, cfg, 1, "INSPECT PATTERN LEVEL 1")
    steps += ["OXIDE ETCH DRY", _strip(rng, cfg), "CLEAN AFTER ETCH"]
    steps += _optional_measure(rng, "MEASURE CD LEVEL 1", cfg)
    steps += ["DEPOSIT POLYSILICON", "ANNEAL POLYSILICON"]
    steps += _litho(rng, cfg, 2, "PATTERN INSPECTION LEVEL 2")
    steps += ["POLYSILICON ETCH DRY", _strip(rng, cfg), "CLEAN AFTER POLY ETCH"]
    steps += _optional_measure(rng, "MEASURE CD LEVEL 2", cfg)
    steps += _implant_tail(rng, cfg, ["IMPLANT N-TYPE"])
    steps += _optional_measure(rng, "MEASURE SHEET RESISTANCE", cfg)
    return steps


def _generic_core_cycle(
    rng: random.Random,
    cfg: AugmentationOptions,
    level: int,
    family: str,
) -> list[str]:
    etch = "OXIDE ETCH DRY" if family in {"igbt", "ic"} else "OXIDE ETCH"
    steps = ["THERMAL OXIDATION"]
    steps += _optional_measure(rng, "MEASURE OXIDE THICKNESS", cfg)
    steps += _litho(rng, cfg, level, f"INSPECT PATTERN LEVEL {level}")
    steps += [etch, _strip(rng, cfg), "CLEAN AFTER ETCH"]
    steps += _optional_measure(rng, f"MEASURE CD LEVEL {level}", cfg)
    steps += _implant_tail(
        rng, cfg, ["IMPLANT N-TYPE" if family == "ic" else "IMPLANT SOURCE REGION"]
    )
    steps += _optional_measure(rng, "MEASURE SHEET RESISTANCE", cfg)
    return steps


def _core_cycle_count(rng: random.Random, family: str, cfg: AugmentationOptions) -> int:
    if cfg.litho_cycles is None:
        return {"mosfet": 2, "igbt": 4, "ic": 2}[family]
    if not 3 <= cfg.litho_cycles <= 6:
        raise ValueError(f"{family} litho_cycles must be between 3 and 6")
    return cfg.litho_cycles


def _gen_core_cycles(
    rng: random.Random,
    family: str,
    cfg: AugmentationOptions,
) -> tuple[list[str], int]:
    target = _core_cycle_count(rng, family, cfg)
    if family == "igbt":
        base = _igbt_base_cycles(rng, cfg, include_poly_gate_cycle=target >= 4)
        base_levels = 4 if target >= 4 else 3
    else:
        base = {
            "mosfet": _mosfet_base_cycles,
            "ic": _ic_base_cycles,
        }[family](rng, cfg)
        base_levels = {"mosfet": 2, "ic": 2}[family]
    steps = list(base)
    for level in range(base_levels + 1, target + 1):
        steps += _generic_core_cycle(rng, cfg, level, family)
    return steps, target + 1


def _gen_ild_block(
    rng: random.Random,
    cfg: AugmentationOptions,
) -> list[str]:
    return [
        _choice(
            rng,
            ("DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC"),
            cfg.synonym_style,
        ),
        _choice(rng, ("DENSIFY DIELECTRIC", "DENSIFY OXIDE"), cfg.synonym_style),
        *_optional_measure(rng, ("MEASURE FILM THICKNESS", "MEASURE DIELECTRIC THICKNESS"), cfg),
        _choice(rng, ("CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC"), cfg.synonym_style),
        *_optional_measure(rng, ("MEASURE PLANARITY", "MEASURE SURFACE PLANARITY"), cfg),
    ]


def _gen_via_fill(
    rng: random.Random,
    family: str,
    cfg: AugmentationOptions,
) -> list[str]:
    steps = ["DEPOSIT BARRIER METAL"]
    if family == "ic":
        steps += ["DEPOSIT TUNGSTEN SEED", "FILL VIA TUNGSTEN"]
    else:
        steps += ["DEPOSIT METAL SEED", "FILL VIA METAL"]
    if _flag(rng, cfg.cmp_after_via_fill, 0.8):
        steps.append(_choice(rng, ("CMP VIA FILL", "CMP METAL"), cfg.synonym_style))
    steps += _optional_measure(
        rng,
        ("MEASURE CONTACT RESISTANCE", "MEASURE VIA RESISTANCE"),
        cfg,
    )
    return steps


def _gen_via_block(
    rng: random.Random,
    cfg: AugmentationOptions,
    level: int,
    family: str,
) -> list[str]:
    steps = _litho(
        rng,
        cfg,
        level,
        _choice(rng, ("VIA INSPECTION", "VIA OPENING INSPECTION"), cfg.synonym_style),
    )
    steps += [
        _choice(
            rng,
            ("VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA"),
            cfg.synonym_style,
        ),
        _strip(rng, cfg),
        "CLEAN AFTER VIA ETCH",
    ]
    steps += _optional_measure(rng, "MEASURE VIA CD", cfg)
    steps += _gen_via_fill(rng, family, cfg)
    return steps


def _gen_metal_block(
    rng: random.Random,
    cfg: AugmentationOptions,
    level: int,
    family: str,
    layer: int = 1,
) -> list[str]:
    metal_dep = "DEPOSIT METAL 1" if layer == 1 else "DEPOSIT TOP METAL"
    etch = "METAL ETCH DRY" if family in {"igbt", "ic"} else "METAL ETCH"
    steps = [metal_dep, _choice(rng, ("ANNEAL METAL 1", "ANNEAL METAL"), cfg.synonym_style)]
    steps += _optional_measure(rng, "MEASURE METAL THICKNESS", cfg, 0.45)
    steps += _litho(rng, cfg, level, "METAL PATTERN INSPECTION")
    steps += [etch, _strip(rng, cfg), "CLEAN AFTER METAL ETCH"]
    steps += _optional_measure(rng, "MEASURE LINE WIDTH", cfg)
    return steps


def _gen_passivation_block(
    rng: random.Random,
    cfg: AugmentationOptions,
) -> list[str]:
    return [
        _choice(rng, ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"), cfg.synonym_style),
        "CURE PASSIVATION",
        *_optional_measure(
            rng, ("MEASURE PASSIVATION THICKNESS", "MEASURE PASSIVATION QUALITY"), cfg
        ),
        _choice(rng, ("OPEN PAD WINDOW", "OPEN BOND PAD WINDOW"), cfg.synonym_style),
        _choice(rng, ("PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO"), cfg.synonym_style),
        _choice(rng, ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW"), cfg.synonym_style),
        _choice(
            rng,
            ("PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH"),
            cfg.synonym_style,
        ),
        _strip(rng, cfg),
        "CLEAN PAD OPENING",
        *_optional_measure(rng, "MEASURE PAD OPENING", cfg),
    ]


def _gen_backside_block(
    rng: random.Random,
    family: str,
    cfg: AugmentationOptions,
) -> list[str]:
    if family == "ic":
        steps = []
        if _flag(rng, cfg.extra_measurements, 0.6):
            steps.append("BACKSIDE THINNING CHECK")
        steps += [
            _choice(rng, ("BACKSIDE CLEAN", "BACKSIDE CLEAN FINAL"), cfg.synonym_style),
            "DEPOSIT BACKSIDE PROTECTION",
            "BACKSIDE ANNEAL",
        ]
        if _flag(rng, cfg.extra_measurements, 0.5):
            steps.append("FRONTSIDE CLEAN FINAL")
        return steps
    return [
        "BACKSIDE CLEAN",
        "BACKSIDE GRIND",
        *_optional_measure(rng, ("MEASURE THICKNESS", "MEASURE WAFER THICKNESS"), cfg),
        "BACKSIDE ETCH CLEAN",
        "BACKSIDE RINSE",
        "BACKSIDE DRY",
        "BACKSIDE METALLIZATION PREP",
        "DEPOSIT BACKSIDE METAL",
        "BACKSIDE ANNEAL",
        *_optional_measure(rng, "MEASURE BACKSIDE CONTACT", cfg),
    ]


def _gen_final_inspection(
    rng: random.Random,
    family: str,
    cfg: AugmentationOptions,
) -> list[str]:
    steps = ["FINAL CLEAN"]
    steps += _optional_measure(rng, "FINAL THICKNESS MEASURE", cfg, 0.8)
    steps += _optional_measure(rng, "FINAL GEOMETRY CHECK", cfg, 0.8)
    if family == "ic":
        steps += _optional_measure(rng, "FINAL OXIDE CHECK", cfg, 0.55)
    steps += _optional_measure(rng, "FINAL CD INSPECTION", cfg, 0.5)
    steps += _optional_measure(rng, "FINAL PARTICLE INSPECTION", cfg, 0.8)
    if family == "ic" and _flag(rng, cfg.extra_measurements, 0.5):
        steps.append("FINAL ELECTRICAL TEST PREP")
    return steps


def _gen_test_suite(
    rng: random.Random,
    family: str,
    cfg: AugmentationOptions,
) -> list[str]:
    param = _choice(rng, ("PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST"), cfg.synonym_style)
    family_test = {
        "mosfet": "THRESHOLD VOLTAGE TEST",
        "igbt": "BREAKDOWN VOLTAGE TEST",
        "ic": _choice(rng, ("THRESHOLD VOLTAGE TEST", "PARAMETRIC TEST"), cfg.synonym_style),
    }[family]
    steps = [param, "LEAKAGE TEST"]
    if family_test != param:
        steps.append(family_test)
    steps.append("SWITCHING TEST")
    steps += ["WAFER SORT TEST", "YIELD ANALYSIS"]
    return steps


def _gen_suffix(rng: random.Random, family: str, cfg: AugmentationOptions) -> list[str]:
    steps = [_choice(rng, ("LOT RELEASE", "FINAL LOT RELEASE"), cfg.synonym_style)]
    if family == "ic" and _flag(rng, cfg.extra_measurements, 0.5):
        steps.append("PACKAGE PREPARATION")
    steps.append("SHIP LOT")
    return steps


def generate_augmented_sequence(
    family: str,
    rng: random.Random,
    options: AugmentationOptions | None = None,
) -> list[str]:
    """Generate one sequence using all documented variation axes."""
    family = normalize_family(family)
    cfg = options or AugmentationOptions()

    steps = []
    steps += _gen_prefix(rng)
    steps += _gen_initial_measurements(rng, family, cfg)
    steps += _gen_pre_process_clean(rng, family, cfg)
    steps += _gen_family_prep(rng, family, cfg)
    steps += _intermediate_clean(rng, cfg)
    steps += _gen_first_oxidation(rng, family, cfg)
    core_steps, via_level = _gen_core_cycles(rng, family, cfg)
    steps += core_steps
    steps += _intermediate_clean(rng, cfg)
    steps += _gen_ild_block(rng, cfg)
    steps += _gen_via_block(rng, cfg, via_level, family)
    metal_level = via_level + 1
    steps += _gen_metal_block(rng, cfg, metal_level, family, layer=1)
    if _flag(rng, cfg.second_metal_layer, 0.35):
        steps += _gen_metal_block(rng, cfg, metal_level + 1, family, layer=2)
    steps += _gen_passivation_block(rng, cfg)
    steps += _gen_backside_block(rng, family, cfg)
    steps += _gen_final_inspection(rng, family, cfg)
    steps += _gen_test_suite(rng, family, cfg)
    steps += _gen_suffix(rng, family, cfg)
    return steps


def generate_augmented_dataset(
    family: str,
    count: int,
    *,
    seed: int = 42,
    options: AugmentationOptions | None = None,
    validate: bool = True,
) -> list[list[str]]:
    """Generate unique augmented sequences, rejecting anything the organizer validator flags."""
    rng = random.Random(seed)
    sequences: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    max_attempts = max(count * 50, 100)

    while len(sequences) < count and attempts < max_attempts:
        attempts += 1
        seq = generate_augmented_sequence(family, rng, options)
        key = tuple(seq)
        if key in seen:
            continue
        if validate:
            violations = validate_sequence(seq)
            if violations:
                first = violations[0]
                print(
                    f"[WARN] augmented generator produced invalid sequence "
                    f"({first.rule}: {first.description}); skipping",
                    file=sys.stderr,
                )
                continue
        seen.add(key)
        sequences.append(seq)

    if len(sequences) < count:
        raise RuntimeError(f"Only generated {len(sequences)}/{count} unique sequences")
    return sequences


def write_augmented_csv(path: str | Path, sequences: list[list[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SEQUENCE_ID", "STEP"])
        for i, seq in enumerate(sequences, start=1):
            seq_id = f"seq_{i:04d}"
            for step in seq:
                writer.writerow([seq_id, step])


def family_output_name(family: str) -> str:
    return FAMILY_FILE_NAMES[normalize_family(family)]
