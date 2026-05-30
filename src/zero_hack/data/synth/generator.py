import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from zero_hack.data.synth import vocab
from zero_hack.eval import validator as _validator

_mod = _validator._generator_module()
DEPOSITION_STEPS: frozenset[str] = frozenset(_mod.DEPOSITION_STEPS)
CLEAN_STEPS: frozenset[str] = frozenset(_mod.CLEAN_STEPS)

KNOWN_FAMILIES = ("mosfet", "igbt", "ic")
_SAFE_CLEANS = [
    "HF DIP",
    "WAFER SURFACE CLEAN",
    "CLEAN AFTER ETCH",
    "RCA CLEAN 1",
    "WET CLEAN RCA1",
]

_CYCLE_KIND_REP = {
    "oxide": "OXIDE ETCH",
    "poly": "DEPOSIT POLYSILICON",
    "field": "DEPOSIT FIELD OXIDE",
    "window": "ETCH SILICON OR OXIDE WINDOW",
}


class _Ctx:
    """Mutable generation state: the emitted steps, the RNG, and coverage counts."""

    def __init__(self, rng: random.Random, counts: Counter) -> None:
        self.rng = rng
        self.counts = counts
        self.steps: list[str] = []
        self._level = 0

    def _raw(self, token: str) -> None:
        self.steps.append(token)
        self.counts[token] += 1

    def emit(self, token: str) -> None:
        """Append ``token``, injecting a clean first if a deposition needs one."""
        if token in DEPOSITION_STEPS and not self._has_clean(12):
            self._raw(self.pick(_SAFE_CLEANS))
        self._raw(token)

    def _has_clean(self, window: int) -> bool:
        return any(s in CLEAN_STEPS for s in self.steps[-window:])

    def pick(self, choices: list[str]) -> str:
        """Return the least-used choice (ties broken randomly)."""
        best = min(self.counts[c] for c in choices)
        pool = [c for c in choices if self.counts[c] == best]
        return self.rng.choice(pool)

    def pick_emit(self, choices: list[str]) -> None:
        self.emit(self.pick(choices))

    def want(self, token: str, p: float) -> bool:
        """True if ``token`` is still uncovered (forced) or with probability ``p``."""
        return self.counts[token] == 0 or self.rng.random() < p

    def maybe(self, token: str, p: float) -> None:
        if self.want(token, p):
            self.emit(token)

    def maybe_pick(self, choices: list[str], p: float) -> None:
        if any(self.counts[c] == 0 for c in choices) or self.rng.random() < p:
            self.pick_emit(choices)

    def pick_branch(self, rep: dict[str, str]) -> str:
        """Pick the branch key whose representative token is least-used."""
        best = min(self.counts[t] for t in rep.values())
        pool = [key for key, token in rep.items() if self.counts[token] == best]
        return self.rng.choice(pool)

    def next_level(self) -> int:
        if self._level < vocab.MAX_LITHO_LEVEL:
            self._level += 1
        return self._level

    @property
    def level(self) -> int:
        return self._level


def _pre_dep_clean(ctx: _Ctx) -> None:
    ctx.pick_emit(["HF DIP", "WAFER SURFACE CLEAN", "CLEAN AFTER ETCH"])


def _strip(ctx: _Ctx) -> None:
    ctx.pick_emit(["STRIP PHOTORESIST", "STRIP RESIST", vocab.strip_level_token(ctx.level)])


def _litho(ctx: _Ctx, inspection) -> int:
    level = ctx.next_level()
    ctx.emit("SPIN COAT PHOTORESIST")
    ctx.emit("SOFT BAKE")
    ctx.emit(vocab.align_token(level))
    ctx.emit(vocab.expose_token(level))
    ctx.maybe("POST EXPOSE BAKE", 0.4)
    ctx.emit("DEVELOP PHOTORESIST")
    ctx.emit(inspection(level))
    ctx.maybe("HARD BAKE", 0.4)
    return level


def _anneal_tail(ctx: _Ctx) -> None:
    ctx.maybe("PRE ANNEAL CHECK", 0.3)
    ctx.pick_emit(["RAPID THERMAL ANNEAL", "DRIVE IN DIFFUSION", "LIGHT ANNEAL"])
    ctx.maybe_pick(
        [
            "MEASURE SHEET RESISTANCE",
            "MEASURE JUNCTION DEPTH",
            "MEASURE JUNCTION PROFILE",
            "MEASURE DEVICE PARAMETER",
        ],
        0.6,
    )


def _prefix(ctx: _Ctx) -> None:
    ctx.emit("RECEIVE WAFER LOT")
    ctx.emit("LOT IDENTIFICATION")
    ctx.pick_emit(["INITIAL WAFER INSPECTION", "PRE CLEAN INSPECTION"])


def _initial_measurements(ctx: _Ctx) -> None:
    ctx.maybe_pick(
        ["MEASURE THICKNESS", "MEASURE INITIAL THICKNESS", "MEASURE INITIAL GEOMETRY"], 0.7
    )
    ctx.maybe_pick(["MEASURE SURFACE PARTICLES", "MEASURE SURFACE DEFECTS"], 0.7)


def _pre_clean(ctx: _Ctx) -> None:
    ctx.pick_emit(["PRE CLEAN WAFER", "WAFER CLEAN PRE PROCESS"])
    ctx.maybe("BACKSIDE CLEAN", 0.5)
    ctx.maybe("FRONTSIDE CLEAN", 0.5)
    ctx.pick_emit(["RCA CLEAN 1", "WET CLEAN RCA1"])
    ctx.pick_emit(["RCA CLEAN 2", "WET CLEAN RCA2"])
    ctx.emit("HF DIP")
    ctx.maybe_pick(["DRY WAFER", "DRY WAFER BACKSIDE"], 0.5)


def _substrate_prep(ctx: _Ctx) -> str:
    variant = ctx.pick_branch(
        {
            "mosfet": "SUBSTRATE CHECK",
            "igbt": "EPITAXIAL WAFER CHECK",
            "ic": "GRINDING WAFER BACKSIDE",
        }
    )
    if variant == "mosfet":
        for token in (
            "SUBSTRATE CHECK",
            "EPITAXY PREP",
            "EPITAXIAL DEPOSITION",
            "MEASURE EPITAXY THICKNESS",
            "MEASURE RESISTIVITY",
            "EPITAXY ANNEAL",
            "WAFER SURFACE CLEAN",
        ):
            ctx.emit(token)
    elif variant == "igbt":
        ctx.emit("EPITAXIAL WAFER CHECK")
        ctx.emit("MEASURE EPITAXY THICKNESS")
        ctx.emit("MEASURE RESISTIVITY")
        ctx.maybe("EPITAXIAL REWORK CHECK", 0.4)
        ctx.emit("EPITAXIAL LAYER PREP")
        ctx.emit("WAFER SURFACE CLEAN")
    else:
        ctx.emit("WAFER CLEAN PRE-GRIND")
        ctx.emit("GRINDING WAFER BACKSIDE")
        ctx.pick_emit(["MEASURE GEOMETRY", "MEASURE INITIAL GEOMETRY"])
        ctx.emit("ETCH WET BACKSIDE")
        ctx.emit("RINSE WET WAFER_EDGE")
        ctx.emit("DRY WAFER BACKSIDE")
        ctx.emit("BACKSIDE CLEAN")
        ctx.emit("MEASURE BACKSIDE ROUGHNESS")
        ctx.emit("SURFACE PREP FOR DEPOSITION")
    return variant


def _first_oxidation(ctx: _Ctx) -> None:
    ctx.emit("THERMAL OXIDATION")
    ctx.maybe_pick(["GATE OXIDE PREP", "GATE OXIDE GROWTH"], 0.4)
    if ctx.want("DEPOSIT PAD OXIDE", 0.5) or ctx.counts["DEPOSIT GATE OXIDE OR DIELECTRIC"] == 0:
        ctx.maybe("SURFACE PREP FOR DEPOSITION", 0.7)
        ctx.pick_emit(["DEPOSIT PAD OXIDE", "DEPOSIT GATE OXIDE OR DIELECTRIC"])
        ctx.emit("ANNEAL OXIDE")
        ctx.maybe("MEASURE OXIDE QUALITY", 0.4)
    else:
        ctx.maybe("ANNEAL OXIDE", 0.5)
    ctx.maybe_pick(
        ["MEASURE OXIDE THICKNESS", "MEASURE FILM THICKNESS", "MEASURE GATE OXIDE THICKNESS"], 0.7
    )
    ctx.maybe("OXIDE STRIP", 0.25)


def _spacer(ctx: _Ctx) -> None:
    """MOSFET LDD spacer sub-block. Compact so IMPLANT LDD stays within the
    15-step opener window of the parent poly cycle's DEVELOP."""
    ctx.emit("DEPOSIT SPACER DIELECTRIC")
    ctx.emit("ANISOTROPIC ETCH SPACER")  # blanket etch — rule-exempt, no mask needed
    ctx.emit("IMPLANT LDD")
    ctx.maybe("MEASURE SPACER WIDTH", 0.5)
    ctx.pick_emit(["LIGHT ANNEAL", "RAPID THERMAL ANNEAL"])


def _level_inspection(ctx: _Ctx):
    return lambda lv: ctx.pick(
        [vocab.inspect_level_token(lv), vocab.pattern_inspection_level_token(lv)]
    )


def _cycle(ctx: _Ctx, kind: str) -> None:
    if kind == "oxide":
        ctx.emit("THERMAL OXIDATION")
        ctx.maybe("ANNEAL OXIDE", 0.3)
        ctx.maybe_pick(["MEASURE OXIDE THICKNESS", "MEASURE FILM THICKNESS"], 0.4)
        _litho(ctx, _level_inspection(ctx))
        ctx.pick_emit(["OXIDE ETCH", "OXIDE ETCH DRY"])
        _strip(ctx)
        ctx.pick_emit(["CLEAN AFTER ETCH", "CLEAN AFTER OXIDE ETCH"])
        ctx.maybe_pick(["MEASURE OPENING CD", vocab.cd_level_token(ctx.level)], 0.6)
        ctx.pick_emit(["IMPLANT WELL", "IMPLANT P BODY", "IMPLANT N-TYPE"])
        _anneal_tail(ctx)
    elif kind == "poly":
        _pre_dep_clean(ctx)
        ctx.emit("DEPOSIT POLYSILICON")
        ctx.pick_emit(["POLYSILICON ANNEAL", "ANNEAL POLYSILICON"])
        ctx.maybe("MEASURE POLY THICKNESS", 0.5)
        _litho(ctx, lambda lv: "POLY PATTERN INSPECTION")
        ctx.pick_emit(["POLYSILICON ETCH", "POLYSILICON ETCH DRY"])
        _strip(ctx)
        ctx.emit("CLEAN AFTER POLY ETCH")
        ctx.maybe("MEASURE GATE CD", 0.5)
        ctx.pick_emit(
            [
                "IMPLANT SOURCE DRAIN",
                "IMPLANT CHANNEL STOP",
                "IMPLANT N-TYPE",
                "IMPLANT SOURCE REGION",
            ]
        )
        if ctx.want("DEPOSIT SPACER DIELECTRIC", 0.45):
            _spacer(ctx)
        else:
            _anneal_tail(ctx)
    elif kind == "field":
        _pre_dep_clean(ctx)
        ctx.emit("DEPOSIT FIELD OXIDE")
        ctx.pick_emit(["DENSIFY OXIDE", "DENSIFY DIELECTRIC"])
        ctx.maybe("MEASURE FILM THICKNESS", 0.4)
        _litho(ctx, lambda lv: "FIELD PATTERN INSPECTION")
        ctx.emit("FIELD OXIDE ETCH")
        _strip(ctx)
        ctx.emit("CLEAN AFTER FIELD ETCH")
        ctx.maybe("MEASURE LINE WIDTH", 0.5)
        ctx.pick_emit(["IMPLANT SOURCE DRAIN", "IMPLANT DRAIN / CATHODE REGION"])
        _anneal_tail(ctx)
    elif kind == "window":
        ctx.emit("THERMAL OXIDATION")
        _litho(ctx, lambda lv: "P BODY WINDOW INSPECTION")
        ctx.emit("ETCH SILICON OR OXIDE WINDOW")
        _strip(ctx)
        ctx.emit("CLEAN AFTER WINDOW ETCH")
        ctx.maybe("MEASURE WINDOW CD", 0.5)
        ctx.pick_emit(["IMPLANT N BUFFER", "IMPLANT P BODY", "IMPLANT SOURCE REGION"])
        ctx.maybe("MEASURE DEVICE PARAMETER", 0.3)
        _anneal_tail(ctx)
    else:  # pragma: no cover - guarded by _CYCLE_KIND_REP
        raise ValueError(f"unknown cycle kind {kind!r}")


def _ild_block(ctx: _Ctx) -> None:
    _pre_dep_clean(ctx)
    ctx.pick_emit(["DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC"])
    ctx.pick_emit(["DENSIFY DIELECTRIC", "DENSIFY OXIDE"])
    ctx.maybe_pick(["MEASURE FILM THICKNESS", "MEASURE DIELECTRIC THICKNESS"], 0.6)
    ctx.pick_emit(["CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC"])
    ctx.maybe_pick(["MEASURE PLANARITY", "MEASURE SURFACE PLANARITY"], 0.7)


def _via_block(ctx: _Ctx) -> None:
    _litho(ctx, lambda lv: ctx.pick(["VIA INSPECTION", "VIA OPENING INSPECTION"]))
    ctx.pick_emit(["VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA"])
    _strip(ctx)
    ctx.emit("CLEAN AFTER VIA ETCH")
    ctx.maybe("MEASURE VIA CD", 0.5)
    ctx.emit("DEPOSIT BARRIER METAL")
    ctx.pick_emit(["DEPOSIT METAL SEED", "DEPOSIT TUNGSTEN SEED"])
    ctx.pick_emit(["FILL VIA METAL", "FILL VIA TUNGSTEN"])
    ctx.pick_emit(["CMP METAL", "CMP VIA FILL"])
    ctx.maybe_pick(["MEASURE CONTACT RESISTANCE", "MEASURE VIA RESISTANCE"], 0.6)


def _metal_block(ctx: _Ctx) -> None:
    _pre_dep_clean(ctx)
    ctx.pick_emit(["DEPOSIT METAL 1", "DEPOSIT TOP METAL"])
    ctx.pick_emit(["ANNEAL METAL 1", "ANNEAL METAL"])
    ctx.maybe("MEASURE METAL THICKNESS", 0.5)
    _litho(ctx, lambda lv: "METAL PATTERN INSPECTION")
    ctx.pick_emit(["METAL ETCH", "METAL ETCH DRY"])
    _strip(ctx)
    ctx.emit("CLEAN AFTER METAL ETCH")
    ctx.maybe("MEASURE LINE WIDTH", 0.5)


def _passivation(ctx: _Ctx) -> None:
    _pre_dep_clean(ctx)
    ctx.pick_emit(["DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER"])
    ctx.emit("CURE PASSIVATION")
    ctx.maybe_pick(["MEASURE PASSIVATION THICKNESS", "MEASURE PASSIVATION QUALITY"], 0.7)
    ctx.pick_emit(["OPEN PAD WINDOW", "OPEN BOND PAD WINDOW"])
    ctx.pick_emit(["PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO"])
    ctx.pick_emit(["DEVELOP PAD WINDOW", "DEVELOP PHOTORESIST"])
    ctx.pick_emit(["PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH"])
    _strip(ctx)
    ctx.emit("CLEAN PAD OPENING")
    ctx.maybe("MEASURE PAD OPENING", 0.6)


def _backside(ctx: _Ctx) -> None:
    ctx.maybe("BACKSIDE THINNING CHECK", 0.4)
    ctx.pick_emit(["BACKSIDE CLEAN", "BACKSIDE CLEAN FINAL"])
    ctx.maybe("DEPOSIT BACKSIDE PROTECTION", 0.4)
    ctx.emit("BACKSIDE GRIND")
    ctx.pick_emit(["MEASURE THICKNESS", "MEASURE WAFER THICKNESS"])
    ctx.emit("BACKSIDE ETCH CLEAN")
    ctx.emit("BACKSIDE RINSE")
    ctx.emit("BACKSIDE DRY")
    ctx.emit("BACKSIDE METALLIZATION PREP")
    ctx.pick_emit(["BACKSIDE CLEAN", "BACKSIDE CLEAN FINAL"])
    ctx.emit("DEPOSIT BACKSIDE METAL")
    ctx.emit("BACKSIDE ANNEAL")
    ctx.emit("MEASURE BACKSIDE CONTACT")


def _final_inspection(ctx: _Ctx) -> None:
    ctx.emit("FINAL CLEAN")
    for token in (
        "FINAL THICKNESS MEASURE",
        "FINAL GEOMETRY CHECK",
        "FINAL OXIDE CHECK",
        "FINAL CD INSPECTION",
        "FINAL PARTICLE INSPECTION",
        "FRONTSIDE CLEAN FINAL",
        "FINAL ELECTRICAL TEST PREP",
    ):
        ctx.maybe(token, 0.6)


def _test_suite(ctx: _Ctx) -> None:
    ctx.pick_emit(["PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST"])
    ctx.emit("LEAKAGE TEST")
    ctx.pick_emit(["THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST"])
    ctx.emit("SWITCHING TEST")
    ctx.emit("WAFER SORT TEST")
    ctx.emit("YIELD ANALYSIS")


def _suffix(ctx: _Ctx) -> None:
    ctx.pick_emit(["LOT RELEASE", "FINAL LOT RELEASE"])
    ctx.maybe("PACKAGE PREPARATION", 0.4)
    ctx.emit("SHIP LOT")


@dataclass(frozen=True)
class SynthSequence:
    family_label: str
    sequence_id: str
    steps: tuple[str, ...]


def generate_one(rng: random.Random, counts: Counter | None = None) -> list[str]:
    """Assemble one valid, coherent sequence. ``counts`` (shared across a run)
    drives coverage; pass ``None`` for a standalone, self-contained sequence."""
    ctx = _Ctx(rng, Counter() if counts is None else counts)
    _prefix(ctx)
    _initial_measurements(ctx)
    _pre_clean(ctx)
    _substrate_prep(ctx)
    _first_oxidation(ctx)

    n_front = rng.choice([3, 4])
    for _ in range(n_front):
        _cycle(ctx, ctx.pick_branch(_CYCLE_KIND_REP))

    _ild_block(ctx)
    _via_block(ctx)
    _metal_block(ctx)
    if n_front == 3 and rng.random() < 0.4:  # second metal layer fits within level<=6
        _metal_block(ctx)

    _passivation(ctx)
    _backside(ctx)
    _final_inspection(ctx)
    _test_suite(ctx)
    _suffix(ctx)
    return ctx.steps


def sample_family_label(rng: random.Random, synthetic_n: int = 12, unk_prob: float = 0.25) -> str:
    if rng.random() < unk_prob:
        return "UNK"
    if rng.random() < 0.5:
        return rng.choice(KNOWN_FAMILIES)
    return f"synthetic_{rng.randrange(synthetic_n) + 1:02d}"


def generate_dataset(
    count: int,
    *,
    seed: int = 42,
    synthetic_n: int = 12,
    unk_prob: float = 0.25,
) -> list[SynthSequence]:
    rng = random.Random(seed)
    counts: Counter = Counter()
    out: list[SynthSequence] = []
    for i in range(count):
        steps = generate_one(rng, counts)
        violations = _validator.validate_sequence(steps)
        if violations:  # backstop: the planner should make this unreachable
            raise RuntimeError(f"generated invalid sequence #{i}: {violations[:2]}")
        label = sample_family_label(rng, synthetic_n, unk_prob)
        out.append(SynthSequence(label, f"synth_{i + 1:05d}", tuple(steps)))
    return out


def write_dataset_csv(path: str | Path, dataset: list[SynthSequence]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FAMILY", "SEQUENCE_ID", "STEP"])
        for record in dataset:
            for step in record.steps:
                writer.writerow([record.family_label, record.sequence_id, step])
