"""Generate genuinely NOVEL, rule-valid, family-less process flows (eval-only).

Contrast with ``scripts/generate_pseudo_families.py``: that script recombines the
MOSFET / IGBT / IC grammars at *whole-block* granularity (prep=igbt, cycles=ic, ...),
so every emitted block is a verbatim copy of one official family's block generator.
This script does NOT call any family block generator. Instead it composes each
sequence from atomic, *role-typed* process operations drawn from a single
cross-family pool, with every micro-choice (which clean, which etch variant, which
implant species, which via metallurgy, which test) made independently per slot.

The only structure imposed is what the organizer actually enforces:
  * the 10 process-logic rules (``validate_sequence`` must return zero violations),
  * the closed training vocabulary (no ``<UNK_STEP>``; level-numbered tokens kept
    inside seen ranges: ALIGN/EXPOSE LITHO LEVEL <= 6, no MEASURE CD LEVEL > 2),

plus the handful of *global* orderings the rules require (pad window after
passivation+cure, tests after cure, backside metal after cure, ship after sort,
litho levels monotonic). Everything else -- family prep, which implant goes with
which etch, whether epitaxy and backside-grind coexist, mixing threshold AND
breakdown tests in one flow -- is sampled freely. The result is a flow that
satisfies process logic yet matches NO official family: a novel grammar, not a
per-block remix. Because no held-out family's block generator is ever invoked,
these sets cannot leak a held-out family's block grammar into training (they are
eval-only regardless and must never enter training).

A built-in novelty audit quantifies divergence from all three real families via
step n-gram overlap, and asserts zero exact-sequence collisions.

Output (eval-only):
    data/eval/novel_families/<profile>/raw.csv      # FAMILY,SEQUENCE_ID,STEP
    data/eval/novel_families/<profile>/meta.json     # config + accept + novelty stats

Usage:
    uv run python scripts/generate_novel_families.py --count 5000
    uv run python scripts/generate_novel_families.py --count 200 --profiles novel_mixed
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from zero_hack import PROJECT_ROOT
from zero_hack.data.augmented_generator import (
    AugmentationOptions,
    _litho,
    _strip,
)
from zero_hack.eval.validator import _generator_module, validate_sequence

DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "eval" / "novel_families"


def known_vocabulary() -> set[str]:
    """Step inventory of the official families (what the model has in its vocab)."""
    from zero_hack.data import load_sequence_records
    from zero_hack.data.datasets import FAMILY_FILE_NAMES
    from zero_hack.models.common import DEFAULT_SPLITS_DIR

    steps: set[str] = set()
    for family, fname in FAMILY_FILE_NAMES.items():
        stem = fname.removesuffix(".csv")
        for split in ("train", "valid", "test"):
            path = Path(DEFAULT_SPLITS_DIR) / f"{stem}_{split}.csv"
            if path.exists():
                for rec in load_sequence_records(path, family=family):
                    steps.update(rec.steps)
    return steps


# Maximum litho level present in the training vocabulary (ALIGN/EXPOSE LITHO LEVEL
# n only exist for n <= 6). Every leveled litho block consumes one level, so the
# total of (process cycles + via block + metal blocks) must not exceed this.
MAX_LITHO_LEVEL = 6

_REAL_FAMILIES = ("mosfet", "igbt", "ic")


# ----------------------------------------------------------------------------- #
# Role-typed option pools. Sourced from the validator's own rule sets where they
# exist (so "what counts as a clean / deposition / etch" can never drift from the
# checker), plus literal pools for the choices the rules don't constrain
# (inspection labels, synonyms). All pools are intersected with the live training
# vocabulary at runtime, so nothing here can emit an out-of-vocab step.
# ----------------------------------------------------------------------------- #
_M = _generator_module()

# Cleans that satisfy RULE_DEP_NO_CLEAN but are not themselves depositions (so a
# clean never needs its own preceding clean). Post-etch cleans are reserved for
# their etch slots; this pool is the "general surface prep" cleans.
_GENERAL_CLEANS = (
    frozenset(_M.CLEAN_STEPS)
    - frozenset(_M.DEPOSITION_STEPS)
    - {s for s in _M.CLEAN_STEPS if s.startswith("CLEAN AFTER") or s.startswith("BACKSIDE")}
)

# Cycle-level depositions a litho-etch-implant cycle can start from. Excludes the
# fixed-role depositions (barrier/seed/metal/passivation/backside) handled inline.
_CYCLE_DEPOSITIONS = (
    "THERMAL OXIDATION",
    "DEPOSIT POLYSILICON",
    "DEPOSIT FIELD OXIDE",
    "DEPOSIT GATE OXIDE OR DIELECTRIC",
    "DEPOSIT SPACER DIELECTRIC",
    "DEPOSIT PAD OXIDE",
)

# Patterned etches usable inside a generic cycle (metal/via/passivation etches are
# handled in their own blocks; backside etch needs no mask).
_CYCLE_ETCHES = (
    "OXIDE ETCH",
    "OXIDE ETCH DRY",
    "POLYSILICON ETCH",
    "POLYSILICON ETCH DRY",
    "ETCH SILICON OR OXIDE WINDOW",
    "FIELD OXIDE ETCH",
)
_POST_ETCH_CLEANS = (
    "CLEAN AFTER ETCH",
    "CLEAN AFTER OXIDE ETCH",
    "CLEAN AFTER POLY ETCH",
    "CLEAN AFTER WINDOW ETCH",
    "CLEAN AFTER FIELD ETCH",
)
_IMPLANTS = tuple(sorted(_M.IMPLANT_STEPS))
_POST_IMPLANT_ANNEALS = ("RAPID THERMAL ANNEAL", "DRIVE IN DIFFUSION", "LIGHT ANNEAL")

# In-vocab inspection labels for a patterned litho block. INSPECT PATTERN LEVEL n
# only exists for n == 1, so higher-level cycles must use a *named* inspection.
_CYCLE_INSPECTIONS = (
    "INSPECT PATTERN LEVEL 1",
    "PATTERN INSPECTION LEVEL 1",
    "PATTERN INSPECTION LEVEL 2",
    "POLY PATTERN INSPECTION",
    "P BODY WINDOW INSPECTION",
    "FIELD PATTERN INSPECTION",
)
_VIA_INSPECTIONS = ("VIA INSPECTION", "VIA OPENING INSPECTION")
_VIA_ETCHES = ("VIA ETCH", "VIA ETCH THROUGH DIELECTRIC", "DIELECTRIC ETCH VIA")
_VIA_SEEDS = ("DEPOSIT METAL SEED", "DEPOSIT TUNGSTEN SEED")
_VIA_FILLS = ("FILL VIA METAL", "FILL VIA TUNGSTEN")
_VIA_CMPS = ("CMP VIA FILL", "CMP METAL")
_ILD_DEPS = ("DEPOSIT INTERLAYER DIELECTRIC", "DEPOSIT INTERLEVEL DIELECTRIC")
_ILD_DENSIFY = ("DENSIFY DIELECTRIC", "DENSIFY OXIDE")
_ILD_CMPS = ("CMP DIELECTRIC", "CMP INTERLAYER DIELECTRIC")
_METAL_ANNEALS = ("ANNEAL METAL 1", "ANNEAL METAL")
_METAL_ETCHES = ("METAL ETCH", "METAL ETCH DRY")
_PASS_DEPS = ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER")
_PAD_OPENS = ("OPEN PAD WINDOW", "OPEN BOND PAD WINDOW")
_PAD_LITHOS = ("PAD WINDOW LITHO", "OPEN PAD WINDOW LITHO")
_PAD_DEVELOPS = ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW")
_PASS_ETCHES = ("PASSIVATION ETCH PAD OPENING", "PASSIVATION ETCH")
_PARAM_TESTS = ("PARAMETRIC TEST", "ELECTRICAL PARAMETRIC TEST")
_FAMILY_TESTS = ("THRESHOLD VOLTAGE TEST", "BREAKDOWN VOLTAGE TEST")


@dataclass(frozen=True)
class NovelProfile:
    """Probability knobs for one novel-flow flavour (not a family trait table)."""

    name: str
    min_cycles: int = 2
    max_cycles: int = 4
    p_epitaxy: float = 0.4
    p_backside_grind: float = 0.4
    p_cycle_deposition: float = 0.7
    p_implant: float = 0.75
    p_second_metal: float = 0.35
    p_measure: float = 0.5
    p_threshold_test: float = 0.6
    p_breakdown_test: float = 0.6
    p_via_cmp: float = 0.8


PROFILES: dict[str, NovelProfile] = {
    # Balanced novel flow: moderate optional-step density, frequent cross-family
    # implant/test mixing.
    "novel_mixed": NovelProfile(name="novel_mixed"),
    # Lean flows: few optional steps, short cycle count -- stresses the minimal
    # rule-valid skeleton.
    "novel_sparse": NovelProfile(
        name="novel_sparse",
        min_cycles=2,
        max_cycles=3,
        p_epitaxy=0.15,
        p_backside_grind=0.15,
        p_cycle_deposition=0.5,
        p_implant=0.5,
        p_second_metal=0.1,
        p_measure=0.15,
        p_threshold_test=0.4,
        p_breakdown_test=0.4,
    ),
    # Dense flows: many optional measures, both family tests, long cycles.
    "novel_dense": NovelProfile(
        name="novel_dense",
        min_cycles=3,
        max_cycles=4,
        p_epitaxy=0.6,
        p_backside_grind=0.6,
        p_cycle_deposition=0.9,
        p_implant=0.9,
        p_second_metal=0.6,
        p_measure=0.85,
        p_threshold_test=0.85,
        p_breakdown_test=0.85,
    ),
}


class _Pools:
    """Option pools intersected with the live training vocabulary."""

    def __init__(self, vocab: set[str]) -> None:
        def keep(options: tuple[str, ...]) -> tuple[str, ...]:
            kept = tuple(o for o in options if o in vocab)
            if not kept:
                raise ValueError(f"no in-vocab option among {options}")
            return kept

        self.cleans = keep(tuple(sorted(_GENERAL_CLEANS)))
        self.cycle_deps = keep(_CYCLE_DEPOSITIONS)
        self.cycle_etches = keep(_CYCLE_ETCHES)
        self.post_etch_cleans = keep(_POST_ETCH_CLEANS)
        self.implants = keep(_IMPLANTS)
        self.post_implant_anneals = keep(_POST_IMPLANT_ANNEALS)
        self.cycle_inspections = keep(_CYCLE_INSPECTIONS)
        self.via_inspections = keep(_VIA_INSPECTIONS)
        self.via_etches = keep(_VIA_ETCHES)
        self.via_seeds = keep(_VIA_SEEDS)
        self.via_fills = keep(_VIA_FILLS)
        self.via_cmps = keep(_VIA_CMPS)
        self.ild_deps = keep(_ILD_DEPS)
        self.ild_densify = keep(_ILD_DENSIFY)
        self.ild_cmps = keep(_ILD_CMPS)
        self.metal_anneals = keep(_METAL_ANNEALS)
        self.metal_etches = keep(_METAL_ETCHES)
        self.pass_deps = keep(_PASS_DEPS)
        self.pad_opens = keep(_PAD_OPENS)
        self.pad_lithos = keep(_PAD_LITHOS)
        self.pad_develops = keep(_PAD_DEVELOPS)
        self.pass_etches = keep(_PASS_ETCHES)
        self.param_tests = keep(_PARAM_TESTS)
        self.family_tests = keep(_FAMILY_TESTS)
        # Measures are safe anywhere (they never trigger a rule); draw from vocab.
        self.measures = tuple(sorted(m for m in vocab if m.startswith("MEASURE")))


def _clean(rng: random.Random, pools: _Pools) -> str:
    return rng.choice(pools.cleans)


def _maybe_measure(rng: random.Random, pools: _Pools, prob: float) -> list[str]:
    if pools.measures and rng.random() < prob:
        return [rng.choice(pools.measures)]
    return []


def generate_novel_sequence(
    profile: NovelProfile,
    rng: random.Random,
    pools: _Pools,
) -> list[str]:
    """Assemble one novel, rule-valid, family-less flow.

    Rule preconditions are satisfied *by construction*: a clean is emitted
    immediately before every deposition (RULE_DEP_NO_CLEAN), a full litho block
    before every patterned etch (RULE_ETCH_NO_MASK / RULE_METAL_ETCH_NO_LITHO),
    an etch+develop before every implant (RULE_IMPLANT_NO_MASK), a deposition or
    fill before every CMP (RULE_CMP_NO_DEP); litho levels increase monotonically
    (RULE_LITHO_LEVEL_SKIP); and the passivation -> tests -> backside-metal ->
    sort -> ship global orderings follow from block order.
    """
    # Free-choice synonym style for the shared litho primitive.
    cfg = AugmentationOptions(synonym_style="random")
    steps: list[str] = []
    level = 1  # next litho level to consume

    n_cycles = rng.randint(profile.min_cycles, profile.max_cycles)
    # Leveled litho budget: cycles + via(1) + metal layers must stay <= MAX level.
    max_metal = max(1, min(2, MAX_LITHO_LEVEL - n_cycles - 1))
    n_metal = 2 if (max_metal == 2 and rng.random() < profile.p_second_metal) else 1

    # --- intake + pre-process clean ----------------------------------------- #
    steps += [
        "RECEIVE WAFER LOT",
        "LOT IDENTIFICATION",
        rng.choice(["INITIAL WAFER INSPECTION", "PRE CLEAN INSPECTION"]),
    ]
    steps += _maybe_measure(rng, pools, profile.p_measure)
    steps += ["WAFER CLEAN PRE PROCESS" if rng.random() < 0.5 else "PRE CLEAN WAFER"]
    steps += ["RCA CLEAN 1", "RCA CLEAN 2", "HF DIP"]

    # --- novel prep: epitaxy and/or backside-grind, independently (or neither) #
    if rng.random() < profile.p_epitaxy:
        steps += [_clean(rng, pools), "EPITAXIAL DEPOSITION"]
        steps += _maybe_measure(rng, pools, profile.p_measure)
        steps += ["EPITAXY ANNEAL"]
    if rng.random() < profile.p_backside_grind:
        steps += ["GRINDING WAFER BACKSIDE", "ETCH WET BACKSIDE", "BACKSIDE CLEAN"]
        steps += _maybe_measure(rng, pools, profile.p_measure)

    # --- first oxidation ----------------------------------------------------- #
    steps += [_clean(rng, pools), "THERMAL OXIDATION"]
    steps += _maybe_measure(rng, pools, profile.p_measure)

    # --- process cycles: each micro-choice independent (cross-family mixing) -- #
    for _ in range(n_cycles):
        if rng.random() < profile.p_cycle_deposition:
            steps += [_clean(rng, pools), rng.choice(pools.cycle_deps)]
            steps += _maybe_measure(rng, pools, profile.p_measure)
        steps += _litho(rng, cfg, level, rng.choice(pools.cycle_inspections))
        level += 1
        steps += [
            rng.choice(pools.cycle_etches),
            _strip(rng, cfg),
            rng.choice(pools.post_etch_cleans),
        ]
        steps += _maybe_measure(rng, pools, profile.p_measure)
        if rng.random() < profile.p_implant:
            steps += [rng.choice(pools.implants), rng.choice(pools.post_implant_anneals)]
            steps += _maybe_measure(rng, pools, profile.p_measure)

    # --- ILD block ----------------------------------------------------------- #
    steps += [_clean(rng, pools), rng.choice(pools.ild_deps), rng.choice(pools.ild_densify)]
    steps += _maybe_measure(rng, pools, profile.p_measure)
    steps += [rng.choice(pools.ild_cmps)]
    steps += _maybe_measure(rng, pools, profile.p_measure)

    # --- via block ----------------------------------------------------------- #
    steps += _litho(rng, cfg, level, rng.choice(pools.via_inspections))
    level += 1
    steps += [rng.choice(pools.via_etches), _strip(rng, cfg), "CLEAN AFTER VIA ETCH"]
    steps += _maybe_measure(rng, pools, profile.p_measure)
    steps += [_clean(rng, pools), "DEPOSIT BARRIER METAL", rng.choice(pools.via_seeds)]
    steps += [rng.choice(pools.via_fills)]
    if rng.random() < profile.p_via_cmp:
        steps += [rng.choice(pools.via_cmps)]
    steps += _maybe_measure(rng, pools, profile.p_measure)

    # --- metal block(s) ------------------------------------------------------ #
    for layer in range(n_metal):
        metal_dep = "DEPOSIT METAL 1" if layer == 0 else "DEPOSIT TOP METAL"
        steps += [_clean(rng, pools), metal_dep, rng.choice(pools.metal_anneals)]
        steps += _maybe_measure(rng, pools, profile.p_measure)
        steps += _litho(rng, cfg, level, "METAL PATTERN INSPECTION")
        level += 1
        steps += [rng.choice(pools.metal_etches), _strip(rng, cfg), "CLEAN AFTER METAL ETCH"]
        steps += _maybe_measure(rng, pools, profile.p_measure)

    # --- passivation + pad window (after deposit+cure) ----------------------- #
    steps += [_clean(rng, pools), rng.choice(pools.pass_deps), "CURE PASSIVATION"]
    steps += _maybe_measure(rng, pools, profile.p_measure)
    steps += [
        rng.choice(pools.pad_opens),
        rng.choice(pools.pad_lithos),
        rng.choice(pools.pad_develops),
        rng.choice(pools.pass_etches),
        _strip(rng, cfg),
        "CLEAN PAD OPENING",
    ]
    steps += _maybe_measure(rng, pools, profile.p_measure)

    # --- backside metallization (after cure) --------------------------------- #
    steps += [
        "BACKSIDE CLEAN",
        "BACKSIDE GRIND",
        "BACKSIDE ETCH CLEAN",
        "BACKSIDE RINSE",
        "BACKSIDE DRY",
        "BACKSIDE METALLIZATION PREP",
        "DEPOSIT BACKSIDE METAL",
        "BACKSIDE ANNEAL",
    ]
    steps += _maybe_measure(rng, pools, profile.p_measure)

    # --- final inspection ---------------------------------------------------- #
    steps += ["FINAL CLEAN"]
    steps += _maybe_measure(rng, pools, profile.p_measure)

    # --- test suite (after cure) + free family-test mixing ------------------- #
    steps += [rng.choice(pools.param_tests), "LEAKAGE TEST"]
    if rng.random() < profile.p_threshold_test:
        steps += ["THRESHOLD VOLTAGE TEST"]
    if rng.random() < profile.p_breakdown_test:
        steps += ["BREAKDOWN VOLTAGE TEST"]
    steps += ["SWITCHING TEST", "WAFER SORT TEST", "YIELD ANALYSIS"]

    # --- suffix (ship after sort) -------------------------------------------- #
    steps += [rng.choice(["LOT RELEASE", "FINAL LOT RELEASE"])]
    if rng.random() < 0.3:
        steps += ["PACKAGE PREPARATION"]
    steps += ["SHIP LOT"]
    return steps


@dataclass
class GenStats:
    requested: int
    attempts: int = 0
    invalid: int = 0
    out_of_vocab: int = 0
    duplicates: int = 0
    accepted: int = 0
    rule_counts: dict[str, int] = field(default_factory=dict)

    @property
    def raw_validity_rate(self) -> float:
        return round((self.attempts - self.invalid) / self.attempts, 4) if self.attempts else 0.0


def generate_novel_dataset(
    profile: NovelProfile,
    count: int,
    *,
    vocab: set[str],
    seed: int = 42,
    max_attempts_factor: int = 50,
) -> tuple[list[list[str]], GenStats]:
    rng = random.Random(seed)
    pools = _Pools(vocab)
    sequences: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    stats = GenStats(requested=count)
    max_attempts = max(count * max_attempts_factor, 200)

    while len(sequences) < count and stats.attempts < max_attempts:
        stats.attempts += 1
        seq = generate_novel_sequence(profile, rng, pools)
        violations = validate_sequence(seq)
        if violations:
            stats.invalid += 1
            rule = violations[0].rule
            stats.rule_counts[rule] = stats.rule_counts.get(rule, 0) + 1
            continue
        if any(step not in vocab for step in seq):
            stats.out_of_vocab += 1
            continue
        key = tuple(seq)
        if key in seen:
            stats.duplicates += 1
            continue
        seen.add(key)
        sequences.append(seq)

    stats.accepted = len(sequences)
    return sequences, stats


# ----------------------------------------------------------------------------- #
# Novelty audit: quantify how far the generated set sits from every real family.
# ----------------------------------------------------------------------------- #
def _ngrams(seq: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)} if len(seq) >= n else set()


def _family_reference(
    ref_count: int, seed: int
) -> tuple[dict[int, dict[str, set]], set[tuple[str, ...]]]:
    """Build per-family n-gram sets and the set of full family sequences.

    Uses the official family generator so the reference reflects exactly what a
    model would have been trained on.
    """
    from zero_hack.data.augmented_generator import generate_augmented_dataset

    ns = (3, 4, 5)
    fam_ngrams: dict[int, dict[str, set]] = {n: {f: set() for f in _REAL_FAMILIES} for n in ns}
    fam_seqs: set[tuple[str, ...]] = set()
    for fam in _REAL_FAMILIES:
        seqs = generate_augmented_dataset(fam, ref_count, seed=seed, validate=True)
        for seq in seqs:
            fam_seqs.add(tuple(seq))
            for n in ns:
                fam_ngrams[n][fam] |= _ngrams(seq, n)
    return fam_ngrams, fam_seqs


def novelty_report(
    novel_seqs: list[list[str]],
    fam_ngrams: dict[int, dict[str, set]],
    fam_seqs: set[tuple[str, ...]],
) -> dict:
    """For each n, mean fraction of n-grams unseen in ANY family, and the mean
    nearest-family overlap (max over families of the fraction shared)."""
    report: dict = {"exact_family_collisions": 0, "ngram": {}}
    report["exact_family_collisions"] = sum(1 for s in novel_seqs if tuple(s) in fam_seqs)

    for n, per_fam in fam_ngrams.items():
        union = set().union(*per_fam.values())
        novel_fracs: list[float] = []
        nearest_overlaps: list[float] = []
        for seq in novel_seqs:
            grams = _ngrams(seq, n)
            if not grams:
                continue
            novel_fracs.append(len(grams - union) / len(grams))
            nearest_overlaps.append(max(len(grams & per_fam[f]) / len(grams) for f in per_fam))
        report["ngram"][str(n)] = {
            "mean_globally_novel_fraction": round(sum(novel_fracs) / len(novel_fracs), 4)
            if novel_fracs
            else 0.0,
            "mean_nearest_family_overlap": round(sum(nearest_overlaps) / len(nearest_overlaps), 4)
            if nearest_overlaps
            else 0.0,
        }
    return report


def write_novel_csv(path: Path, profile_name: str, sequences: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["FAMILY", "SEQUENCE_ID", "STEP"])
        for i, seq in enumerate(sequences, start=1):
            seq_id = f"{profile_name}_{i:05d}"
            for step in seq:
                writer.writerow([profile_name, seq_id, step])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5000, help="Sequences per profile.")
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(PROFILES),
        choices=list(PROFILES),
        help="Which novel-flow profiles to generate.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument(
        "--ref-count",
        type=int,
        default=400,
        help="Family sequences per family used to build the novelty n-gram reference.",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Validate/report only; do not write CSVs."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    vocab = known_vocabulary()
    print(f"known training vocabulary: {len(vocab)} steps")
    print(f"building family n-gram reference ({args.ref_count}/family)...")
    fam_ngrams, fam_seqs = _family_reference(args.ref_count, args.seed)

    summary: dict[str, dict] = {}
    for name in args.profiles:
        profile = PROFILES[name]
        seqs, stats = generate_novel_dataset(profile, args.count, vocab=vocab, seed=args.seed)

        revalidated_invalid = sum(1 for s in seqs if validate_sequence(s))
        oov = sorted({step for s in seqs for step in s} - vocab)
        lengths = [len(s) for s in seqs]
        unique = len({tuple(s) for s in seqs})
        novelty = novelty_report(seqs, fam_ngrams, fam_seqs)

        ok = (
            revalidated_invalid == 0
            and not oov
            and unique == len(seqs)
            and novelty["exact_family_collisions"] == 0
        )
        print(f"=== {name} ===")
        print(f"  accepted             : {stats.accepted}/{stats.requested}")
        print(f"  raw validity rate    : {stats.raw_validity_rate}  (attempts={stats.attempts})")
        print(f"  re-validated invalid : {revalidated_invalid}")
        print(f"  out-of-vocab steps   : {len(oov)} {oov[:5]}")
        print(f"  unique               : {unique}/{len(seqs)}")
        print(f"  exact family collisions: {novelty['exact_family_collisions']}")
        for n, d in novelty["ngram"].items():
            print(
                f"  {n}-gram: novel={d['mean_globally_novel_fraction']:.3f} "
                f"nearest-family overlap={d['mean_nearest_family_overlap']:.3f}"
            )
        if lengths:
            mean_length = sum(lengths) // len(lengths)
            print(f"  length min/mean/max  : {min(lengths)}/{mean_length}/{max(lengths)}")
        if stats.rule_counts:
            print(f"  rejected-by-rule     : {stats.rule_counts}")
        print(f"  CORRECT              : {'YES' if ok else 'NO'}\n")

        if not args.no_write:
            prof_dir = out_root / name
            write_novel_csv(prof_dir / "raw.csv", name, seqs)
            (prof_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "profile": asdict(profile),
                        "seed": args.seed,
                        "stats": asdict(stats),
                        "raw_validity_rate": stats.raw_validity_rate,
                        "out_of_vocab_steps": oov,
                        "n_sequences": len(seqs),
                        "unique": unique,
                        "novelty": novelty,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"  wrote {prof_dir / 'raw.csv'} + meta.json\n")

        summary[name] = {
            "accepted": stats.accepted,
            "raw_validity_rate": stats.raw_validity_rate,
            "revalidated_invalid": revalidated_invalid,
            "out_of_vocab": len(oov),
            "unique": unique,
            "exact_family_collisions": novelty["exact_family_collisions"],
            "novelty_ngram": novelty["ngram"],
            "correct": ok,
        }

    print("SUMMARY:", json.dumps(summary, indent=2))
    if not all(v["correct"] for v in summary.values()):
        raise SystemExit("Some profiles failed the correctness/novelty audit (see above).")


if __name__ == "__main__":
    main()
