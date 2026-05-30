# Neurosymbolic Process-Sequence Data Generator — Design

**Date:** 2026-05-30
**Status:** Approved design, pending implementation plan
**Scope:** A training-data generator for the Zero One Hack_01 "Prozesslogik lernen und benchmarken" track.

---

## 1. Motivation

The eval grades a model on three tasks (next-step prediction, sequence completion,
anomaly detection) over MOSFET / IGBT / IC sequences **plus an unknown family**.

Our current training data is built from the three reference families. Two problems:

1. **Vocabulary gap.** The three families, even recombined, never emit a real slice
   of the documented vocabulary — e.g. `IMPLANT LDD`, `DEPOSIT SPACER DIELECTRIC`,
   `ANISOTROPIC ETCH SPACER`, `LIGHT ANNEAL`, `MEASURE SPACER WIDTH`,
   `DEPOSIT BACKSIDE PROTECTION`, `MEASURE DEVICE PARAMETER`. A model has these in
   vocab (they appear in the rules doc) but has never seen them in context, so
   predictions at those positions are effectively random.

2. **Family overfit.** Template-per-family generation lets the model shortcut on the
   family token ("I see MOSFET → emit MOSFET cycles") instead of learning the
   process rules. That collapses on the unknown family at eval.

We want a generator that produces sequences which are: **valid** (pass the 10-rule
validator), **full-vocabulary** (every token appears in valid contexts), **maximally
diverse** (far beyond 3×combinatorics), and **family-robust** (no reliance on the
family token).

## 2. Approach: precondition-driven planning, not generate-then-check

The 10 rules are all *dependencies*: a token is legal only if an enabler already
happened — either within a recent window (deposit needs a clean ≤12; etch needs a
develop ≤12; implant needs an open mask ≤15; metal-etch needs litho ≤15; CMP needs a
deposit ≤6) or globally (tests after cure; ship after sort; pad-open after passivation;
backside-metal after cure; litho levels in order).

So we **generate by planning forward over a rule-state, emitting the enabler whenever a
precondition is unmet.** Invalid output becomes structurally impossible; the validator
drops to a backstop/audit. This is the symbolic half.

The only stochastic part is *which* legal unit to fire next and how to fill its slots —
**uniform random**, with a thin coverage controller on top. That is the entire "neuro"
half.

### Explicitly rejected: a "naturalness" model

We considered biasing selection toward process-sensible adjacencies (a hand-set or
learned Markov bias). **Rejected.** That is exactly the knowledge the downstream model
is supposed to learn; baking it into the generator leaks the answer into the data and
narrows diversity. Random-but-valid is *better* training data — the model cannot
shortcut on the generator's surface statistics, so it must learn the real constraints.
The generator stays dumb and correct; the model is what gets smart.

## 3. Architecture

```
canonical vocab (roles + synonyms)
        │
   unit library  ──────────────┐
        │                       │  (each unit: requires / effects / ordered slots)
   precondition planner  ◄──────┘
        │   walks PHASES order; fires a legal unit or injects a connector first
   uniform-random selection + coverage controller
        │
   family label sampling (+ training-time UNK dropout)
        │
   validate_sequence backstop  +  independent audit (validity / coverage / shape)
        │
   data/generated/neurosymbolic/<run>/...
```

### 3.1 Typed vocabulary (one-time annotation)

Every step name → a **role** (`clean`, `deposit`, `oxidize`, `litho_align`,
`litho_develop`, `inspect`, `etch`, `strip`, `implant`, `anneal`, `cmp`, `fill`,
`measure:*`, `test`, `logistics`). Synonyms collapse to one role + a synonym group
(`STRIP PHOTORESIST` / `STRIP RESIST`; `RCA CLEAN 1` / `WET CLEAN RCA1`).

### 3.2 Unit schema

A unit is pure data:

```
Unit:
  phase:     one of phases.PHASES        # which macro-slot it belongs to
  requires:  list of state predicates    # the 10 rules, as preconditions
  effects:   state mutations             # what it makes true for later units
  body:      ordered list of slots
Slot:
  role:      e.g. clean | deposit | etch | implant | measure
  choices:   [token, …]                  # synonyms / species → pick one
  optional:  p                           # include with probability p
  param:     level | species | film      # filled from context
```

A "block" is a unit; a "connection" between blocks is the planner satisfying the next
unit's `requires` — by reusing something already in the window, or injecting a tiny
**connector** unit. Connectors use the same schema, so there is no second mechanism.

### 3.3 Rule-state and the 10 rules as preconditions

| State field | Rule served | Predicate checked at the triggering token |
|---|---|---|
| `last_clean` | RULE_DEP_NO_CLEAN | deposit needs `pos − last_clean ≤ 12` |
| `last_develop` | RULE_ETCH_NO_MASK | etch needs `pos − last_develop ≤ 12` (spacer etch exempt) |
| `last_mask_open` (etch or develop) | RULE_IMPLANT_NO_MASK | implant needs `≤ 15` |
| `last_litho` (expose+develop) | RULE_METAL_ETCH_NO_LITHO | metal-etch needs `≤ 15` |
| `last_deposit` | RULE_CMP_NO_DEP | cmp needs `pos − last_deposit ≤ 6` |
| `level` | RULE_LITHO_LEVEL_SKIP | align(n) needs `level == n−1` |
| `passivation_cured` | RULE_TEST_BEFORE_PASSIVATION, RULE_BACKSIDE_BEFORE_PASSIVATION, RULE_PAD_OPEN_BEFORE_DEP | gate those units |
| `sort_done` | RULE_SHIP_BEFORE_TEST | gate SUFFIX |

Preconditions are evaluated at the exact position the triggering token will land
(accounting for the unit's own lead steps), which is what makes the guarantee real
rather than approximate.

### 3.4 The planner

```
emit PREFIX
for phase in PHASES order:
    while phase quota not met:
        goal = pick_legal_or_desired_unit(phase, state)   # uniform random + coverage
        for enabler in goal.unmet_preconditions(state):
            emit(enabler)                                  # inject CLEAN / LITHO connector
        emit(goal); state.update(goal)
emit SUFFIX
assert validate_sequence(seq) == []                        # backstop, should ~never fire
```

Walking `PHASES` in order makes the *global* ordering rules hold for free (passivation
is emitted before tests because the phase order says so). The "emit the enabler first"
move makes every *local* rule a precondition. Together: valid by construction.

### 3.5 The unit library (by phase)

**Connectors** (injected on demand; deliberately varied so block joins are not
stereotyped):
- `CLEAN` → one of {HF DIP · RCA1 · RCA2 · pre-clean · surface clean · post-etch clean}
  **or** a triplet `RCA1→RCA2→HF DIP [→dry]`. Sets `last_clean`.
- `LITHO(n)` → spin → soft bake → align(n) → expose(n) → [PEB] → develop →
  inspect(variant) → [hard bake]. Sets `last_develop`, `last_litho`, `last_mask_open`,
  `level=n`. The universal mask-opener.

**Prep phases:** `PREFIX` (fixed) · `INITIAL_MEAS` (optional measures) ·
`PRE_CLEAN` (mandatory clean block) · `SUBSTRATE_PREP` (variants: epitaxial-grow ·
epitaxial-check · backside-grind-first; family-agnostic) · `FIRST_OXIDATION`.

**Front-end cycles → all map to the single `PROCESS_CYCLE` phase.** All share the shape
*[deposit?] → LITHO → etch → strip → clean → [implant → anneal] → [measure]*, differing
only by slot fills: `OXIDE_CYCLE`, `POLY_CYCLE`, `FIELD_OXIDE_CYCLE`, `WINDOW_CYCLE`,
`DIELECTRIC_CYCLE`, and `SPACER_SUBBLOCK` (deposit spacer dielectric → anisotropic etch
spacer *(mask-exempt)* → implant LDD → [light anneal] → [measure spacer width]). The
spacer subblock is the unit that reaches the otherwise-unreachable spacer vocabulary,
and needs no mask, so it drops in after any poly cycle.

The **implant slot accepts all 8 species** (`WELL · LDD · P BODY · N BUFFER ·
CHANNEL STOP · DRAIN/CATHODE · N-TYPE · SOURCE DRAIN`) because the cycle's own
etch/develop satisfies `RULE_IMPLANT_NO_MASK`. Same for the **anneal slot**
(RTA · drive-in · light anneal). This is the main source of combinatorial coverage.

**Interconnect (→ `ILD_BLOCK`, `VIA_BLOCK`, `METAL_BLOCK` phases):**
`ILD_BLOCK` (clean → deposit ILD → densify → measure → cmp → measure planarity) ·
`VIA_CYCLE` (LITHO → via etch → strip → clean → barrier → seed(metal·tungsten) →
fill(metal·tungsten) → cmp → measure) · `METAL_CYCLE` (clean → deposit metal → anneal →
LITHO → metal etch → strip → clean → measure), repeatable 1–N, capped at `level ≤ 6`.

**Back-end (→ `PASSIVATION_BLOCK`, `BACKSIDE_BLOCK`, `FINAL_INSPECTION`, `TEST_SUITE`,
`SUFFIX`):** `PASSIVATION` (deposit → cure → measure → pad-open subblock; sets
`passivation_cured`) · `BACKSIDE` (requires `cured`; reaches `DEPOSIT BACKSIDE
PROTECTION`) · `FINAL_INSPECTION` (optional final measures) · `TEST_SUITE` (requires
`cured`; sets `sort_done`) · `SUFFIX` (requires `sort_done`).

### 3.6 Diversity levers (independent, multiplicative)

1. **Unit selection & count** — number of front-end cycles (3–6), which types, order;
   metal-layer count; whether a spacer subblock attaches.
2. **Synonyms** — every role has 2–3 spellings, chosen independently per occurrence.
3. **Optional steps** — each `[optional]` is an independent coin flip → `2^k` per seq.
4. **Parameter fills** — implant species (8), film, etch variant, anneal type,
   inspection name — free within each unit.
5. **Connector realization** — a required clean is met by an existing window step
   (no insert), a single clean, or a triplet; a required mask by any litho variant.

The **coverage controller** rides only on levers 2 and 4: when a token lags below its
floor, "least-used filler wins" for that role until it catches up. Everything else is
uniform random.

### 3.7 Family label decoupled from content (unknown-family robustness)

Content is generated family-agnostically, so the same content can wear any label.
Sample the family token from `{mosfet, igbt, ic} ∪ {synthetic_1..N} ∪ {UNK}`, and apply
**family-token dropout (~25% → UNK) at training time**. The model learns "family" is a
weak hint, not an identity — which is what survives an unseen family at eval. Optionally,
make each synthetic family a *biased policy preset* (e.g. favors tungsten vias + dense
measurements) so families are coherent-but-overlapping rather than disjoint.

### 3.8 Validation

- **Backstop:** every sequence runs through `validate_sequence`; the rare reject is
  dropped (should be ~0% if preconditions are correct).
- **Audit:** independent re-validation + coverage report (every token ≥ floor) + length
  histogram + phase/block shape check (see §4).

## 4. Binding to the eval segmentation layers

There are two segmentation layers in the repo; the units must agree with both.

- **`src/zero_hack/eval/phases.py`** — the fine 14-phase backbone (`PHASES`,
  `steps_to_phases`, `phase_runs`, `step_candidate_phases`). Currently standalone (only
  its test imports it).
- **`src/zero_hack/eval/blocks.py` + `_major_block` in `completion.py`** — the coarser
  grouping that computes the **scored** Block-level Accuracy metric (mirrored in the
  organizer's `data/industrial/eval_metrics.py`).

Bindings:

1. **`phases.PHASES` is the planner's macro-order — shared, not copied.** Each unit
   declares its phase; many units map to one phase (`PROCESS_CYCLE` holds all front-end
   cycle units + the spacer subblock).
2. **`step_candidate_phases` is a build-time consistency oracle.** It is the inverse of
   unit→token: a test asserts every token a unit emits has that unit's phase in its
   candidate set, and that `phase_runs(seq)` is a **monotonic subsequence of `PHASES`**
   for every generated sequence.
3. **Validate against the coarse layer too.** Tests also run `block_runs` (coarse) and
   assert it is well-formed, because that is the scored view.
4. **`phases.py` already anticipates the full-vocab units** — it maps
   `DEPOSIT SPACER DIELECTRIC`, `ANISOTROPIC ETCH SPACER`, `LIGHT ANNEAL`,
   `MEASURE SPACER WIDTH`, `MEASURE DEVICE PARAMETER` into `PROCESS_CYCLE`, confirming
   the backbone is right.
5. **Reconcile vocab drift before building.** `phases.py` lists tokens not in
   `generation_rules.md` (`ANNEAL DIELECTRIC`, `MEASURE SURFACE UNIFORMITY`). Pin one
   canonical vocabulary so generator, validator, labeler, and scorer all agree — else a
   unit could emit a token the scorer's labeler treats as `UNKNOWN`.

## 5. Module layout (per CLAUDE.md conventions)

```
src/zero_hack/data/synth/
    vocab.py      # canonical vocab, roles, synonym groups (single source of truth)
    state.py      # RuleState + precondition predicates (the 10 rules)
    units.py      # Unit/Slot schema + the unit & connector library
    planner.py    # forward planner + coverage controller + family-label sampling
scripts/generate_neurosymbolic.py     # CLI: count, seed, out-dir, coverage floor
data/generated/neurosymbolic/<run>/   # FAMILY,SEQUENCE_ID,STEP CSV + meta.json
tests/test_synth_units.py             # per-unit phase-consistency, requires/effects
tests/test_synth_planner.py           # property tests (see §6)
```

Family-token **dropout** is a training-time concern and lives in the data-loading path,
not in the generator (the generator records the sampled label; the loader masks it).

## 6. Testing strategy

- **Unit-level:** each unit's tokens are phase-consistent via `step_candidate_phases`;
  `requires`/`effects` match the rule each encodes.
- **Property (over many seeded sequences):**
  - every sequence passes `validate_sequence` (zero rejects expected);
  - `phase_runs(seq)` is a monotonic subsequence of `phases.PHASES`;
  - `block_runs(seq)` is well-formed (the scored view);
  - over an N-sequence run, every canonical-vocab token clears the coverage floor;
  - lengths fall within the eval band (reference ±20%, ~107–151 steps → target band).
- **Determinism:** same seed → identical output.

## 7. Decisions resolved

- Selection policy: **uniform random**, no naturalness model.
- Coverage: **least-used filler** for lagging tokens; run until floor reached.
- Family: **synthetic labels + UNK dropout**, content-agnostic.
- Recombination: **shared universal backbone (PHASES order) + free token fill in any
  compatible slot** — i.e. conservative backbone, aggressive coverage.

## 8. Open items

- Canonical vocabulary reconciliation (generation_rules.md vs phases.py) — must be done
  first.
- Number of synthetic family labels `N`, and whether synthetic families get biased
  policy presets or are pure relabels.
- Coverage floor `K` and target sequence-count per run.
- One engine or two presets: the same generator can produce the pseudo-OOD eval set via
  a more aggressive preset — confirm we want to unify.
