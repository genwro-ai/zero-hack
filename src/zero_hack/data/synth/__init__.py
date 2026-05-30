"""Neurosymbolic process-sequence data generator.

Validity is baked in via a precondition-aware planner (see design doc
``docs/superpowers/specs/2026-05-30-neurosymbolic-sequence-generator-design.md``):
coherent process *units* are assembled in the official phase backbone, and any
local rule whose precondition is unmet is satisfied by injecting a minimal
connector before the triggering step. The official ``validate_sequence`` runs as
a backstop only.
"""

from zero_hack.data.synth.generator import (
    SynthSequence,
    generate_dataset,
    generate_one,
    sample_family_label,
    write_dataset_csv,
)
from zero_hack.data.synth.vocab import CANONICAL_VOCAB

__all__ = [
    "CANONICAL_VOCAB",
    "SynthSequence",
    "generate_dataset",
    "generate_one",
    "sample_family_label",
    "write_dataset_csv",
]
