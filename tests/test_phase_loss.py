import torch
from torch.nn import functional as F

from zero_hack.data import SequenceRecord, build_vocabulary
from zero_hack.eval.phases import PHASES, primary_phase_for_step
from zero_hack.models.phase_loss import (
    DEFAULT_PHASE_IGNORE_INDEX,
    NextPhaseLoss,
    PhaseLogitProjector,
    PhaseTargetLookup,
    build_token_phase_ids,
    phase_id_lookup,
)


def test_build_token_phase_ids_ignores_specials_and_maps_steps():
    records = [
        SequenceRecord("mosfet", "seq_1", ("RECEIVE WAFER LOT", "LOT RELEASE")),
    ]
    vocabulary = build_vocabulary(records)
    phase_ids = build_token_phase_ids(vocabulary)
    lookup = phase_id_lookup()

    assert phase_ids[vocabulary.pad_id] == DEFAULT_PHASE_IGNORE_INDEX
    assert phase_ids[vocabulary.token_to_id["<FAMILY_MOSFET>"]] == DEFAULT_PHASE_IGNORE_INDEX
    assert phase_ids[vocabulary.token_to_id["LOT RELEASE"]] == lookup["SUFFIX"]


def test_phase_logit_projector_pools_token_logits_by_phase():
    records = [
        SequenceRecord(
            "mosfet",
            "seq_1",
            ("RECEIVE WAFER LOT", "DEPOSIT METAL 1", "LOT RELEASE"),
        ),
    ]
    vocabulary = build_vocabulary(records)
    projector = PhaseLogitProjector.from_vocabulary(vocabulary)
    logits = torch.full((1, len(vocabulary.id_to_token)), -20.0)
    logits[0, vocabulary.token_to_id["DEPOSIT METAL 1"]] = 8.0

    phase_logits = projector(logits)

    predicted = PHASES[int(phase_logits.argmax(dim=-1).item())]
    assert predicted == primary_phase_for_step("DEPOSIT METAL 1")


def test_next_phase_loss_is_differentiable_and_matches_manual_ce():
    records = [
        SequenceRecord("mosfet", "seq_1", ("RECEIVE WAFER LOT", "LOT RELEASE")),
    ]
    vocabulary = build_vocabulary(records)
    target = torch.tensor([vocabulary.token_to_id["LOT RELEASE"]])
    logits = torch.randn(1, len(vocabulary.id_to_token), requires_grad=True)
    aux = NextPhaseLoss.from_vocabulary(vocabulary, weight=0.25)

    output = aux(logits, target, return_output=True)
    manual = F.cross_entropy(output.phase_logits, output.phase_targets) * 0.25

    assert torch.allclose(output.loss, manual)
    output.loss.backward()
    assert logits.grad is not None


def test_phase_target_lookup_uses_contextual_sequence_labels():
    record = SequenceRecord(
        "mosfet",
        "seq_1",
        (
            "SPIN COAT PHOTORESIST",
            "ALIGN MASK METAL",
            "METAL ETCH",
            "STRIP PHOTORESIST",
        ),
    )
    lookup = PhaseTargetLookup.from_records([record])
    batch = {
        "family": ["mosfet"],
        "sequence_id": ["seq_1"],
        "position": torch.tensor([0]),
    }

    targets = lookup.targets_for_batch(batch)

    assert targets.tolist() == [phase_id_lookup()["METAL_BLOCK"]]
