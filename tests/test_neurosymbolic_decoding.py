import torch

from zero_hack.data import Vocabulary
from zero_hack.models.neurosymbolic.decoding import (
    NeurosymbolicHardHead,
    NeurosymbolicShapedHead,
    infer_process_state,
    shape_logits,
    topk_steps,
)


def _vocab(tokens: list[str]) -> Vocabulary:
    all_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK_STEP>", "<FAMILY_UNKNOWN>", *tokens]
    return Vocabulary(
        token_to_id={token: idx for idx, token in enumerate(all_tokens)},
        id_to_token=tuple(all_tokens),
    )


def test_hard_head_blocks_candidate_that_introduces_rule_violation() -> None:
    vocab = _vocab(["DEPOSIT METAL 1", "RCA CLEAN 1"])
    logits = torch.zeros(len(vocab.id_to_token))

    shaped = NeurosymbolicHardHead().shape_logits([], logits, vocab)

    assert torch.isneginf(shaped[vocab.token_to_id["DEPOSIT METAL 1"]])
    assert torch.isfinite(shaped[vocab.token_to_id["RCA CLEAN 1"]])


def test_hard_head_masks_special_tokens() -> None:
    vocab = _vocab(["RCA CLEAN 1"])
    logits = torch.zeros(len(vocab.id_to_token))

    shaped = shape_logits([], logits, vocab, mode="hard")

    assert torch.isneginf(shaped[vocab.token_to_id["<BOS>"]])
    assert torch.isneginf(shaped[vocab.token_to_id["<FAMILY_UNKNOWN>"]])
    assert torch.isfinite(shaped[vocab.token_to_id["RCA CLEAN 1"]])


def test_shaped_head_boosts_repair_step_after_etch_debt() -> None:
    prefix = [
        "PRE CLEAN WAFER",
        "THERMAL OXIDATION",
        "SPIN COAT PHOTORESIST",
        "SOFT BAKE",
        "ALIGN MASK LEVEL 1",
        "EXPOSE LITHO LEVEL 1",
        "DEVELOP PHOTORESIST",
        "OXIDE ETCH",
    ]
    vocab = _vocab(["STRIP PHOTORESIST", "MEASURE OXIDE THICKNESS"])
    logits = torch.zeros(len(vocab.id_to_token))

    shaped = NeurosymbolicShapedHead().shape_logits(prefix, logits, vocab)

    assert (
        shaped[vocab.token_to_id["STRIP PHOTORESIST"]]
        > shaped[vocab.token_to_id["MEASURE OXIDE THICKNESS"]]
    )


def test_infer_process_state_is_family_agnostic() -> None:
    state = infer_process_state(["SPIN COAT PHOTORESIST", "SOFT BAKE"])

    assert state.resist_on
    assert state.litho_stage == "soft_bake"


def test_infer_process_state_uses_canonical_phase_labels() -> None:
    state = infer_process_state(["PRE CLEAN WAFER", "DEPOSIT PASSIVATION", "CURE PASSIVATION"])

    assert state.phase == "PASSIVATION_BLOCK"


def test_topk_steps_skips_all_masked_tokens() -> None:
    vocab = _vocab(["RCA CLEAN 1"])
    logits = torch.full((len(vocab.id_to_token),), -torch.inf)

    assert topk_steps(logits, vocab, k=5) == []
