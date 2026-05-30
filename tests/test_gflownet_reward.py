from zero_hack import INDUSTRIAL_DATA_DIR
from zero_hack.data import load_sequence_records
from zero_hack.models.gflownet.reward import ProcessReward, RewardConfig


def _ic_record():
    return load_sequence_records(INDUSTRIAL_DATA_DIR / "IC_variants.csv", family="ic")[0]


def test_reward_accepts_valid_reference_sequence():
    record = _ic_record()
    reward = ProcessReward(
        [record],
        config=RewardConfig(style_weight=0.0, min_length=100, max_length=200),
    )

    out = reward.evaluate(record.family, [], record.steps)

    assert out.is_valid
    assert out.violations == ()
    assert out.components["validity"] > 0
    assert out.components["phase"] > 0


def test_reward_rejects_prefix_mismatch():
    record = _ic_record()
    reward = ProcessReward([record], config=RewardConfig(style_weight=0.0))

    out = reward.evaluate(record.family, ["LOT IDENTIFICATION"], record.steps)

    assert not out.is_valid
    assert out.violations == ("PREFIX_MISMATCH",)


def test_reward_rejects_early_ship_lot():
    record = _ic_record()
    reward = ProcessReward([record], config=RewardConfig(style_weight=0.0))
    sequence = list(record.steps)
    sequence.insert(20, "SHIP LOT")

    out = reward.evaluate(record.family, [], sequence)

    assert not out.is_valid
    assert out.violations == ("EARLY_SHIP_LOT",)


def test_reward_rejects_short_terminal_sequence():
    record = _ic_record()
    reward = ProcessReward([record], config=RewardConfig(style_weight=0.0))
    sequence = list(record.steps[:20]) + ["SHIP LOT"]

    out = reward.evaluate(record.family, [], sequence)

    assert not out.is_valid
    assert out.violations == ("LENGTH_OUT_OF_RANGE",)
