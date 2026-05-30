import pytest

from zero_hack.eval.score import score_anomaly, score_completion, score_task


def test_completion_uses_official_token_accuracy_denominator():
    metrics = score_completion(
        {"ex1": ["A", "B", "C"]},
        {"ex1": ["A"]},
    )

    assert metrics["all"]["token_accuracy"] == 1.0
    assert metrics["all"]["norm_edit_distance"] == pytest.approx(0.6667)


def test_completion_block_accuracy_uses_official_block_signature():
    metrics = score_completion(
        {"ex1": ["SPIN COAT PHOTORESIST", "OXIDE ETCH", "PARAMETRIC TEST"]},
        {"ex1": ["ALIGN MASK LEVEL 1", "METAL ETCH", "LEAKAGE TEST"]},
    )

    assert metrics["all"]["block_accuracy"] == 1.0


def test_anomaly_auc_uses_valid_score_as_positive_class():
    metrics = score_anomaly(
        {
            "valid": {"is_valid": 1, "rule": None},
            "invalid": {"is_valid": 0, "rule": "RULE_DEP_NO_CLEAN"},
        },
        {
            "valid": {"is_valid": 1, "score": 0.9, "predicted_rule": None},
            "invalid": {
                "is_valid": 0,
                "score": 0.1,
                "predicted_rule": "RULE_DEP_NO_CLEAN",
            },
        },
    )

    assert metrics["all"]["roc_auc"] == 1.0
    assert metrics["all"]["rule_attribution_accuracy"] == 1.0


def test_score_task_accepts_official_next_step_name(tmp_path):
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    gt.write_text("EXAMPLE_ID,NEXT_STEP\nex1,B\n", encoding="utf-8")
    pred.write_text("EXAMPLE_ID,RANK_1,RANK_2,RANK_3,RANK_4,RANK_5\nex1,B,,,,\n", encoding="utf-8")

    metrics = score_task("next-step", ground_truth=gt, predictions=pred)

    assert metrics["all"]["top1"] == 1.0
