from zero_hack.data.datasets import SequenceRecord, write_sequence_records
from zero_hack.models.common import load_split_records, report_splits, split_role


def _records(family: str, split: str, count: int = 2) -> list[SequenceRecord]:
    return [
        SequenceRecord(
            family=family,
            sequence_id=f"{split}_{idx}",
            steps=("RECEIVE WAFER LOT", "LOT IDENTIFICATION", "SHIP LOT"),
        )
        for idx in range(count)
    ]


def test_load_split_records_discovers_standard_and_diverse_splits(tmp_path):
    for family_name, family in [("MOSFET", "mosfet"), ("IGBT", "igbt"), ("IC", "ic")]:
        for split in ("train", "valid", "test", "test_standard", "test_diverse"):
            write_sequence_records(
                tmp_path / f"{family_name}_{split}.csv",
                _records(family, split),
            )

    bundle = load_split_records(tmp_path, holdout_family="ic")

    assert bundle.counts()["train"] == 4
    assert bundle.counts()["valid"] == 4
    assert bundle.counts()["test_standard"] == 4
    assert bundle.counts()["test_diverse"] == 4
    assert bundle.counts()["test_standard_ic"] == 2
    assert bundle.counts()["test_diverse_ic"] == 2

    splits = report_splits(bundle)
    assert "test_standard" in splits
    assert "test_diverse" in splits
    assert "test_standard_ic" in splits
    assert "test_diverse_ic" in splits
    assert split_role("test_standard_ic", bundle) == "ood/standard"
    assert split_role("test_diverse_mosfet", bundle) == "id/diverse"
