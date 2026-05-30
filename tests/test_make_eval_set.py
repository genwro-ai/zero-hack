import importlib.util

from zero_hack import PROJECT_ROOT
from zero_hack.data import SequenceRecord

_SPEC = importlib.util.spec_from_file_location(
    "make_eval_set",
    PROJECT_ROOT / "scripts" / "make_eval_set.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
_MAKE_EVAL_SET = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MAKE_EVAL_SET)


def _record(sequence_id: str) -> SequenceRecord:
    return SequenceRecord(family="ic", sequence_id=sequence_id, steps=("RECEIVE WAFER LOT",))


def _family_record(family: str, sequence_id: str) -> SequenceRecord:
    return SequenceRecord(family=family, sequence_id=sequence_id, steps=("RECEIVE WAFER LOT",))


def test_anomaly_record_pools_are_disjoint_when_possible():
    records = [_record(f"seq_{idx}") for idx in range(5)]

    valid_records, invalid_records = _MAKE_EVAL_SET._split_anomaly_record_pools(
        records,
        n_valid=3,
    )

    assert [record.sequence_id for record in valid_records] == ["seq_0", "seq_1", "seq_2"]
    assert [record.sequence_id for record in invalid_records] == ["seq_3", "seq_4"]
    assert {record.sequence_id for record in valid_records}.isdisjoint(
        record.sequence_id for record in invalid_records
    )


def test_anomaly_record_pools_wrap_only_when_no_invalid_pool_exists():
    records = [_record(f"seq_{idx}") for idx in range(3)]

    valid_records, invalid_records = _MAKE_EVAL_SET._split_anomaly_record_pools(
        records,
        n_valid=5,
    )

    assert valid_records == records
    assert invalid_records == records


def test_eval_family_selection_honors_non_test_split():
    class Bundle:
        records = {
            "valid": [
                _family_record("mosfet", "m_valid"),
                _family_record("ic", "ic_valid"),
            ],
            "test_ic": [_family_record("ic", "ic_test")],
        }

    split, records = _MAKE_EVAL_SET._select_records(Bundle, ("ic",), "valid")

    assert split == "valid"
    assert [record.sequence_id for record in records] == ["ic_valid"]
