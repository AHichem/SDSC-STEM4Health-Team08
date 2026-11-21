import pandas as pd
import pytest

from brpmm.data import (
    TIME_COL,
    from_binary,
    load_calibration,
    load_test,
    sensor_columns,
    to_binary,
)


def test_load_calibration_adds_time_index_and_drops_constants(fake_root):
    ts, labels = load_calibration("B", root=fake_root)
    assert ts[TIME_COL].tolist() == [0, 1, 2, 0, 1]
    assert "hot_cold_switch" not in ts.columns
    assert sensor_columns(ts) == ["aeration_rate"]
    assert labels.to_dict() == {"B_001": "accepted", "B_005": "rejected"}


def test_load_test_keeps_constants_when_asked(fake_root):
    ts = load_test("B", root=fake_root, drop_constant=False)
    assert "hot_cold_switch" in ts.columns


def test_mismatched_labels_raise(fake_root):
    path = fake_root / "process B" / "calibration data" / "calibration_labels.csv"
    pd.DataFrame({"batch_id": ["B_001"], "label": ["accepted"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="do not match"):
        load_calibration("B", root=fake_root)


def test_binary_encoding_round_trips():
    labels = pd.Series({"A_1": "accepted", "A_2": "rejected"})
    y = to_binary(labels)
    assert y.tolist() == [0, 1]
    assert from_binary(y).equals(labels)
