"""Shared fixtures. Tests never touch the real challenge data."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    """Minimal copy of the challenge folder layout for process B."""
    for split in ("calibration data", "test data"):
        (tmp_path / "process B" / split).mkdir(parents=True)
    ts = pd.DataFrame(
        {
            "batch_id": ["B_001"] * 3 + ["B_005"] * 2,
            "aeration_rate": [8.0, 7.9, 8.1, 6.0, 6.1],
            "hot_cold_switch": 0.0,
        }
    )
    ts.to_csv(tmp_path / "process B" / "calibration data" / "calibration_data.csv", index=False)
    ts.to_csv(tmp_path / "process B" / "test data" / "test_data.csv", index=False)
    pd.DataFrame({"batch_id": ["B_001", "B_005"], "label": ["accepted", "rejected"]}).to_csv(
        tmp_path / "process B" / "calibration data" / "calibration_labels.csv", index=False
    )
    return tmp_path
