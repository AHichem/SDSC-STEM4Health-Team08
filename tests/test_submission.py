import pandas as pd
import pytest

from brpmm.submission import build_submission


@pytest.fixture
def template() -> pd.DataFrame:
    return pd.DataFrame(
        {"batch_id": ["A_001", "A_002", "B_002"], "project": ["A", "A", "B"], "label": "rejected"}
    )


def test_build_submission_maps_predictions(template):
    preds = {
        "A": pd.Series({"A_001": "accepted", "A_002": "rejected"}),
        "B": pd.Series({"B_002": "accepted"}),
    }
    sub = build_submission(preds, template)
    assert sub["label"].tolist() == ["accepted", "rejected", "accepted"]


def test_missing_prediction_is_rejected(template):
    preds = {"A": pd.Series({"A_001": "accepted"}), "B": pd.Series({"B_002": "accepted"})}
    with pytest.raises(ValueError, match="invalid/missing labels"):
        build_submission(preds, template)
