import pytest

from brpmm.config import data_dir


def test_env_var_overrides_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BRPMM_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path


def test_missing_env_dir_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("BRPMM_DATA_DIR", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        data_dir()
