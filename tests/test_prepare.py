from pathlib import Path

import pytest

from brpmm.data import load_calibration
from brpmm.prepare import stage_raw


@pytest.fixture
def source(fake_root: Path) -> Path:
    (fake_root / "process A").mkdir()
    for name in ("BRPMM_template.csv", "Leaderboard_template_project2.ipynb"):
        (fake_root / name).write_text("x")
    return fake_root


def test_stage_copies_tree_and_loader_reads_it(source, tmp_path):
    dest = tmp_path / "staged"
    staged = stage_raw(source, dest)
    assert (dest / "process B" / "calibration data" / "calibration_data.csv") in staged
    ts, labels = load_calibration("B", root=dest)
    assert len(labels) == 2


def test_stage_can_link_and_rerun_skips_unchanged(source, tmp_path):
    dest = tmp_path / "linked"
    first = stage_raw(source, dest, link=True)
    target = dest / "BRPMM_template.csv"
    assert target.is_symlink()
    inode = target.lstat().st_ino
    assert stage_raw(source, dest, link=True) == first
    assert target.lstat().st_ino == inode  # not recreated


def test_refuses_same_folder(source):
    with pytest.raises(ValueError):
        stage_raw(source, source)
