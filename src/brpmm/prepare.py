"""Stage the challenge data into ``data/raw`` (gitignored) so all code reads one place.

    python -m brpmm.prepare              # copy from the Renku / local organiser folder
    python -m brpmm.prepare --link       # symlink instead of copying (saves ~270 MB)
    python -m brpmm.prepare --source DIR

The folder layout is kept identical to the organisers' one, so ``brpmm.data``
works unchanged on either location.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from brpmm.config import PROCESSES, RAW_DIR, external_data_dir

_TOP_LEVEL_FILES = ("BRPMM_template.csv", "Leaderboard_template_project2.ipynb")


def _files_to_stage(source: Path) -> list[Path]:
    files = [source / name for name in _TOP_LEVEL_FILES]
    for process in PROCESSES:
        files += sorted(p for p in (source / f"process {process}").rglob("*") if p.is_file())
    missing = [f for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing in {source}: {[str(m) for m in missing]}")
    return files


def stage_raw(source: Path, dest: Path = RAW_DIR, link: bool = False) -> list[Path]:
    """Copy (or symlink) every challenge file from ``source`` to ``dest``.

    Files that already exist with the same size are skipped.
    """
    if source.resolve() == dest.resolve():
        raise ValueError("source and destination are the same folder")
    staged = []
    for src in _files_to_stage(source):
        dst = dest / src.relative_to(source)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            staged.append(dst)
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if link:
            dst.symlink_to(src.resolve())
        else:
            shutil.copy2(src, dst)
        staged.append(dst)
    return staged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--dest", type=Path, default=RAW_DIR)
    parser.add_argument("--link", action="store_true", help="symlink instead of copy")
    args = parser.parse_args(argv)
    source = args.source or external_data_dir()
    staged = stage_raw(source, args.dest, link=args.link)
    print(f"Staged {len(staged)} files from {source} -> {args.dest}")


if __name__ == "__main__":
    main()
