"""Mirror LeRobot training checkpoints to Google Drive and restore them.

LeRobot writes checkpoints to `<run>/checkpoints/<step>/` and points a
`checkpoints/last` symlink at the newest one. We train on Colab's local disk
(Drive's FUSE mount handles symlinks badly) and copy finished checkpoints to
Drive, keeping only the newest few: each ACT checkpoint is ~0.6 GB with
optimizer state, and free Drive has 15 GB.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

LAST = "last"


def _step_dirs(ckpt_root: Path) -> list[Path]:
    """Numeric checkpoint dirs, oldest first."""
    if not ckpt_root.is_dir():
        return []
    dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and not p.is_symlink() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def last_complete_step(local_run: str | Path) -> int | None:
    """Step that `checkpoints/last` points to. LeRobot only moves the symlink
    after a checkpoint is fully written, so everything up to it is complete."""
    last = Path(local_run) / "checkpoints" / LAST
    if not last.is_symlink():
        return None
    name = Path(os.readlink(last)).name
    return int(name) if name.isdigit() else None


def sync_to_drive(local_run: str | Path, drive_run: str | Path, keep: int = 2) -> list[str]:
    """Copy complete checkpoints not yet on Drive, then prune old ones on both
    sides (never the one `last` points to). Returns the step names copied."""
    local_ckpts = Path(local_run) / "checkpoints"
    drive_ckpts = Path(drive_run) / "checkpoints"
    last_step = last_complete_step(local_run)
    if last_step is None:
        return []

    drive_ckpts.mkdir(parents=True, exist_ok=True)
    copied = []
    for step_dir in _step_dirs(local_ckpts):
        if int(step_dir.name) > last_step or (drive_ckpts / step_dir.name).exists():
            continue
        tmp = drive_ckpts / f"{step_dir.name}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(step_dir, tmp)
        tmp.rename(drive_ckpts / step_dir.name)
        copied.append(step_dir.name)

    for root in (drive_ckpts, local_ckpts):
        for old in _step_dirs(root)[:-keep]:
            if int(old.name) != last_step:
                shutil.rmtree(old)
    return copied


def restore_from_drive(drive_run: str | Path, local_run: str | Path) -> Path | None:
    """Copy the newest Drive checkpoint to the local run dir and recreate the
    `last` symlink. Returns the `train_config.json` to resume from, or None if
    there is nothing to resume."""
    drive_steps = _step_dirs(Path(drive_run) / "checkpoints")
    if not drive_steps:
        return None
    newest = drive_steps[-1]
    local_ckpts = Path(local_run) / "checkpoints"
    local_ckpts.mkdir(parents=True, exist_ok=True)
    if not (local_ckpts / newest.name).exists():
        shutil.copytree(newest, local_ckpts / newest.name)

    last = local_ckpts / LAST
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(newest.name, target_is_directory=True)
    return last / "pretrained_model" / "train_config.json"


class CheckpointSyncer:
    """Background thread calling `sync_to_drive` every `interval` seconds,
    plus a final sync on exit."""

    def __init__(self, local_run, drive_run, keep: int = 2, interval: float = 60.0):
        self.args = (local_run, drive_run, keep)
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sync(self):
        try:
            for step in sync_to_drive(*self.args):
                print(f"[sync] checkpoint {step} -> Drive", flush=True)
        except Exception as exc:  # never kill training because of a sync hiccup
            print(f"[sync] failed: {exc!r}", flush=True)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sync()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sync()
        return False
