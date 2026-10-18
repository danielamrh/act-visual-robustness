"""Mirror LeRobot training checkpoints to Google Drive and restore them.

LeRobot writes checkpoints to `<run>/checkpoints/<step>/` and points a
`checkpoints/last` symlink at the newest one. We train on Colab's local disk
(Drive's FUSE mount handles symlinks badly) and copy checkpoints to Drive.

Drive is the scarce resource: each ACT checkpoint is ~0.6 GB with optimizer
state, free Drive has 15 GB, and files deleted through the Colab mount go to
the Drive *trash*, which keeps counting against the quota. When the quota is
exhausted, Colab's FUSE layer accepts writes locally and fails to upload them
silently. So we
  - mirror only every `drive_every` steps (plus the final checkpoint) and
    keep only the newest one on Drive,
  - check the free Drive space before every copy and fail loudly,
  - expose the sync state for the notebook's status line.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path

LAST = "last"
GB = 1e9


class DriveSpaceError(OSError):
    pass


def _step_dirs(ckpt_root: Path) -> list[Path]:
    """Numeric checkpoint dirs, oldest first."""
    if not ckpt_root.is_dir():
        return []
    dirs = [p for p in ckpt_root.iterdir() if p.is_dir() and not p.is_symlink() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def last_complete_step(local_run: str | Path) -> int | None:
    """Step that `checkpoints/last` points to. LeRobot only moves the symlink
    after a checkpoint is fully written, so everything up to it is complete."""
    last = Path(local_run) / "checkpoints" / LAST
    if not last.is_symlink():
        return None
    name = Path(os.readlink(last)).name
    return int(name) if name.isdigit() else None


def drive_free_bytes(path: str | Path) -> int:
    """Free bytes on the filesystem holding `path` (the Drive quota on Colab's mount)."""
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def sync_to_drive(
    local_run: str | Path,
    drive_run: str | Path,
    keep_drive: int = 1,
    keep_local: int = 2,
    drive_every: int | None = None,
    final: bool = False,
    margin_bytes: float = 0.5 * GB,
) -> list[str]:
    """Copy the newest complete checkpoint that is due (step % drive_every == 0,
    or any step when `final`) and not yet on Drive; then prune old checkpoints
    (Drive: keep `keep_drive`, local: keep `keep_local`, never the one `last`
    points to). Raises DriveSpaceError instead of copying into a full Drive.
    Returns the step names copied."""
    local_ckpts = Path(local_run) / "checkpoints"
    drive_ckpts = Path(drive_run) / "checkpoints"
    last_step = last_complete_step(local_run)
    if last_step is None:
        return []

    due = [d for d in _step_dirs(local_ckpts)
           if int(d.name) <= last_step and (final or not drive_every or int(d.name) % drive_every == 0)]  # fmt: skip
    copied = []
    on_drive = {d.name for d in _step_dirs(drive_ckpts)}
    if due and due[-1].name not in on_drive and (not on_drive or int(due[-1].name) > max(map(int, on_drive))):
        step_dir = due[-1]
        need = _dir_size(step_dir)
        free = drive_free_bytes(drive_ckpts)
        if free < need + margin_bytes:
            raise DriveSpaceError(
                f"Drive almost full: {free / GB:.1f} GB free, checkpoint {step_dir.name} needs "
                f"{need / GB:.1f} GB (+{margin_bytes / GB:.1f} GB margin). Empty the Drive trash."
            )
        drive_ckpts.mkdir(parents=True, exist_ok=True)
        tmp = drive_ckpts / f"{step_dir.name}.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(step_dir, tmp)
        tmp.rename(drive_ckpts / step_dir.name)
        copied.append(step_dir.name)

    for root, keep in ((drive_ckpts, keep_drive), (local_ckpts, keep_local)):
        for old in _step_dirs(root)[:-keep]:
            if root is drive_ckpts or int(old.name) != last_step:
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
    """Background thread calling `sync_to_drive` every `interval` seconds and
    mirroring small files (e.g. the training log, kept on local disk so a Drive
    problem can never block training); a final sync on exit copies the last
    checkpoint regardless of `drive_every`. `status()` summarizes the state for
    the notebook's status line."""

    def __init__(self, local_run, drive_run, keep_drive: int = 1, keep_local: int = 2,
                 drive_every: int | None = None, interval: float = 60.0, mirror_files: dict | None = None):
        self.local_run, self.drive_run = local_run, drive_run
        self.kwargs = dict(keep_drive=keep_drive, keep_local=keep_local, drive_every=drive_every)
        self.mirror_files = dict(mirror_files or {})
        self.interval = interval
        self.errors = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sync(self, final: bool = False):
        try:
            for step in sync_to_drive(self.local_run, self.drive_run, final=final, **self.kwargs):
                print(f"[sync] checkpoint {step} -> Drive", flush=True)
            for src, dst in self.mirror_files.items():
                if os.path.exists(src):
                    Path(dst).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
            self.last_error = None
        except Exception as exc:  # never kill training because of a sync problem, but make it visible
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[sync] FAILED ({self.errors}x): {self.last_error}", flush=True)

    def status(self) -> str:
        steps = _step_dirs(Path(self.drive_run) / "checkpoints")
        parts = [f"Drive ckpt {steps[-1].name if steps else '-'}"]
        try:
            parts.append(f"Drive free {drive_free_bytes(self.drive_run) / GB:.1f} GB")
        except OSError:
            parts.append("Drive free ?")
        if self.last_error:
            parts.append(f"SYNC FAILING: {self.last_error[:80]}")
        return " ".join(parts)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sync()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sync(final=True)
        return False
