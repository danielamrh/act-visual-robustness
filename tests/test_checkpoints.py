import os
from collections import namedtuple

import pytest

import avr.checkpoints as ckpt
from avr.checkpoints import CheckpointSyncer, DriveSpaceError, last_complete_step, restore_from_drive, sync_to_drive


def _can_symlink(tmp_path):
    try:
        (tmp_path / "_probe").symlink_to(tmp_path, target_is_directory=True)
        return True
    except OSError:
        return False


@pytest.fixture
def need_symlinks(tmp_path):
    if not _can_symlink(tmp_path):
        pytest.skip("symlinks not permitted on this system")


def _make_ckpt(run, step, last=True, size=10):
    d = run / "checkpoints" / f"{step:06d}"
    (d / "pretrained_model").mkdir(parents=True)
    (d / "pretrained_model" / "train_config.json").write_text("{}")
    (d / "training_state").mkdir()
    (d / "training_state" / "optimizer.bin").write_bytes(b"x" * size)
    if last:
        link = run / "checkpoints" / "last"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(d.name, target_is_directory=True)


def _names(path):
    return sorted(p.name for p in path.iterdir()) if path.exists() else []


def test_nothing_to_sync_without_last(tmp_path):
    assert sync_to_drive(tmp_path / "local", tmp_path / "drive") == []


def test_mirrors_only_due_steps_and_keeps_newest(tmp_path, need_symlinks):
    local, drive = tmp_path / "local", tmp_path / "drive"
    for step in (4000, 8000, 12000):
        _make_ckpt(local, step)
    assert sync_to_drive(local, drive, drive_every=8000) == ["008000"]  # 12000 is not due, 4000 is older
    _make_ckpt(local, 16000)
    assert sync_to_drive(local, drive, drive_every=8000) == ["016000"]
    assert _names(drive / "checkpoints") == ["016000"]  # keep_drive=1: the older mirror is pruned
    assert _names(local / "checkpoints") == ["012000", "016000", "last"]  # keep_local=2
    assert sync_to_drive(local, drive, drive_every=8000) == []  # idempotent


def test_final_sync_copies_last_checkpoint_regardless_of_schedule(tmp_path, need_symlinks):
    local, drive = tmp_path / "local", tmp_path / "drive"
    for step in (8000, 12000):
        _make_ckpt(local, step)
    sync_to_drive(local, drive, drive_every=8000)
    assert sync_to_drive(local, drive, drive_every=8000, final=True) == ["012000"]
    assert _names(drive / "checkpoints") == ["012000"]


def test_in_progress_checkpoint_is_never_copied(tmp_path, need_symlinks):
    local, drive = tmp_path / "local", tmp_path / "drive"
    _make_ckpt(local, 8000)
    _make_ckpt(local, 16000, last=False)  # still being written
    assert last_complete_step(local) == 8000
    assert sync_to_drive(local, drive, drive_every=8000, final=True) == ["008000"]


def test_full_drive_fails_loudly_instead_of_copying(tmp_path, need_symlinks, monkeypatch):
    local, drive = tmp_path / "local", tmp_path / "drive"
    _make_ckpt(local, 8000, size=1000)
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(ckpt.shutil, "disk_usage", lambda p: Usage(10_000, 9_500, 500))
    with pytest.raises(DriveSpaceError, match="Empty the Drive trash"):
        sync_to_drive(local, drive, margin_bytes=0)
    assert _names(drive / "checkpoints") == []


def test_syncer_reports_failures_and_mirrors_log(tmp_path, need_symlinks, monkeypatch):
    local, drive = tmp_path / "local", tmp_path / "drive"
    _make_ckpt(local, 8000, size=1000)
    log = tmp_path / "train.log"
    log.write_text("step:8K loss:0.1\n")
    syncer = CheckpointSyncer(local, drive, drive_every=8000, mirror_files={log: drive / "train.log"})
    syncer._sync()
    assert (drive / "train.log").read_text() == log.read_text()
    assert "Drive ckpt 008000" in syncer.status() and "FAILING" not in syncer.status()

    _make_ckpt(local, 16000, size=1000)
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(ckpt.shutil, "disk_usage", lambda p: Usage(10_000, 9_990, 10))
    syncer._sync()
    assert syncer.errors == 1 and "SYNC FAILING" in syncer.status()


def test_restore_recreates_last(tmp_path, need_symlinks):
    local, drive = tmp_path / "local", tmp_path / "drive"
    for step in (5000, 10000):
        _make_ckpt(local, step)
    sync_to_drive(local, drive)

    fresh = tmp_path / "fresh_session"
    cfg = restore_from_drive(drive, fresh)
    assert cfg == fresh / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    assert cfg.exists()
    assert os.readlink(fresh / "checkpoints" / "last") == "010000"
    assert last_complete_step(fresh) == 10000


def test_restore_without_checkpoints(tmp_path):
    assert restore_from_drive(tmp_path / "drive", tmp_path / "local") is None
