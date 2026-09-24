import os

import pytest

from avr.checkpoints import last_complete_step, restore_from_drive, sync_to_drive


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


def _make_ckpt(run, step, last=True):
    d = run / "checkpoints" / f"{step:06d}"
    (d / "pretrained_model").mkdir(parents=True)
    (d / "pretrained_model" / "train_config.json").write_text("{}")
    (d / "training_state").mkdir()
    (d / "training_state" / "training_step.json").write_text(str(step))
    if last:
        link = run / "checkpoints" / "last"
        if link.is_symlink():
            link.unlink()
        link.symlink_to(d.name, target_is_directory=True)


def _names(path):
    return sorted(p.name for p in path.iterdir()) if path.exists() else []


def test_nothing_to_sync_without_last(tmp_path):
    assert sync_to_drive(tmp_path / "local", tmp_path / "drive") == []


def test_sync_copies_only_complete_and_prunes(tmp_path, need_symlinks):
    local, drive = tmp_path / "local", tmp_path / "drive"
    for step in (5000, 10000, 15000):
        _make_ckpt(local, step)
    _make_ckpt(local, 20000, last=False)  # still being written

    assert last_complete_step(local) == 15000
    copied = sync_to_drive(local, drive, keep=2)
    assert copied == ["005000", "010000", "015000"]
    assert _names(drive / "checkpoints") == ["010000", "015000"]
    # local: oldest pruned, in-progress and `last` untouched
    assert _names(local / "checkpoints") == ["015000", "020000", "last"]
    # idempotent
    assert sync_to_drive(local, drive, keep=2) == []


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
