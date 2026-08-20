"""Tests for the flock-based project lock (Task 3)."""
import multiprocessing
import time
from pathlib import Path

import pytest

from piper_trainer import cli, lock
from piper_trainer.config import Project


@pytest.fixture
def proj(tmp_path) -> Project:
    return Project(root=tmp_path / "proj", name="t")


def test_lock_file_created_and_removed(proj):
    path = proj.root / lock.LOCK_NAME
    with lock.project_lock(proj, command="prepare"):
        assert path.exists()
        holder = lock._read_holder(path)
        assert holder["command"] == "prepare"
        assert "started" in holder and "host" in holder
    assert not path.exists()


def test_second_acquire_in_same_process_fails(proj):
    # two separate opens = two file descriptions = a genuine conflict,
    # not the same-fd no-op that would pass for the wrong reason
    with lock.project_lock(proj, command="prepare"):
        with pytest.raises(lock.LockBusy):
            with lock.project_lock(proj, command="train"):
                pass


def test_lock_released_on_exception(proj):
    path = proj.root / lock.LOCK_NAME
    with pytest.raises(RuntimeError, match="boom"):
        with lock.project_lock(proj, command="prepare"):
            raise RuntimeError("boom")
    assert not path.exists()
    with lock.project_lock(proj, command="train"):
        pass  # immediately re-acquirable


def test_wait_times_out(proj):
    with lock.project_lock(proj, command="prepare"):
        t0 = time.monotonic()
        with pytest.raises(lock.LockBusy, match="locked by 'prepare'"):
            with lock.project_lock(proj, command="train", wait=0.4):
                pass
        assert time.monotonic() - t0 >= 0.3
    # and succeeds once free within the wait window
    with lock.project_lock(proj, command="train", wait=1.0):
        pass


def test_busy_message_names_holder(proj):
    with lock.project_lock(proj, command="transcribe"):
        with pytest.raises(lock.LockBusy) as exc:
            with lock.project_lock(proj, command="train"):
                pass
        assert "transcribe" in str(exc.value)
        assert "started" in str(exc.value)


def _child_hold_lock(root: str) -> None:
    # No synchronization primitive back to the parent: a SIGKILLed process
    # does not release multiprocessing locks, and the parent would deadlock
    # on set(). The parent always kills the child; the flock — unlike a
    # multiprocessing Event — is released by the kernel on death.
    proj = Project(root=Path(root), name="t")
    with lock.project_lock(proj, command="prepare"):
        time.sleep(60)


def test_real_second_process_conflicts_and_kill9_releases(proj):
    """The property a PID file cannot give: SIGKILL releases immediately."""
    p = multiprocessing.Process(target=_child_hold_lock, args=(str(proj.root),))
    p.start()
    deadline = time.monotonic() + 5
    while not (proj.root / lock.LOCK_NAME).exists():
        assert time.monotonic() < deadline, "child never created the lock"
        time.sleep(0.02)
    with pytest.raises(lock.LockBusy):
        with lock.project_lock(proj, command="train", wait=0.1):
            pass
    p.kill()  # SIGKILL — no cleanup code runs
    p.join(timeout=5)
    assert not p.is_alive()
    # the kernel released it: acquire succeeds at once, no stale lock
    with lock.project_lock(proj, command="train", wait=5):
        pass


def test_read_only_command_does_not_acquire(proj, tmp_path, monkeypatch):
    """validate must never block or create a lock — the UI calls it mid-job."""
    proj.ensure()  # no metadata: validate still runs (and reports), unlocked
    rc = cli.main(["validate", str(proj.root)])
    assert rc == 1  # findings exist...
    assert not (proj.root / lock.LOCK_NAME).exists()  # ...but no lock touched
