"""Advisory per-project lock.

Mechanism: an exclusive flock on `<project>/.piper-trainer.lock`. The kernel
releases it when the holding process dies — including SIGKILL and container
teardown — which a PID file cannot guarantee: a PID recorded inside one
container namespace is meaningless to a liveness check in another. The file's
*content* (command, start time, host) is written for the failure message and
is advisory only; the lock itself is the flock.

Scope: flock is per-kernel. This protects concurrent CLI runs and containers
on the SAME host, which is the deployment model. It does NOT protect a
project directory shared over NFS between machines.

The lock file is removed on release. Re-opening per attempt plus an
inode check after acquiring makes that safe: a process that locks an inode
the holder already unlinked detects the mismatch and retries on the live
path, so two processes can never both believe they hold the lock.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

LOCK_NAME = ".piper-trainer.lock"


class LockBusy(RuntimeError):
    """Another process holds the project lock."""


def _read_holder(path: Path) -> dict:
    try:
        return json.loads(path.read_text() or "{}")
    except (json.JSONDecodeError, OSError):
        return {}


def _describe_busy(path: Path, command: str) -> str:
    holder = _read_holder(path)
    who = holder.get("command", "an unknown command")
    started = holder.get("started", "unknown start time")
    host = holder.get("host", "unknown host")
    return (f"project is locked by '{who}' (started {started} on {host}); "
            f"'{command}' cannot run concurrently. Wait for it to finish, "
            f"stop it, or pass --wait N to wait up to N seconds.")


def _try_acquire(path: Path) -> int | None:
    """Open + flock the path. Returns an fd, or None if busy/stale.

    Re-opened on every call (never a cached fd) so a released-and-unlinked
    file cannot be waited on forever; the inode check after acquiring
    catches the open/unlink race.
    """
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    try:
        live = os.stat(path)
        mine = os.fstat(fd)
        if (live.st_dev, live.st_ino) != (mine.st_dev, mine.st_ino):
            # locked an inode the holder already unlinked — not the live lock
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            return None
    except FileNotFoundError:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return None
    return fd


@contextlib.contextmanager
def project_lock(project, command: str, wait: float = 0.0):
    """Hold the project lock for the duration of the context.

    wait: seconds to retry before raising LockBusy (0 = fail immediately).
    Released on normal exit, on exception, and by the kernel on death —
    no cleanup code is relied upon.
    """
    path = project.root / LOCK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, wait)
    fd = None
    while (fd := _try_acquire(path)) is None:
        if time.monotonic() >= deadline:
            raise LockBusy(_describe_busy(path, command))
        time.sleep(0.1)
    os.truncate(fd, 0)
    os.write(fd, json.dumps({
        "command": command,
        "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": socket.gethostname(),
    }).encode() + b"\n")
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)  # unlink BEFORE close: new openers find nothing
        os.close(fd)          # close releases the flock for stragglers
