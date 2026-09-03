"""Runtime settings for the API layer. Environment-first, like the rest of
the stack: the container is the deployment unit, and /workspace is the only
volume that matters."""
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path

DEFAULT_WORKSPACE = "/workspace"


def workspace() -> Path:
    """Root directory holding one subdirectory per voice project.

    PIPER_WORKSPACE wins. Otherwise /workspace when it exists (the
    container layout); on a bare host — where the README's
    `pip install -e '.[api]'; piper-trainer serve` path runs — fall back
    to ./workspace so the UI does not silently show an empty project list
    against a directory that does not exist.
    """
    env = os.environ.get("PIPER_WORKSPACE")
    if env:
        return Path(env)
    if Path(DEFAULT_WORKSPACE).exists():
        return Path(DEFAULT_WORKSPACE)
    return Path.cwd() / "workspace"


def allow_parallel() -> bool:
    """Run jobs for different projects concurrently (design doc §1.2: they
    contend for the same GPU, so this is an explicit opt-in)."""
    return os.environ.get("PIPER_ALLOW_PARALLEL", "").lower() in (
        "1", "true", "yes")


def cancel_grace() -> float:
    """Seconds between SIGTERM and SIGKILL on cancellation. Lightning writes
    last.ckpt at epoch end, so a generous grace loses at most one epoch."""
    return float(os.environ.get("PIPER_CANCEL_GRACE", "30"))


def version() -> str:
    try:
        return importlib.metadata.version("piper-trainer")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"
