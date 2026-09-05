"""Live loss curve source: tail Lightning's CSV logger output.

The trainer's progress bar carries no loss values when stdout is not a
TTY, and piper1-gpl never logs anything named "training_loss" — the real
metrics are loss_g / loss_d / train_mel in training and val_loss / val_mel
in validation. So the Train screen's curve is fed from data, not bar
scraping: build_command points the trainer at Lightning's built-in
CSVLogger (same lightning_logs/version_N tree the default TensorBoard
logger uses, so checkpoint discovery is untouched), and the runner tails
metrics.csv while training runs, printing one line per epoch:

    Epoch 12: training_loss=2.8719
    Epoch 24: validation_loss=2.5301

Those lines flow through the log like any other output and the UI's
existing parser picks them up. loss_g vs val_loss is the honest pair:
val_loss is loss_g computed on the validation split, so the solid and
dashed curves share a scale.
"""
from __future__ import annotations

import csv
import io
import re
import threading
import time
from pathlib import Path

POLL_SECONDS = 2.0


def newest_metrics(runs_dir: Path, since_ts: float) -> Path | None:
    """The newest metrics.csv created at or after since_ts.

    Discriminating this run's file from previous runs' takes two guards.
    The epsilon absorbs filesystem timestamp granularity: a file created
    a hair after `since_ts` can legally carry an st_mtime a hair BEFORE
    it, and dropping it would silently kill the whole live curve. And
    because coarse mtimes can also tie (which would let a stale version
    dir win a plain max-by-mtime), the authoritative order is Lightning's
    version counter — it increments per run within default_root_dir, so
    the current run is always the highest version_N present.
    """
    if not runs_dir.exists():
        return None
    cands = [
        p for p in runs_dir.glob("lightning_logs/version_*/metrics.csv")
        if p.stat().st_mtime >= since_ts - 1.0
    ]
    if not cands:
        return None
    return max(cands, key=lambda p: (_version_no(p), p.stat().st_mtime))


def _version_no(p: Path) -> int:
    # The version component only counts where it actually is — the csv's
    # parent dir. A full-path search lets a project named "version_9"
    # outrank every real version dir.
    m = re.fullmatch(r"version_(\d+)", p.parent.name)
    return int(m.group(1)) if m else -1


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def tail_metrics(runs_dir: Path, stop: threading.Event, say,
                 poll: float = POLL_SECONDS) -> None:
    """Follow the run's metrics.csv until `stop` is set, printing one
    training_loss line per epoch (when the epoch closes or on final drain)
    and one validation_loss line per check epoch. Tolerates the file not
    existing yet (or ever, for a run that dies before the first step)."""
    started = time.time()
    path: Path | None = None
    said_train: set[int] = set()
    said_val: set[int] = set()

    def scan(final: bool) -> bool:
        nonlocal path
        # Re-resolve every poll, but only ever switch to a STRICTLY HIGHER
        # version: the epsilon window can admit a previous run's file when
        # the tail starts before this run's csv exists, and version_N
        # increments per run, so the current run's file is the only thing
        # that can legitimately outrank whatever was picked first.
        best = newest_metrics(runs_dir, started)
        if best is not None and (path is None
                                 or _version_no(best) > _version_no(path)):
            path = best
        if path is None:
            return False
        try:
            text = path.read_text()
        except OSError:
            return False
        # Re-read the whole file every poll and dedupe emissions by epoch.
        # A byte cursor breaks here: Lightning rewrites metrics.csv with a
        # new header the first time a val column appears, which leaves any
        # recorded offset pointing mid-file — the val point is lost and
        # later epochs replay from stale rows. Re-reading is stateless
        # against any rewrite; the said-sets keep output idempotent.
        rows = csv.DictReader(io.StringIO(text))
        per_epoch: dict[int, float] = {}
        vals: dict[int, float] = {}
        for rec in rows:
            epoch = _as_int(rec.get("epoch"))
            if epoch is None:
                continue
            loss_g = _as_float(rec.get("loss_g"))
            if loss_g is not None:
                per_epoch[epoch] = loss_g     # the epoch's last row wins
            val_loss = _as_float(rec.get("val_loss"))
            if val_loss is not None:
                vals[epoch] = val_loss
        # An epoch's training line is only final once a later epoch exists
        # — the newest stays open until the next poll closes it, or the
        # run ends and this final pass flushes it.
        epochs = sorted(per_epoch)
        closed = epochs if final else epochs[:-1]
        for epoch in closed:
            if epoch not in said_train:
                said_train.add(epoch)
                say(f"Epoch {epoch}: training_loss={per_epoch[epoch]:.4f}")
        for epoch in sorted(vals):
            if epoch not in said_val:
                said_val.add(epoch)
                say(f"Epoch {epoch}: validation_loss={vals[epoch]:.4f}")
        return True

    # The stop event doubles as the end-of-run signal: the runner sets it
    # once the subprocess exits, and this final pass catches rows written
    # between the last poll and exit, including the last epoch's line.
    while True:
        scan(final=False)
        if stop.is_set():
            scan(final=True)
            return
        time.sleep(poll)

