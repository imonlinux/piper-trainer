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
import threading
import time
from pathlib import Path

POLL_SECONDS = 2.0


def newest_metrics(runs_dir: Path, since_ts: float) -> Path | None:
    """The newest metrics.csv created at or after since_ts.

    The timestamp filter matters on resumes: previous runs already have
    metrics.csv files under older version_N dirs, and this run's file
    only appears a moment after the trainer starts.
    """
    if not runs_dir.exists():
        return None
    cands = [
        p for p in runs_dir.glob("lightning_logs/version_*/metrics.csv")
        if p.stat().st_mtime >= since_ts
    ]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


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
    training_loss line per epoch (on epoch change or final drain) and one
    validation_loss line per check epoch. Tolerates the file not existing
    yet (or ever, for a run that dies before the first step)."""
    started = time.time()
    path: Path | None = None
    offset = 0
    header: list[str] | None = None
    said_val: set[int] = set()
    open_epoch: int | None = None
    open_loss: float | None = None

    def say_loss(epoch: int, loss: float) -> None:
        say(f"Epoch {epoch}: training_loss={loss:.4f}")

    def flush_open() -> None:
        nonlocal open_epoch, open_loss
        if open_epoch is not None and open_loss is not None:
            say_loss(open_epoch, open_loss)
        open_epoch, open_loss = None, None

    def handle(row: list[str]) -> None:
        nonlocal header, open_epoch, open_loss
        if header is None:
            if row and row[0].strip() == "epoch":
                header = row
            return  # rows before the header (or a stray blank) are noise
        rec = dict(zip(header, row))
        epoch = _as_int(rec.get("epoch"))
        if epoch is None:
            return
        loss_g = _as_float(rec.get("loss_g"))
        if loss_g is not None:
            if open_epoch is not None and epoch != open_epoch:
                flush_open()
            open_epoch, open_loss = epoch, loss_g
        val_loss = _as_float(rec.get("val_loss"))
        if val_loss is not None and epoch not in said_val:
            said_val.add(epoch)
            say(f"Epoch {epoch}: validation_loss={val_loss:.4f}")

    def drain() -> bool:
        nonlocal path, offset, header
        if path is None:
            path = newest_metrics(runs_dir, started)
            if path is None:
                return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size <= offset:
            return False
        with path.open("r", newline="") as f:
            f.seek(offset)
            chunk = f.read()
        # Never consume a partial line: the writer may be mid-flush, and
        # rows are only re-readable from the recorded offset.
        cut = chunk.rfind("\n")
        if cut == -1:
            return False
        offset += cut + 1
        for line in chunk[: cut + 1].splitlines():
            if line.strip():
                handle(next(csv.reader([line])))
        return True
        for line in chunk.splitlines():
            if line.strip():
                handle(next(csv.reader([line])))
        return True

    # The stop event doubles as the end-of-run signal: the runner sets it
    # once the subprocess exits, and this final pass catches rows written
    # between the last poll and exit, including the last epoch's line.
    while True:
        progressed = drain()
        if stop.is_set():
            flush_open()
            return
        if not progressed:
            time.sleep(poll)
