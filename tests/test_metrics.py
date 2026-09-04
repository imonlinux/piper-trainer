"""Tests for the metrics.csv tail that feeds the live loss curve."""
import threading
import time
from pathlib import Path

from piper_trainer.metrics import newest_metrics, tail_metrics


def feed(path: Path, rows: list[list[str]]) -> None:
    with path.open("a", newline="") as f:
        for row in rows:
            f.write(",".join(row) + "\n")


def wait_for(cond, timeout=5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_tail_prints_one_loss_line_per_epoch(tmp_path):
    """Header, two training rows in epoch 0 (last wins), an epoch-1 row,
    and a validation row. The epoch-0 line flushes when epoch 1 starts;
    epoch 1's own line flushes on the final drain."""
    vdir = tmp_path / "lightning_logs" / "version_0"
    vdir.mkdir(parents=True)
    csv_path = vdir / "metrics.csv"

    said: list[str] = []
    stop = threading.Event()
    tail = threading.Thread(
        target=tail_metrics, args=(tmp_path, stop, said.append, 0.01),
        daemon=True)
    tail.start()

    feed(csv_path, [
        ["epoch", "step", "loss_g", "loss_d", "train_mel", "val_loss", "val_mel"],
        ["0", "10", "2.9", "0.8", "1.1", "", ""],
        ["0", "20", "2.8719", "0.7", "1.0", "", ""],
        ["0", "20", "", "", "", "2.5301", "0.9"],
        ["1", "30", "2.75", "0.7", "0.98", "", ""],
    ])
    # epoch 0's line flushes the moment epoch 1's row lands
    assert wait_for(
        lambda: "Epoch 0: training_loss=2.8719" in said
        and "Epoch 0: validation_loss=2.5301" in said)

    stop.set()
    tail.join(timeout=5)
    assert not tail.is_alive()
    # exactly one training line per epoch, one val line per check epoch;
    # per-epoch ordering is unspecified (the val row shares epoch 0)
    assert sorted(said) == sorted([
        "Epoch 0: training_loss=2.8719",
        "Epoch 0: validation_loss=2.5301",
        "Epoch 1: training_loss=2.7500",
    ])


def test_tail_ignores_older_runs_metrics(tmp_path):
    """On a resume, previous runs' metrics.csv files exist under older
    version dirs; only this run's file may be tailed."""
    old = tmp_path / "lightning_logs" / "version_0"
    old.mkdir(parents=True)
    feed(old / "metrics.csv", [
        ["epoch", "step", "loss_g"],
        ["41", "10", "1.5"],
    ])

    said: list[str] = []
    stop = threading.Event()
    tail = threading.Thread(
        target=tail_metrics, args=(tmp_path, stop, said.append, 0.01),
        daemon=True)
    tail.start()
    time.sleep(0.05)  # let it sample while only the stale file exists

    new = tmp_path / "lightning_logs" / "version_1"
    new.mkdir(parents=True)
    feed(new / "metrics.csv", [
        ["epoch", "step", "loss_g"],
        ["41", "20", "1.234"],
    ])
    # A single-epoch run has no epoch change, so its line flushes at stop.
    # Equality proves the stale file's epoch 41 (1.5000) never leaked in.
    stop.set()
    tail.join(timeout=5)
    assert said == ["Epoch 41: training_loss=1.2340"]


def test_newest_metrics_filters_by_timestamp(tmp_path):
    v0 = tmp_path / "lightning_logs" / "version_0"
    v1 = tmp_path / "lightning_logs" / "version_1"
    v0.mkdir(parents=True)
    v1.mkdir(parents=True)
    (v0 / "metrics.csv").write_text("epoch,step\n")
    time.sleep(0.02)
    marker = time.time()
    time.sleep(0.02)
    (v1 / "metrics.csv").write_text("epoch,step\n")

    assert newest_metrics(tmp_path, since_ts=0) == v1 / "metrics.csv"
    assert newest_metrics(tmp_path, since_ts=marker) == v1 / "metrics.csv"
    assert newest_metrics(tmp_path, since_ts=time.time() + 5) is None


def test_tail_survives_missing_file(tmp_path):
    """A run that dies before the first step writes no metrics.csv; the
    tail must exit cleanly when stopped."""
    said: list[str] = []
    stop = threading.Event()
    tail = threading.Thread(
        target=tail_metrics, args=(tmp_path, stop, said.append, 0.01),
        daemon=True)
    tail.start()
    time.sleep(0.05)
    stop.set()
    tail.join(timeout=5)
    assert said == []
    assert not tail.is_alive()
