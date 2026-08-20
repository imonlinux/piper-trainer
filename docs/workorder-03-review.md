# Work order 03 — Review package

**Work order:** `docs/workorder-03-cli-completion.md` (CLI completion and concurrency safety)
**Date:** 2026-08-20
**Status:** all six tasks complete; nothing committed — tree left for review.

---

## 1. `git diff --stat`

```
 README.md                       |  25 ++++++--
 pyproject.toml                  |   6 ++
 src/piper_trainer/clean.py      |  95 ++++++++++++++++++++++++++-----
 src/piper_trainer/cli.py        | 132 ++++++++++++++++++++++++++-------------
 src/piper_trainer/train.py      |   4 +-
 src/piper_trainer/transcribe.py |  12 +++-
 tests/test_clean.py             | 133 ++++++++++++++++++++++++++++++++++++-
 tests/test_train.py             |  13 ++++
 tests/test_transcribe.py        |  19 ++++++
 9 files changed, 374 insertions(+), 65 deletions(-)
```

New (untracked) files:

| File | Purpose |
|---|---|
| `src/piper_trainer/lock.py` | Task 3 — flock-based project lock |
| `tests/test_lock.py` | 8 lock tests, incl. a real second process + SIGKILL |
| `docs/workorder-03-cli-completion.md` | the work order itself |

---

## 2. `git diff` — lock module and `cli.py`

`cli.py` diff: shown in the session transcript (all `--wait` flags via
`add_lock()`, the `_locked()` context-manager helper wrapping every mutating
handler, `--files-only` on restore, `--max-epochs` help now formatted from
`DEFAULT_MAX_EPOCHS`). Omitted here for space; reproduce with:

```bash
git diff src/piper_trainer/cli.py
```

`lock.py` is new, so `git diff` shows nothing. Full contents:

```python
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
```

---

## 3. `pytest -q` tail

```
........................................                                 [100%]
112 passed in 0.78s
```

(112 = 95 carried over from WO2 + 17 new: 8 lock, 6 restore, 1 describe
split ×2, 1 transcribe malformed-preserve, 1 help/constant agreement.)
Stable across three consecutive full-suite runs.

Acceptance greps:

| Check | Result |
|---|---|
| `pytest -q` | 112 passed |
| `grep -rn "os.getpid\|pidfile" src/` | no hits |

---

## 4. Mutation-check result

11/11 mutations caught, suite restored clean:

| Mutation | Caught by |
|---|---|
| 1a transcribe drops malformed lines | 1 test |
| 1b describe conflates the two repairs | 2 tests |
| 1b apply stats conflate the two repairs | 1 test |
| 1c help text drifts from the constant | 1 test |
| T2 restore never re-adds rows | 4 tests |
| T2 restore ignores files_only | 1 test |
| T2 most-recent manifest entry does not win | 1 test |
| T2 restore overwrites existing rows | 1 test |
| T3 lock file never removed | 2 tests |
| T3 busy message hides the holder | 2 tests |
| T3 --wait gives up immediately | 1 test |

---

## 5. Task 5 output — container verification, verbatim

All runs inside the rebuilt `piper-trainer:cuda` image (rebuild hit the
cached torch/piper layers; only the CLI layer rebuilt), real ffmpeg,
DeepFilterNet, and auditok, fresh container per command, `--user
$(id -u):$(id -g)`, workspace bind-mounted.

### 5a. Adversarial `raw/`

Sources created:

```
a.very.dotted.name.wav
foo bar.wav
foo?bar.wav
foo.mp3
foo.wav
Ünïcödé námé.m4a
```

First run:

```
$ piper-trainer prepare /workspace/wo3proj --energy-threshold 45
converted: 6
renamed: {'foo bar.wav': 'foo_bar_wav.wav', 'foo.mp3': 'foo_mp3.wav',
          'foo.wav': 'foo_wav.wav', 'foo?bar.wav': 'foo_bar_wav_2.wav',
          'Ünïcödé námé.m4a': '_n_c_d_n_m_.wav'}
denoised: 6
clips: 0          <- pure sine tones; see note below
finalized: 0
```

`work/48k/` listing:

```
a.very.dotted.name.wav
foo_bar_wav_2.wav
foo_bar_wav.wav
foo_mp3.wav
foo_wav.wav
_n_c_d_n_m_.wav
```

Second (`--force`) run: identical mapping, `diff` of the two listings →
`NAMES IDENTICAL ACROSS RUNS`. Six sources, six unique outputs, none
overwritten, mapping reported.

Note: the six sources were sine tones, which segmentation correctly
rejects (and DeepFilterNet removes outright). That is expected behaviour
for synthetic input, not a failure; 5b uses real speech.

### 5b. Idempotency with real audio

Speech synthesized with espeak-ng inside the container (100 s). Real
DeepFilterNet pass included:

```
$ prepare --energy-threshold 55
converted: 1 / denoised: 1 / clips: 11 / finalized: 11     (wav_count=11)

$ prepare --energy-threshold 45
converted: skipped / denoised: skipped / clips: 8 / finalized: 8
wav_count=8        <- reflects 45 ONLY; the 11 clips from th=55 are gone

$ prepare --energy-threshold 45        (fresh container)
converted: skipped / denoised: skipped / clips: skipped / finalized: skipped
wav_count=8        <- mtime_ns fingerprint: no spurious re-run after restart

$ prepare --energy-threshold 45 --force
converted: 1 / denoised: 1 / clips: 8 / finalized: 8
```

### 5c. Lock behaviour

Long `prepare` (30 MB of speech) started in a detached container:

```
lock file present: yes
{"command": "prepare", "started": "2026-08-20T22:22:20Z", "host": "c4f98cb14bb7"}
```

Concurrent `train` (fresh container, same host):

```
project is locked by 'prepare' (started 2026-08-20T22:22:20Z on
c4f98cb14bb7); 'train' cannot run concurrently. Wait for it to finish,
stop it, or pass --wait N to wait up to N seconds.
rc=1
```

Concurrent `validate` — does not block, does not acquire:

```
✗ [no-metadata] /workspace/wo3big/dataset/metadata.csv does not exist
```

Then `docker kill` (SIGKILL) the prepare container:

```
lock file still on disk (stale-looking): yes
train acquires immediately after SIGKILL: rc=0
lock file after the next command released it: no
```

The kernel released the flock on process death; the leftover file was an
empty shell the next acquirer detected via the inode check and cleaned up.
No stale-lock heuristics, no PID liveness guessing — the property a PID
file could not provide.

---

## 6. Disagreements and judgment calls

1. **Task 4 dependency list vs reality.** `src/` imports only `auditok` and
   `faster-whisper`; `numpy` and `soundfile` are never imported directly
   (they are transitive needs of faster-whisper/auditok). Followed the
   order's explicit list verbatim and noted the reason in pyproject.
2. **`describe()` signature extended** to `describe(plan, total_rows,
   endings=None)` — Task 1b requires reporting *which* ending repair
   applies, which needs file state the function didn't receive. CLI passes
   `metadata.line_endings()`.
3. **Restore ordering**: "sort by clip id" applied to the *appended* rows
   only; existing rows and malformed lines keep their positions (a
   whole-file sort would churn every future diff). Malformed lines are
   preserved through the restore rewrite via `raw_lines` — same class of
   surprise as Task 1a, though the order only asked about transcribe.
4. **`--retranscribe` does not preserve old malformed lines** — it is the
   explicit full-overwrite pass; Task 1a was scoped to `--only-missing`.
5. **`clean` locks only around `apply()`** — the dry-run/plan-build phase
   is read-only per the order's own table. `init` is unlocked (absent from
   the table; it creates the project the lock would live in). `train`
   holds the lock for the run including `--dry-run` (it writes
   `target_epochs` and checkpoints).
6. **Lock file unlink race** (beyond spec): the file is removed on release,
   so an opener can lock an inode the holder already unlinked. Defended by
   re-opening per attempt plus a post-acquire inode check — documented in
   the module docstring, exercised implicitly by the multiprocess tests.
7. **A test bug worth recording**: the first version of the SIGKILL test
   deadlocked — the killed child died holding a `multiprocessing.Event`'s
   internal mutex (those are *not* kernel-released, unlike flock), so the
   parent's `set()` blocked forever. The child now just sleeps; the
   incident is explained in a comment in `tests/test_lock.py`.
8. **README Status section updated** to describe what the CUDA container
   runs covered (naming, idempotency, locking) and what remains untested
   end to end (`transcribe`/`train`/`export`; CPU variant).
