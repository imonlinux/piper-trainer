"""Disk-backed job manager (design doc §1 — the single most important
decision).

Rules implemented here:

- **Jobs are processes, not threads.** Each job runs
  `python -m piper_trainer.api.runner <job-dir>` in a subprocess with its own
  session; a crashed or OOM-killed run cannot take the API down.
- **State is written to disk on every transition.** `jobs/<id>/job.json` is
  the record; this process is its single writer. The runner subprocess never
  touches job.json — it reports through structured stdout lines
  (`##TARGET`, `##PROGRESS`, `##RESULT`) and its exit code, which removes the
  two-writer race by construction.
- **The log file is the source of truth.** The supervisor tees every output
  line to `jobs/<id>/log.txt` and fans it out to subscribers; a browser
  refresh mid-run loses nothing.
- **One running job per project** unless `allow_parallel` is set (they
  contend for the same GPU and the same directories). Everything else queues
  in submit order.
- **Nothing restarts itself.** On startup, `rescan()` marks `running` jobs
  whose PID is gone as `failed` with `error: "interrupted"`. Jobs still
  `queued` from a previous process lifetime are surfaced but NOT adopted —
  `POST /jobs/{id}/start` adopts one deliberately.

Known limitation: a runner subprocess that survives its manager (API restart
without container teardown) is left alone while its PID lives; an operator
can see it in `ps` and the project lock still protects the directories.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import signal
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

KINDS = ("prepare", "transcribe", "validate", "clean", "restore",
         "train", "export", "ingest", "fetch-checkpoint", "preview")

_PROGRESS_RE = re.compile(r"\bEpoch (\d+)\b")
_DIRECTIVE_RE = re.compile(r"^##(TARGET|PROGRESS|RESULT) (.*)$")


class JobError(RuntimeError):
    """Unknown or un-manageable job."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_job(job_dir: Path) -> dict:
    return json.loads((job_dir / "job.json").read_text())


def _write_job(job_dir: Path, **fields) -> dict:
    """Atomic read-modify-write; job.json is small and written often."""
    job = _read_job(job_dir)
    job.update(fields)
    tmp = job_dir / "job.json.tmp"
    tmp.write_text(json.dumps(job, indent=2) + "\n")
    tmp.replace(job_dir / "job.json")
    return job


def _log_tail(job_dir: Path, limit: int = 300) -> str:
    """Last non-empty log line — the tail of an unhandled traceback.

    Reads only the end of the file: training logs grow to megabytes of
    progress bars, and this runs once per failed job.
    """
    path = job_dir / "log.txt"
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            chunk = fh.read().decode("utf-8", "replace")
    except OSError:
        return ""
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    if not lines:
        return ""
    line = lines[-1]
    return line[:limit - 1] + "…" if len(line) > limit else line


class JobManager:
    """Owns every subprocess and every job.json write for one workspace."""

    def __init__(self, workspace: Path, *, allow_parallel: bool = False,
                 cancel_grace: float = 30.0,
                 runner_cmd: Callable[[Path], list[str]] | None = None):
        self.workspace = Path(workspace)
        self.allow_parallel = allow_parallel
        self.cancel_grace = cancel_grace
        self._runner_cmd = runner_cmd or self._default_runner_cmd
        self._index: dict[str, Path] = {}          # job id -> job dir
        self._pending: list[str] = []              # queued, in submit order
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._progress: dict[str, dict] = {}       # last known progress merge
        self._cancel_requested: set[str] = set()
        self._busy: set[str] = set()               # project names with a run
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._tasks: set[asyncio.Task] = set()     # strong refs, never GC'd

    # ------------------------------------------------------------- discovery

    def projects(self) -> list[Path]:
        if not self.workspace.exists():
            return []
        return sorted(d for d in self.workspace.iterdir()
                      if d.is_dir() and (d / "project.json").exists())

    def iter_job_dirs(self) -> list[Path]:
        out = []
        for proj in self.projects():
            jobs = proj / "jobs"
            if not jobs.exists():
                continue
            for d in jobs.iterdir():
                if d.is_dir() and (d / "job.json").exists():
                    out.append(d)
        return out

    def job_dir(self, job_id: str) -> Path:
        try:
            return self._index[job_id]
        except KeyError:
            # not seen since startup — rescan once (covers jobs that appeared
            # on disk out of band, e.g. written by another process sharing
            # the volume), then give up
            self.rescan()
            try:
                return self._index[job_id]
            except KeyError:
                raise JobError(f"unknown job {job_id!r}") from None

    def rescan(self) -> list[str]:
        """Index every job on disk; mark orphaned runs as interrupted.

        Queued jobs from a previous lifetime are surfaced but never adopted
        automatically (design doc resolved decision #2).
        """
        interrupted = []
        for jd in self.iter_job_dirs():
            self._index[jd.name] = jd
            job = _read_job(jd)
            if job.get("state") != "running":
                continue
            if self._pid_alive(job.get("pid")):
                continue
            _write_job(jd, state="failed", error="interrupted",
                       finished_at=_now(), pid=None)
            interrupted.append(jd.name)
        return interrupted

    @staticmethod
    def _pid_alive(pid) -> bool:
        if not pid:
            return False
        try:
            import os
            os.kill(int(pid), 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True   # exists, owned by someone else
        except (ValueError, TypeError):
            return False

    # ------------------------------------------------------------------ read

    def get(self, job_id: str) -> dict:
        return _read_job(self.job_dir(job_id))

    def list_for_project(self, project_root: Path) -> list[dict]:
        jobs = project_root / "jobs"
        if not jobs.exists():
            return []
        out = [_read_job(d) for d in jobs.iterdir()
               if d.is_dir() and (d / "job.json").exists()]
        return sorted(out, key=lambda j: j["id"], reverse=True)

    def log(self, job_id: str) -> str:
        p = self.job_dir(job_id) / "log.txt"
        return p.read_text(errors="replace") if p.exists() else ""

    # ------------------------------------------------------------- subscribe

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        self._subs.setdefault(job_id, set()).add(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        subs = self._subs.get(job_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subs.pop(job_id, None)

    def _broadcast(self, job_id: str, event: dict) -> None:
        for q in self._subs.get(job_id, ()):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # a slow client must not stall a run
                pass

    # ---------------------------------------------------------------- submit

    async def submit(self, project_root: Path, kind: str,
                     stage: str | None = None,
                     params: dict | None = None,
                     run: bool = True) -> dict:
        """Create a job record; enqueue it unless `run=False`.

        `run=False` exists for callers that must stage inputs before a
        runner can exist (upload ingest): the job stays queued but never
        enters `_pending`, so `_pump` will not pick it up until an explicit
        `start()` after the last byte lands.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind!r}; expected one of "
                             f"{', '.join(KINDS)}")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = f"{stamp}-{kind}-{secrets.token_hex(2)}"
        job_dir = project_root / "jobs" / job_id
        job_dir.mkdir(parents=True)
        (job_dir / "log.txt").touch()
        job = {
            "id": job_id,
            "kind": kind,
            "stage": stage,
            "project": project_root.name,
            "params": params or {},
            "state": "queued",
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "pid": None,
            "progress": None,
            "result": None,
            "artifacts": [],
            "error": None,
        }
        (job_dir / "job.json").write_text(json.dumps(job, indent=2) + "\n")
        self._index[job_id] = job_dir
        self._progress[job_id] = {}
        if run:
            self._pending.append(job_id)
        self._broadcast(job_id, {"type": "state", "job": job})
        self._pump()
        return job

    async def start(self, job_id: str) -> dict:
        """Adopt a queued job deliberately (design doc: nothing restarts
        itself; the user decides)."""
        job_dir = self.job_dir(job_id)
        job = _read_job(job_dir)
        if job["state"] != "queued":
            raise JobError(f"job {job_id} is {job['state']}, not queued")
        if job_id not in self._pending:
            self._pending.append(job_id)
        self._pump()
        return _read_job(job_dir)

    # ------------------------------------------------------------- scheduling

    def _slot_free(self, project_name: str) -> bool:
        if self.allow_parallel:
            return project_name not in self._busy
        return not self._busy

    def _pump(self) -> None:
        """Start every pending job whose project slot is free, in order."""
        i = 0
        while i < len(self._pending):
            job_id = self._pending[i]
            jd = self._index.get(job_id)
            if jd is None:  # cancelled while queued
                self._pending.pop(i)
                continue
            project_name = jd.parent.parent.name
            if not self._slot_free(project_name):
                i += 1
                continue
            self._pending.pop(i)
            self._busy.add(project_name)
            task = asyncio.get_running_loop().create_task(self._supervise(job_id))
            self._tasks.add(task)
            task.add_done_callback(self._on_task_done)

    @staticmethod
    def _default_runner_cmd(job_dir: Path) -> list[str]:
        import sys
        return [sys.executable, "-m", "piper_trainer.api.runner",
                str(job_dir)]

    # ------------------------------------------------------------- supervision

    async def _supervise(self, job_id: str) -> None:
        jd = self.job_dir(job_id)
        project_name = jd.parent.parent.name
        try:
            # A cancel that arrived after _pump popped the id but before
            # this task's first step still sees job.json == "queued" here.
            # Honour it instead of overwriting with "running" and spawning
            # a process the UI already shows as cancelled.
            job = _read_job(jd)
            if job_id in self._cancel_requested or job["state"] == "cancelled":
                _write_job(jd, state="cancelled", finished_at=_now(),
                           error="cancelled while queued")
                self._broadcast(job_id, {
                    "type": "state", "job": _read_job(jd)})
                return
            job = _write_job(jd, state="running", started_at=_now())
            self._broadcast(job_id, {"type": "state", "job": job})
            proc = await asyncio.create_subprocess_exec(
                *self._runner_cmd(jd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )

            self._procs[job_id] = proc
            _write_job(jd, pid=proc.pid)
            result = None
            buf = b""
            with (jd / "log.txt").open("ab") as log_fh:
                while True:
                    chunk = await proc.stdout.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        m = re.search(rb"\r\n|\r|\n", buf)
                        if m is None:
                            break
                        raw, buf = buf[:m.start()], buf[m.end():]
                        text = raw.decode("utf-8", "replace")
                        log_fh.write(text.encode() + b"\n")
                        log_fh.flush()
                        self._broadcast(job_id, {"type": "log", "line": text})
                        result = self._on_line(job_id, text) or result
                    if len(buf) > 1 << 20:  # pathological unterminated line
                        text = buf.decode("utf-8", "replace")
                        buf = b""
                        log_fh.write(text.encode() + b"\n")
                        self._broadcast(job_id, {"type": "log", "line": text})

            code = await proc.wait()
            cancelled = job_id in self._cancel_requested
            if code == 0:
                state, error = "succeeded", None
            elif cancelled:
                state, error = "cancelled", "cancelled by user"
            else:
                state = "failed"
                # Prefer a runner-reported reason, then the last log line (the
                # tail of an unhandled traceback), then the bare exit code. The
                # numeric code stays in exit_code either way.
                error = None
                if isinstance(result, dict) and result.get("error"):
                    error = str(result["error"])
                else:
                    error = _log_tail(jd) or f"exited with code {code}"
            updates: dict = {"state": state, "exit_code": code,
                             "finished_at": _now(), "pid": None, "error": error}
            if result is not None:
                updates["result"] = result
                if isinstance(result, dict) and result.get("artifacts"):
                    updates["artifacts"] = result["artifacts"]
            job = _write_job(jd, **updates)
            self._broadcast(job_id, {"type": "state", "job": job})
        except Exception as exc:
            # Anything from here — a job.json write on a full disk, a decode
            # error — used to leak the project slot and orphan the runner.
            # Record it on the job; the record write itself may fail for the
            # same reason, so it gets its own guard.
            try:
                _write_job(jd, state="failed",
                           error=f"supervisor error: {exc}",
                           finished_at=_now(), pid=None)
                self._broadcast(job_id, {
                    "type": "state", "job": _read_job(jd)})
            except Exception:  # noqa: BLE001 — nothing left to do
                pass
        finally:
            # The slot release must happen on every path; with the default
            # single-slot rule a leaked name blocks every project forever.
            self._procs.pop(job_id, None)
            self._progress.pop(job_id, None)
            self._cancel_requested.discard(job_id)
            self._busy.discard(project_name)
            self._pump()

    def _on_task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logging.getLogger(__name__).error(
                "job supervisor task crashed: %r", exc)

    def _on_line(self, job_id: str, text: str) -> dict | None:
        """Interpret one output line; returns a captured RESULT payload."""
        m = _DIRECTIVE_RE.match(text)
        if m:
            tag, payload = m.group(1), m.group(2)
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return None
            if tag == "RESULT":
                return data
            progress = self._progress.setdefault(job_id, {})
            if tag == "TARGET":
                progress.update(data)
            else:
                progress.update(data)
            merged = dict(progress)
            jd = self.job_dir(job_id)
            job = _write_job(jd, progress=merged)
            self._broadcast(job_id, {"type": "progress",
                                     "progress": merged})
            self._broadcast(job_id, {"type": "state", "job": job})
            return None
        # Lightning's progress bar carries the epoch in plain output
        prog = self._progress.get(job_id)
        if prog and "unit" in prog:
            em = _PROGRESS_RE.search(text)
            if em and em.group(1) != str(prog.get("current")):
                prog["current"] = int(em.group(1))
                jd = self.job_dir(job_id)
                job = _write_job(jd, progress=dict(prog))
                self._broadcast(job_id, {"type": "progress",
                                         "progress": dict(prog)})
                self._broadcast(job_id, {"type": "state", "job": job})
        return None

    # ----------------------------------------------------------------- cancel

    async def cancel(self, job_id: str) -> dict:
        jd = self.job_dir(job_id)
        job = _read_job(jd)
        if job["state"] == "queued":
            if job_id in self._pending:
                self._pending.remove(job_id)
            # The supervisor task may already be scheduled but not yet run
            # (it pops from _pending in _pump before its first step). Mark
            # the intent so its first act is to honour the cancel instead
            # of overwriting this state with "running".
            self._cancel_requested.add(job_id)
            job = _write_job(jd, state="cancelled", finished_at=_now(),
                             error="cancelled while queued")
            self._broadcast(job_id, {"type": "state", "job": job})
            return job
        if job["state"] != "running":
            raise JobError(f"job {job_id} is {job['state']}; nothing to "
                           f"cancel")
        self._cancel_requested.add(job_id)
        proc = self._procs.get(job_id)
        if proc is not None and proc.returncode is None:
            self._signal_group(proc, signal.SIGTERM)
            loop = asyncio.get_running_loop()
            loop.call_later(self.cancel_grace, self._ensure_dead, proc)
        return _read_job(jd)

    @staticmethod
    def _signal_group(proc: asyncio.subprocess.Process, sig: int) -> None:
        import os
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _ensure_dead(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is None:
            self._signal_group(proc, signal.SIGKILL)
