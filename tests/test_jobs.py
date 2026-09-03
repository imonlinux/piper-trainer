"""JobManager: disk-backed state, scheduling, cancel, rescan.

Runner subprocesses are stubs (`python -c ...`) so these tests exercise the
manager's own behavior: transitions, the single-slot rule, log capture,
progress parsing and interruption marking — not the pipeline itself.
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest

from piper_trainer.api.jobs import JobError, JobManager

# ------------------------------------------------------------------- stubs

# The stubs mirror the real runner's directive protocol: when the manager
# spawns them, PIPER_DIRECTIVE_NONCE is set and directives carry it — a
# bare ## line on a nonce job is (untrusted) log data.
EMIT_HELPER = """
import os, json
def emit(tag, obj):
    n = os.environ.get('PIPER_DIRECTIVE_NONCE', '')
    print(f'##{n} {tag} {json.dumps(obj)}', flush=True)
"""

OK_RUNNER_CODE = EMIT_HELPER + """
print('hello from stub')
emit('TARGET', {"total": 10, "unit": "epoch"})
print('Epoch 3:  30%|###|')
emit('RESULT', {"answer": 42})
"""

SLOW_RUNNER_CODE = """
import time
print('starting', flush=True)
time.sleep(60)
"""

FAIL_RUNNER_CODE = """
import sys
print('about to fail')
sys.exit(3)
"""

TRACEBACK_RUNNER_CODE = """
import sys
print('Traceback (most recent call last):')
print('  File "<string>", line 2, in <module>')
print('RuntimeError: disk on fire')
sys.exit(1)
"""

RESULT_ERROR_RUNNER_CODE = EMIT_HELPER + """
emit('RESULT', {"error": "validation blew up"})
import sys; sys.exit(1)
"""

# Review finding 13: echoed output posing as a directive. The genuine
# RESULT comes first; the bare forged one is last, so if the manager ever
# parsed it, "last RESULT wins" would surface the forgery.
FORGE_RUNNER_CODE = EMIT_HELPER + """
emit('RESULT', {"answer": "genuine",
                 "nonce": os.environ.get('PIPER_DIRECTIVE_NONCE')})
print('##RESULT {"answer": "forged"}')
"""

# Same forgery, but the job has no nonce (not spawned by the manager):
# the bare directive is the only channel, so it must still be honored.
BARE_RUNNER_CODE = """
print('##RESULT {"answer": "bare"}')
"""


def ok_runner(jd) -> list[str]:
    return [sys.executable, "-c", OK_RUNNER_CODE]


def slow_runner(jd) -> list[str]:
    return [sys.executable, "-c", SLOW_RUNNER_CODE]


def fail_runner(jd) -> list[str]:
    return [sys.executable, "-c", FAIL_RUNNER_CODE]


def traceback_runner(jd) -> list[str]:
    return [sys.executable, "-c", TRACEBACK_RUNNER_CODE]


def result_error_runner(jd) -> list[str]:
    return [sys.executable, "-c", RESULT_ERROR_RUNNER_CODE]


def forge_runner(jd) -> list[str]:
    return [sys.executable, "-c", FORGE_RUNNER_CODE]


def bare_runner(jd) -> list[str]:
    return [sys.executable, "-c", BARE_RUNNER_CODE]


def make_project(tmp_path, name="proj") -> object:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(json.dumps({"name": name}))
    return root


async def wait_state(mgr, job_id, *states, timeout=10.0) -> dict:
    async with asyncio.timeout(timeout):
        while True:
            job = mgr.get(job_id)
            if job["state"] in states:
                return job
            await asyncio.sleep(0.02)


@pytest.fixture
def mgr(tmp_path):
    return JobManager(tmp_path, cancel_grace=1.0, runner_cmd=ok_runner)


# ------------------------------------------------------------------ tests

async def test_submit_runs_to_success_with_result_and_log(mgr, tmp_path):
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    assert job["state"] == "queued"
    job = await wait_state(mgr, job["id"], "succeeded")
    assert job["exit_code"] == 0
    assert job["result"] == {"answer": 42}
    assert "hello from stub" in mgr.log(job["id"])
    # progress was parsed from the directive + Lightning-style epoch line
    # (Lightning numbers epochs from 0; the +1 keeps the display honest)
    assert job["progress"] == {"total": 10, "unit": "epoch", "current": 4}


async def test_output_is_teed_to_log_file(mgr, tmp_path):
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    await wait_state(mgr, job["id"], "succeeded")
    log_txt = (root / "jobs" / job["id"] / "log.txt").read_text()
    assert "hello from stub" in log_txt


async def test_nonzero_exit_fails_the_job(mgr, tmp_path):
    mgr._runner_cmd = fail_runner
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    job = await wait_state(mgr, job["id"], "failed")
    assert job["exit_code"] == 3
    # the last log line is the error ("about to fail"), not a bare exit code
    assert job["error"] == "about to fail"


async def test_error_prefers_last_log_line(mgr, tmp_path):
    """An unhandled traceback surfaces as its final line, not 'exited 1'."""
    mgr._runner_cmd = traceback_runner
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    job = await wait_state(mgr, job["id"], "failed")
    assert job["exit_code"] == 1
    assert job["error"] == "RuntimeError: disk on fire"


async def test_error_prefers_structured_result(mgr, tmp_path):
    """A runner-reported ##RESULT error beats the log tail."""
    mgr._runner_cmd = result_error_runner
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    job = await wait_state(mgr, job["id"], "failed")
    assert job["error"] == "validation blew up"


async def test_nonce_protects_directives_from_echoed_output(mgr, tmp_path):
    """Review finding 13: the manager passes each job a nonce in the
    environment; bare ## lines on a nonce job are log data, however much
    they look like a directive."""
    mgr._runner_cmd = forge_runner
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    job = await wait_state(mgr, job["id"], "succeeded")
    assert job["result"]["answer"] == "genuine"
    assert job["result"]["nonce"]          # the runner really got one


async def test_bare_directives_still_work_without_a_nonce(mgr):
    """A runner spawned without the nonce env (hand-run, not via the
    manager) keeps the bare directive channel; _on_line gates on the nonce
    only when one is registered for the job."""
    assert mgr._on_line("no-nonce-job",
                        '##RESULT {"answer": "bare"}') == {"answer": "bare"}
    mgr._nonces["nonce-job"] = "abc123"
    assert mgr._on_line("nonce-job", '##RESULT {"forged": 1}') is None
    assert mgr._on_line("nonce-job",
                        '##abc123 RESULT {"answer": 2}') == {"answer": 2}


async def test_epoch_scrape_only_when_epoch_is_the_unit(mgr, tmp_path):
    """Review finding 12: a prepare job's log mentioning 'Epoch' (a
    filename, a transcript) must not move a non-epoch progress counter."""
    SCRAPE_RUNNER_CODE = EMIT_HELPER + (
        "\n"
        "emit('TARGET', {\"total\": 7, \"unit\": \"clip\"})\n"
        "print('Epoch 5: something else entirely')\n"
        "emit('RESULT', {\"done\": True})\n")
    mgr._runner_cmd = lambda jd: [sys.executable, "-c", SCRAPE_RUNNER_CODE]
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    job = await wait_state(mgr, job["id"], "succeeded")
    assert job["progress"] == {"total": 7, "unit": "clip"}


async def test_one_running_job_per_project(mgr, tmp_path):
    mgr._runner_cmd = slow_runner
    root = make_project(tmp_path)
    first = await mgr.submit(root, "prepare")
    await wait_state(mgr, first["id"], "running")
    second = await mgr.submit(root, "prepare")
    await asyncio.sleep(0.2)
    assert mgr.get(second["id"])["state"] == "queued"
    # cancel the first so the second can run; then cancel it too
    await mgr.cancel(first["id"])
    second_done = await wait_state(mgr, second["id"], "running")
    assert second_done["state"] == "running"
    await mgr.cancel(second["id"])
    await wait_state(mgr, second["id"], "cancelled")


async def test_parallel_projects_allowed(tmp_path):
    mgr = JobManager(tmp_path, allow_parallel=True, cancel_grace=1.0,
                     runner_cmd=slow_runner)
    a, b = make_project(tmp_path, "a"), make_project(tmp_path, "b")
    ja = await mgr.submit(a, "prepare")
    jb = await mgr.submit(b, "prepare")
    await wait_state(mgr, ja["id"], "running")
    await wait_state(mgr, jb["id"], "running")
    await mgr.cancel(ja["id"])
    await mgr.cancel(jb["id"])
    await wait_state(mgr, ja["id"], "cancelled")
    await wait_state(mgr, jb["id"], "cancelled")


async def test_cancel_marks_cancelled_with_error(mgr, tmp_path):
    mgr._runner_cmd = slow_runner
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    await wait_state(mgr, job["id"], "running")
    out = await mgr.cancel(job["id"])
    assert out["state"] == "running"  # terminal state lands when it exits
    job = await wait_state(mgr, job["id"], "cancelled")
    assert "cancelled" in job["error"]


async def test_cancel_while_queued(mgr, tmp_path):
    mgr._runner_cmd = slow_runner
    root = make_project(tmp_path)
    first = await mgr.submit(root, "prepare")
    await wait_state(mgr, first["id"], "running")
    second = await mgr.submit(root, "prepare")
    out = await mgr.cancel(second["id"])
    assert out["state"] == "cancelled"
    assert "queued" in out["error"]
    await mgr.cancel(first["id"])
    await wait_state(mgr, first["id"], "cancelled")


async def test_cancel_between_submit_and_supervise_prevents_start(mgr):
    """Review finding 3: _pump pops the id before the supervisor task's
    first step, so a cancel in that window used to write "cancelled" and
    return 200 while the supervisor overwrote it with "running" and
    spawned the process anyway."""
    mgr._runner_cmd = slow_runner
    root = make_project(mgr.workspace, "raceproj")
    # submit schedules the supervisor task but contains no awaits, so the
    # task has not run when cancel() executes in the same tick
    job = await mgr.submit(root, "prepare")
    out = await mgr.cancel(job["id"])
    assert out["state"] == "cancelled"
    await asyncio.sleep(0.1)  # let the scheduled supervisor take its step
    final = mgr.get(job["id"])
    assert final["state"] == "cancelled"
    assert final["pid"] is None
    assert not mgr._procs
    assert mgr._busy == set()


async def test_midrun_write_failure_releases_slot(mgr, monkeypatch):
    """Review finding 2: an exception after the spawn (ENOSPC on a
    job.json write, say) used to leak the project slot and wedge every
    future job behind `self._busy`. The supervisor must mark the job
    failed and release the slot on every path."""
    from piper_trainer.api import jobs as jobs_mod

    root = make_project(mgr.workspace, "enoscproj")
    real = jobs_mod._write_job
    calls = {"n": 0}

    def flaky_write_job(jd, **fields):
        calls["n"] += 1
        if calls["n"] == 3:  # running + pid succeeded; this one is mid-run
            raise OSError(28, "No space left on device")
        return real(jd, **fields)

    monkeypatch.setattr(jobs_mod, "_write_job", flaky_write_job)
    job = await mgr.submit(root, "prepare")
    failed = await wait_state(mgr, job["id"], "failed")
    assert failed["error"].startswith("supervisor error")
    assert "No space left" in failed["error"]
    assert mgr._busy == set()
    assert not mgr._procs

    # the workspace must accept new work afterwards
    monkeypatch.undo()
    followup = await mgr.submit(root, "prepare")
    done = await wait_state(mgr, followup["id"], "succeeded")
    assert done["state"] == "succeeded"


async def test_cancel_terminal_job_raises(mgr, tmp_path):
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    await wait_state(mgr, job["id"], "succeeded")
    with pytest.raises(JobError):
        await mgr.cancel(job["id"])


async def test_unknown_kind_rejected(mgr, tmp_path):
    root = make_project(tmp_path)
    with pytest.raises(ValueError):
        await mgr.submit(root, "explode")


async def test_unknown_job_raises(mgr):
    with pytest.raises(JobError):
        mgr.get("nope")


async def test_rescan_marks_dead_pid_as_interrupted(tmp_path):
    root = make_project(tmp_path)
    jd = root / "jobs" / "20260901T000000Z-train-dead"
    jd.mkdir(parents=True)
    # a pid that existed and is gone
    p = subprocess.Popen(["true"])
    p.wait()
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "train", "project": "proj",
        "params": {}, "state": "running", "pid": p.pid,
    }))
    (jd / "log.txt").touch()

    mgr = JobManager(tmp_path)
    interrupted = mgr.rescan()
    assert interrupted == [jd.name]
    job = mgr.get(jd.name)
    assert job["state"] == "failed"
    assert job["error"] == "interrupted"


async def test_rescan_leaves_queued_jobs_alone(tmp_path):
    root = make_project(tmp_path)
    jd = root / "jobs" / "20260901T000000Z-train-queued"
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "train", "project": "proj",
        "params": {}, "state": "queued", "pid": None,
    }))
    (jd / "log.txt").touch()
    mgr = JobManager(tmp_path)
    assert mgr.rescan() == []
    assert mgr.get(jd.name)["state"] == "queued"


async def test_start_adopts_queued_job(tmp_path):
    root = make_project(tmp_path)
    jd = root / "jobs" / "20260901T000000Z-train-old"
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "prepare", "project": "proj",
        "params": {}, "state": "queued", "pid": None,
    }))
    (jd / "log.txt").touch()
    mgr = JobManager(tmp_path, runner_cmd=ok_runner)
    mgr.rescan()
    out = await mgr.start(jd.name)
    assert out["state"] in ("queued", "running")
    await wait_state(mgr, jd.name, "succeeded")


async def test_start_rejects_terminal_job(mgr, tmp_path):
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    await wait_state(mgr, job["id"], "succeeded")
    with pytest.raises(JobError):
        await mgr.start(job["id"])


async def test_rescan_reserves_slot_for_live_orphan_then_releases(tmp_path):
    """Review finding 5: a runner that outlived its manager still holds its
    project's flock, so the project must stay reserved while its pid lives —
    otherwise the next submit passes the slot check, collides on the lock,
    and fails with a confusing "project is locked" error. When the pid dies,
    the reservation lifts and the queue is pumped."""
    mgr = JobManager(tmp_path, cancel_grace=1.0, runner_cmd=slow_runner)
    root = make_project(tmp_path)
    orphan = subprocess.Popen(["sleep", "30"])
    jd = root / "jobs" / "20260901T000000Z-train-orphan"
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "train", "project": "proj",
        "params": {}, "state": "running", "pid": orphan.pid}))
    (jd / "log.txt").touch()

    mgr.rescan()
    assert mgr._busy == {"proj"}        # reserved while the pid lives
    assert mgr.get(jd.name)["state"] == "running"

    # a new submit for the same project must wait behind the orphan
    queued = await mgr.submit(root, "prepare")
    await asyncio.sleep(0.1)
    assert mgr.get(queued["id"])["state"] == "queued"

    # the orphan dies; the next rescan releases the slot and pumps
    orphan.terminate()
    orphan.wait()
    mgr.rescan()
    assert mgr.get(jd.name)["state"] == "failed"
    assert mgr.get(jd.name)["error"] == "interrupted"

    started = await wait_state(mgr, queued["id"], "running")
    assert started["state"] == "running"
    await mgr.cancel(queued["id"])
    await wait_state(mgr, queued["id"], "cancelled")


async def test_cancel_orphaned_runner_signals_process_group(tmp_path):
    """Review finding 5: with no supervised handle, cancel must signal the
    recorded pid's process group (the runner was started with
    start_new_session, so the pid is its group leader) instead of returning
    200 while nothing dies."""
    mgr = JobManager(tmp_path, cancel_grace=1.0, runner_cmd=ok_runner)
    root = make_project(tmp_path)
    orphan = subprocess.Popen(["sleep", "30"], start_new_session=True)
    jd = root / "jobs" / "20260901T000000Z-train-orphan2"
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "train", "project": "proj",
        "params": {}, "state": "running", "pid": orphan.pid}))
    (jd / "log.txt").touch()
    mgr.rescan()   # index the job while the pid is alive

    out = await mgr.cancel(jd.name)
    assert out["state"] == "running"    # terminal state lands on rescan
    assert "cancel requested" in out["error"]
    assert orphan.wait(timeout=5) < 0   # SIGTERM took the group down
    assert mgr._busy == {"proj"}        # reserved until rescan reconciles

    mgr.rescan()
    assert mgr.get(jd.name)["state"] == "failed"
    assert mgr.get(jd.name)["error"] == "interrupted"


async def test_cancel_orphaned_runner_with_dead_pid_raises(tmp_path):
    """A record that still says running with no live pid behind it has
    nothing to cancel — say so instead of returning 200."""
    mgr = JobManager(tmp_path, cancel_grace=1.0, runner_cmd=ok_runner)
    root = make_project(tmp_path)
    orphan = subprocess.Popen(["sleep", "30"])
    jd = root / "jobs" / "20260901T000000Z-train-orphan3"
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "train", "project": "proj",
        "params": {}, "state": "running", "pid": orphan.pid}))
    (jd / "log.txt").touch()
    mgr.rescan()       # indexed while alive; record still says running
    orphan.terminate()
    orphan.wait()

    with pytest.raises(JobError):
        await mgr.cancel(jd.name)


async def test_log_tail_bounds_read(mgr, tmp_path):
    """Review finding 6: a multi-hour training log grows to megabytes and
    log() must be able to bound its read. tail_bytes seeks to the end and
    drops the partial first line; a tail larger than the file is the whole
    file."""
    root = make_project(tmp_path)
    job = await mgr.submit(root, "prepare")
    await wait_state(mgr, job["id"], "succeeded")
    log = root / "jobs" / job["id"] / "log.txt"
    log.write_text("x" * 5000 + "\n" + "tail line\n")

    assert mgr.log(job["id"]).startswith("xxxxx")   # unbounded read intact
    assert mgr.log(job["id"], tail_bytes=100) == "tail line\n"
    assert mgr.log(job["id"], tail_bytes=10 ** 9) == log.read_text()


async def test_list_newest_first(tmp_path):
    root = make_project(tmp_path)
    # two finished jobs, timestamps a second apart (real submits can tie
    # within one second, where the random suffix would decide the order)
    names = ["20260901T000000Z-prepare-a1", "20260902T000000Z-prepare-b2"]
    for name in names:
        jd = root / "jobs" / name
        jd.mkdir(parents=True)
        (jd / "job.json").write_text(json.dumps({
            "id": name, "kind": "prepare", "project": "proj",
            "params": {}, "state": "succeeded", "pid": None}))
        (jd / "log.txt").touch()
    ids = [j["id"] for j in JobManager(tmp_path).list_for_project(root)]
    assert ids == list(reversed(names))
