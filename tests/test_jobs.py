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

OK_RUNNER_CODE = """
print('hello from stub')
print('##TARGET {"total": 10, "unit": "epoch"}')
print('Epoch 3:  30%|###|')
print('##RESULT {"answer": 42}')
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

RESULT_ERROR_RUNNER_CODE = """
print('##RESULT {"error": "validation blew up"}')
import sys; sys.exit(1)
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
    assert job["progress"] == {"total": 10, "unit": "epoch", "current": 3}


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
