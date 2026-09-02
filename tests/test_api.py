"""Bones API surface: system, projects, files, ingest, jobs, catalog."""
from __future__ import annotations

import asyncio
import json
import sys

import pytest
from fastapi.testclient import TestClient

from piper_trainer import doctor, prepare
from piper_trainer.api import catalog
from piper_trainer.api.app import create_app

OK_RUNNER = ("print('job ran')\n"
             "print('##RESULT {\"done\": true}')\n")
SLOW_RUNNER = ("import time\n"
               "print('starting', flush=True)\n"
               "time.sleep(60)\n")


def stub_runner(code):
    return lambda jd: [sys.executable, "-c", code]


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient over a temp workspace; fast stub runner for real jobs."""
    app = create_app(tmp_path, runner_cmd=stub_runner(OK_RUNNER))
    with TestClient(app) as c:
        yield c


def make_running_job(tmp_path, state="running", pid=None, kind="train"):
    """A job on disk written by hand, as an earlier process would have."""
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    jd = root / "jobs" / f"20260901T000000Z-{kind}-abc1"
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": kind, "project": "proj", "params": {},
        "state": state, "pid": pid,
    }))
    (jd / "log.txt").write_text("old log line\n")
    return jd


# ------------------------------------------------------------------ system

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["version"]


def test_tiers(client):
    r = client.get("/api/tiers")
    assert r.status_code == 200
    assert set(r.json()) == {"low", "medium", "high"}
    assert r.json()["medium"]["sample_rate"] == 22050


def test_doctor_structured(client, monkeypatch):
    monkeypatch.setattr(doctor, "check",
                        lambda: (["✓ ffmpeg on PATH",
                                  "✗ deep-filter on PATH",
                                  "· backend: CUDA 12"], False))
    r = client.get("/api/doctor")
    body = r.json()
    assert body["ok"] is False
    assert [c["status"] for c in body["checks"]] == ["ok", "error", "info"]
    assert "cpu" in body["transcribe_devices"]


def test_espeak_voices(client, monkeypatch):
    monkeypatch.setattr(doctor, "espeak_voices", lambda prefix="": ["en-us"])
    assert client.get("/api/espeak-voices").json() == ["en-us"]
    assert client.get("/api/espeak-voices?prefix=en").json() == ["en-us"]

    def boom(prefix=""):
        raise FileNotFoundError("espeak-ng")
    monkeypatch.setattr(doctor, "espeak_voices", boom)
    assert client.get("/api/espeak-voices").json() == []


# ---------------------------------------------------------------- projects

def test_project_crud_roundtrip(client):
    r = client.post("/api/projects", json={"name": "hal 9000",
                                           "espeak_voice": "en-us"})
    assert r.status_code == 201
    assert r.json()["name"] == "hal_9000"  # sanitized

    assert client.post("/api/projects", json={"name": "hal_9000"}).status_code \
        == 409
    assert client.post("/api/projects", json={"name": "///"}).status_code == 400

    names = [p["name"] for p in client.get("/api/projects").json()]
    assert names == ["hal_9000"]

    detail = client.get("/api/projects/hal_9000").json()
    assert detail["config"]["espeak_voice"] == "en-us"
    assert detail["directories"]["raw"] == 0
    assert detail["dataset"]["rows"] == 0

    r = client.delete("/api/projects/hal_9000")
    assert r.status_code == 200
    assert "moved_to" in r.json()
    assert client.get("/api/projects/hal_9000").status_code == 404
    assert client.get("/api/projects").json() == []


def test_project_id_validation(client, tmp_path):
    (tmp_path / "evil").mkdir()
    # path traversal via the id is refused before any filesystem touch
    assert client.get("/api/projects/../secret").status_code in (400, 404)


def test_project_file_serving_scoped(client):
    client.post("/api/projects", json={"name": "p1"})
    wav = client.app.state.workspace / "p1" / "dataset" / "wavs"
    wav.mkdir(parents=True, exist_ok=True)
    (wav / "a.wav").write_bytes(b"RIFF....")

    r = client.get("/api/projects/p1/files/dataset/wavs/a.wav")
    assert r.status_code == 200
    assert r.content == b"RIFF...."

    # traversal is refused (percent-encoded so httpx cannot normalize the
    # dot segments away before the server sees them)
    assert client.get(
        "/api/projects/p1/files/%2e%2e/%2e%2e/p1/project.json"
    ).status_code == 400
    # non-audio is refused
    assert client.get(
        "/api/projects/p1/files/project.json").status_code == 404


def test_sources_endpoint(client, monkeypatch):
    client.post("/api/projects", json={"name": "p1"})
    raw = client.app.state.workspace / "p1" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "take01.wav").write_bytes(b"fake")
    monkeypatch.setattr(prepare, "probe", lambda p: {
        "codec_name": "pcm_s16le", "sample_rate": "44100",
        "channels": "2", "duration": "12.5"})
    rows = client.get("/api/projects/p1/sources").json()
    assert len(rows) == 1
    assert rows[0]["codec"] == "pcm_s16le"


def test_peaks_endpoint(client, monkeypatch):
    from piper_trainer import peaks
    client.post("/api/projects", json={"name": "p1"})
    raw = client.app.state.workspace / "p1" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "take01.wav").write_bytes(b"fake")
    seen = {}

    def fake_peaks(src, channel="downmix", buckets=2000):
        seen.update(src=src.name, channel=channel, buckets=buckets)
        return {"name": src.name, "channel": channel, "buckets": 3,
                "rate": 8000, "duration": 12.5, "peaks": [0.1, 0.5, 0.9]}

    monkeypatch.setattr(peaks, "compute_peaks", fake_peaks)
    out = client.get("/api/projects/p1/sources/take01.wav/peaks",
                     params={"channel": "left", "buckets": 100}).json()
    assert seen == {"src": "take01.wav", "channel": "left", "buckets": 100}
    assert out["peaks"] == [0.1, 0.5, 0.9]
    # traversal-safe: only a basename reaches raw/
    r = client.get("/api/projects/p1/sources/%2e%2e%2fproject.json/peaks")
    assert r.status_code == 404
    r = client.get("/api/projects/p1/sources/take01.wav/peaks",
                   params={"channel": "sideways"})
    assert r.status_code == 400


# ------------------------------------------------------------------ ingest

def test_upload_ingest_creates_job_and_stages(client, tmp_path):
    client.post("/api/projects", json={"name": "p1"})
    r = client.post("/api/projects/p1/ingest",
                    files=[("files", ("my take.wav", b"RIFF", "audio/wav")),
                           ("files", ("b.flac", b"fLaC", "audio/flac"))])
    assert r.status_code == 202
    job = r.json()
    assert job["kind"] == "ingest"
    # the stub runner finishes the job but does not consume staged files;
    # verify they landed in the job's incoming/ dir
    import time
    deadline = time.time() + 5
    while time.time() < deadline:
        job_now = client.get(f"/api/jobs/{job['id']}").json()
        if job_now["state"] == "succeeded":
            break
        time.sleep(0.05)
    assert job_now["state"] == "succeeded"
    incoming = (tmp_path / "p1" / "jobs" / job["id"] / "incoming")
    assert sorted(p.name for p in incoming.iterdir()) == \
        ["b.flac", "my take.wav"]


# -------------------------------------------------------------------- jobs

def test_job_lifecycle_via_api(client, tmp_path):
    client.post("/api/projects", json={"name": "p1"})
    r = client.post("/api/projects/p1/jobs",
                    json={"kind": "prepare", "params": {"tier": "medium"}})
    assert r.status_code == 202
    job_id = r.json()["id"]

    import time
    deadline = time.time() + 5
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] == "succeeded":
            break
        time.sleep(0.05)
    assert job["state"] == "succeeded"
    assert job["result"] == {"done": True}

    log = client.get(f"/api/jobs/{job_id}/log")
    assert log.status_code == 200
    assert "job ran" in log.text

    assert client.get("/api/jobs/nope").status_code == 404
    bad = client.post("/api/projects/p1/jobs", json={"kind": "nope"})
    assert bad.status_code == 400


def test_job_list_newest_first(client):
    client.post("/api/projects", json={"name": "p1"})
    r1 = client.post("/api/projects/p1/jobs", json={"kind": "prepare"}).json()
    r2 = client.post("/api/projects/p1/jobs", json={"kind": "prepare"}).json()
    import time
    deadline = time.time() + 5
    while time.time() < deadline:
        states = [client.get(f"/api/jobs/{i}").json()["state"]
                  for i in (r1["id"], r2["id"])]
        if all(s == "succeeded" for s in states):
            break
        time.sleep(0.05)
    listed = client.get("/api/projects/p1/jobs").json()
    ids = [j["id"] for j in listed]
    # newest first by id; both submits may land in the same second, where
    # the random suffix decides ties — so assert membership + sort order
    # (exact recency ordering is covered by test_jobs.py::test_list_newest_first)
    assert set(ids[:2]) == {r1["id"], r2["id"]}
    assert ids == sorted(ids, reverse=True)


def test_cancel_via_api(client):
    app = create_app(client.app.state.workspace,
                     runner_cmd=stub_runner(SLOW_RUNNER))
    with TestClient(app) as c:
        c.post("/api/projects", json={"name": "p1"})
        job = c.post("/api/projects/p1/jobs",
                     json={"kind": "prepare"}).json()
        import time
        deadline = time.time() + 5
        while time.time() < deadline:
            if c.get(f"/api/jobs/{job['id']}").json()["state"] == "running":
                break
            time.sleep(0.05)
        r = c.post(f"/api/jobs/{job['id']}/cancel")
        assert r.status_code == 200
        deadline = time.time() + 10
        while time.time() < deadline:
            state = c.get(f"/api/jobs/{job['id']}").json()["state"]
            if state == "cancelled":
                break
            time.sleep(0.05)
        assert state == "cancelled"
        # cancelling again is a 409: nothing left to cancel
        assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 409


def test_start_adopts_stale_queued_job(client, tmp_path):
    jd = make_running_job(tmp_path, state="queued")
    r = client.post(f"/api/jobs/{jd.name}/start")
    assert r.status_code == 200
    import time
    deadline = time.time() + 5
    while time.time() < deadline:
        state = client.get(f"/api/jobs/{jd.name}").json()["state"]
        if state == "succeeded":
            break
        time.sleep(0.05)
    assert state == "succeeded"


def test_websocket_stream(client, tmp_path):
    jd = make_running_job(tmp_path, state="succeeded")
    with client.websocket_connect(
            f"/api/jobs/{jd.name}/stream") as ws:
        state = ws.receive_json()
        assert state["type"] == "state"
        assert state["job"]["state"] == "succeeded"
        reset = ws.receive_json()
        assert reset["type"] == "log_reset"
        assert "old log line" in reset["text"]
