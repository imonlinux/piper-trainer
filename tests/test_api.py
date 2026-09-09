"""Bones API surface: system, projects, files, ingest, jobs, catalog."""
from __future__ import annotations

import asyncio
import io
import json
import subprocess
import sys
import time
import types

import pytest
from fastapi.testclient import TestClient

from piper_trainer import doctor, prepare
from piper_trainer.api import app as app_mod
from piper_trainer.api import catalog
from piper_trainer.api.app import create_app

OK_RUNNER = ("import os, json\n"
             "def emit(tag, obj):\n"
             "    n = os.environ.get('PIPER_DIRECTIVE_NONCE', '')\n"
             "    print(f'##{n} {tag} {json.dumps(obj)}', flush=True)\n"
             "print('job ran')\n"
             "emit('RESULT', {\"done\": True})\n")
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
    # piper's bundled data wins when importable; system espeak-ng is the
    # flagged fallback; no list at all degrades to source "none".
    monkeypatch.setattr(doctor, "piper_espeak_voices",
                        lambda: ["de", "en-gb-x-rp", "en-us"])
    body = client.get("/api/espeak-voices?prefix=en").json()
    assert body == {"source": "piper",
                    "voices": ["en-gb-x-rp", "en-us"]}

    monkeypatch.setattr(doctor, "piper_espeak_voices", lambda: None)
    monkeypatch.setattr(doctor, "espeak_voices",
                        lambda prefix="": ["en-us", "en-gb"])
    body = client.get("/api/espeak-voices?prefix=en-gb").json()
    assert body == {"source": "system", "voices": ["en-gb"]}

    def boom(prefix=""):
        raise FileNotFoundError("espeak-ng")
    monkeypatch.setattr(doctor, "espeak_voices", boom)
    assert client.get("/api/espeak-voices").json() == {"source": "none",
                                                       "voices": []}


def test_piper_espeak_voices_parses_language_directive(tmp_path, monkeypatch):
    # Real en-US voice file: "language en-us 2" — identifier + priority.
    # The priority is not part of the name; keeping it once made the
    # selector offer and default to "en-us 2", which phonemize rejects.
    data = tmp_path / "espeak-ng-data"
    lang = data / "lang" / "gmw"
    lang.mkdir(parents=True)
    (lang / "en-US").write_text(
        "name English (America)\nlanguage en-us 2\nlanguage en 3\n")
    (lang / "en-gb-x-rp").write_text("language en-gb-x-rp 5\n")
    pkg = types.ModuleType("piper")
    sub = types.ModuleType("piper.phonemize_espeak")
    sub.ESPEAK_DATA_DIR = str(data)
    pkg.phonemize_espeak = sub
    monkeypatch.setitem(sys.modules, "piper", pkg)
    monkeypatch.setitem(sys.modules, "piper.phonemize_espeak", sub)
    assert doctor.piper_espeak_voices() == ["en-gb-x-rp", "en-us"]


def test_train_gates_espeak_voice_at_submit(client, monkeypatch):
    # A voice the bundled data does not know must fail at submit with a
    # clear message, not minutes into the run at phonemize time.
    monkeypatch.setattr(doctor, "piper_espeak_voices",
                        lambda: ["en-us", "en-gb-x-rp"])
    assert client.post(
        "/api/projects", json={"name": "hal", "espeak_voice": "en-us 2"}
    ).status_code == 201
    r = client.post("/api/projects/hal/jobs", json={"kind": "train"})
    assert r.status_code == 400
    assert "en-us 2" in r.json()["detail"]
    # the train preview runs the same gate
    r = client.post("/api/projects/hal/preview", json={"stage": "train"})
    assert r.status_code == 400
    # an explicit param overrides the saved value and passes
    r = client.post("/api/projects/hal/jobs",
                    json={"kind": "train", "params": {"espeak_voice": "en-us"}})
    assert r.status_code == 202
    # a non-train preview does not phonemize, so it is not gated
    r = client.post("/api/projects/hal/preview", json={"stage": "segment"})
    assert r.status_code == 202
    # no bundled data -> nothing to check against, gate stays open
    monkeypatch.setattr(doctor, "piper_espeak_voices", lambda: None)
    r = client.post("/api/projects/hal/jobs", json={"kind": "train"})
    assert r.status_code == 202


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
    assert detail["definition"]["espeak_voice"] == "en-us"
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


def test_definition_patch(client, tmp_path):
    client.post("/api/projects", json={"name": "hal", "espeak_voice": "en-us"})

    # set: new keys and type changes, including an existing key fix
    r = client.patch("/api/projects/hal/definition",
                     json={"updates": {"espeak_voice": "en-gb-x-rp",
                                       "target_epochs": 1200,
                                       "transcripts_provided": True}})
    assert r.status_code == 200
    assert r.json()["definition"]["espeak_voice"] == "en-gb-x-rp"
    assert r.json()["definition"]["target_epochs"] == 1200
    assert r.json()["definition"]["transcripts_provided"] is True
    # written through to project.json, not just echoed
    on_disk = json.loads((tmp_path / "hal" / "project.json").read_text())
    assert on_disk["espeak_voice"] == "en-gb-x-rp"

    # null removes; removing a missing key is a no-op, not an error
    r = client.patch("/api/projects/hal/definition",
                     json={"updates": {"target_epochs": None,
                                       "never_there": None}})
    assert r.status_code == 200
    assert "target_epochs" not in r.json()["definition"]

    # name is the directory's identity: refused
    assert client.patch("/api/projects/hal/definition",
                        json={"updates": {"name": "marvin"}}).status_code == 400
    assert client.patch("/api/projects/hal/definition",
                        json={"updates": {"name": "hal"}}).status_code == 200

    # key shape: anything that could confuse readers/writers is refused
    for bad in ("a b", "", "1x", "x/y"):
        assert client.patch("/api/projects/hal/definition",
                            json={"updates": {bad: 1}}).status_code == 400

    # a running job pins the file: 409, nothing written
    jd = tmp_path / "hal" / "jobs" / "20260901T000000Z-train-abc9"
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "train", "project": "hal", "params": {},
        "state": "running", "pid": None,
    }))
    r = client.patch("/api/projects/hal/definition",
                     json={"updates": {"espeak_voice": "en-us"}})
    assert r.status_code == 409
    on_disk = json.loads((tmp_path / "hal" / "project.json").read_text())
    assert on_disk["espeak_voice"] == "en-gb-x-rp"


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
    # PLAYABLE_EXT: matroska is processed but not served (no browser
    # <audio> support); opus is both processable and playable
    (wav / "a.mkv").write_bytes(b"MKV")
    (wav / "a.opus").write_bytes(b"OPUS")
    assert client.get(
        "/api/projects/p1/files/dataset/wavs/a.mkv").status_code == 404
    assert client.get(
        "/api/projects/p1/files/dataset/wavs/a.opus").status_code == 200


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

def test_ingest_rejects_non_audio_extensions(client):
    """Review finding 9: an unchecked upload landed in raw/, vanished from
    sources() and still inflated the directory card."""
    client.post("/api/projects", json={"name": "p1"})
    r = client.post("/api/projects/p1/ingest",
                    files=[("files", ("notes.txt", b"hello", "text/plain"))])
    assert r.status_code == 400
    assert "notes.txt" in r.json()["detail"]


def test_raw_count_ignores_non_audio_files(client, tmp_path):
    client.post("/api/projects", json={"name": "p1"})
    raw = client.app.state.workspace / "p1" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "take.wav").write_bytes(b"RIFF")
    (raw / "notes.txt").write_text("stray")
    detail = client.get("/api/projects/p1").json()
    assert detail["directories"]["raw"] == 1


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


PROBE_RUNNER = (
    "import sys, os, json\n"
    "from pathlib import Path\n"
    "jd = Path(sys.argv[1])\n"
    "inc = jd / 'incoming'\n"
    "seen = ({p.name: p.stat().st_size for p in inc.iterdir()}\n"
    "        if inc.exists() else None)\n"
    "(jd / 'observed.json').write_text(json.dumps(seen))\n"
    "n = os.environ.get('PIPER_DIRECTIVE_NONCE', '')\n"
    "print(f'##{n} RESULT {json.dumps(dict(probed=True))}', flush=True)\n")


def test_ingest_stages_every_byte_before_runner_starts(tmp_path):
    """Review finding 1: the runner used to start while the upload loop
    was still filling incoming/, so any file bigger than one 1 MiB read
    chunk was moved into raw/ truncated. The runner must see every staged
    byte on its first line of life."""
    def probe_cmd(jd):
        return [sys.executable, "-c", PROBE_RUNNER, str(jd)]

    app = create_app(tmp_path, runner_cmd=probe_cmd)
    big = b"\0" * (20 * 1024 * 1024)
    with TestClient(app) as c:
        c.post("/api/projects", json={"name": "p1"})
        r = c.post("/api/projects/p1/ingest", files=[
            ("files", ("a.wav", io.BytesIO(big), "audio/wav")),
            ("files", ("b.wav", io.BytesIO(big), "audio/wav")),
        ])
        assert r.status_code == 202
        job_id = r.json()["id"]
        import time
        deadline = time.time() + 10
        while time.time() < deadline:
            state = c.get(f"/api/jobs/{job_id}").json()["state"]
            if state in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.05)
        assert state == "succeeded"
        jd = tmp_path / "p1" / "jobs" / job_id
        seen = json.loads((jd / "observed.json").read_text())
        assert seen == {"a.wav": len(big), "b.wav": len(big)}


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


# --------------------------------------------------------------- previews

def write_preview(root, stage, pid, params=None):
    pdir = root / "work" / "preview" / stage / pid
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "preview.json").write_text(json.dumps({
        "id": pid, "stage": stage, "params": params or {},
        "created_at": "2026-09-02T00:00:00Z",
        "result": {"clip_count": 5, "audio": ["a.wav"]}}))
    return pdir


def test_preview_endpoints(client, tmp_path):
    client.post("/api/projects", json={"name": "p1"})
    root = tmp_path / "p1"
    write_preview(root, "segment", "20260901T000000Z-preview-aaaa",
                  {"source": "take.wav", "energy_threshold": 40})
    write_preview(root, "denoise", "20260831T000000Z-preview-bbbb")

    rows = client.get("/api/projects/p1/previews").json()
    assert [r["id"] for r in rows] == ["20260901T000000Z-preview-aaaa",
                                       "20260831T000000Z-preview-bbbb"]
    assert rows[0]["dir"] == "work/preview/segment/20260901T000000Z-preview-aaaa"

    # promote replays the preview's params as a full prepare run
    r = client.post(
        "/api/projects/p1/previews/20260901T000000Z-preview-aaaa/promote")
    assert r.status_code == 202
    job = r.json()
    assert job["kind"] == "prepare"
    assert job["params"] == {"source": "take.wav", "energy_threshold": 40}
    # ...and saves them, so a plain "run prepare" replays the winner
    assert json.loads((root / "project.json").read_text())["prepare_params"] == {
        "source": "take.wav", "energy_threshold": 40}

    assert client.post("/api/projects/p1/previews/nope/promote").status_code == 404
    # a routable id that fails the NAME_RE guard (no path tricks needed)
    assert client.post(
        "/api/projects/p1/previews/bad%20id/promote").status_code == 400

    # prune clears the whole scratch tree (§2.1: previews are discardable)
    assert client.delete("/api/projects/p1/previews").json() == {"pruned": True}
    assert client.get("/api/projects/p1/previews").json() == []


def test_run_prepare_replays_promoted_params(client, tmp_path):
    """The project page's "run prepare" posts empty params; it must start
    from the tuner's promoted dials, not silently fall back to defaults."""
    client.post("/api/projects", json={"name": "p1"})
    root = tmp_path / "p1"
    write_preview(root, "segment", "20260901T000000Z-preview-aaaa",
                  {"source": "take.wav", "energy_threshold": 41,
                   "min_dur": 1.0})
    client.post(
        "/api/projects/p1/previews/20260901T000000Z-preview-aaaa/promote")

    job = client.post("/api/projects/p1/jobs",
                      json={"kind": "prepare", "params": {}}).json()
    assert job["params"] == {"source": "take.wav", "energy_threshold": 41,
                             "min_dur": 1.0}

    # explicit params win over the saved ones, key by key
    job = client.post(
        "/api/projects/p1/jobs",
        json={"kind": "prepare", "params": {"energy_threshold": 30}},
    ).json()
    assert job["params"] == {"source": "take.wav", "energy_threshold": 30,
                             "min_dur": 1.0}

    # no promote yet -> plain defaults, unchanged behavior
    client.post("/api/projects", json={"name": "p2"})
    job = client.post("/api/projects/p2/jobs",
                      json={"kind": "prepare", "params": {}}).json()
    assert job["params"] == {}


def test_preview_submit_rejects_unknown_stage(client):
    client.post("/api/projects", json={"name": "p1"})
    r = client.post("/api/projects/p1/preview",
                    json={"stage": "nope", "params": {}})
    assert r.status_code == 202  # kind is valid; the stage fails at run time
    job = client.get(f"/api/jobs/{r.json()['id']}").json()
    assert job["kind"] == "preview"
    assert job["params"]["stage"] == "nope"


def test_periodic_rescan_releases_dead_orphan(tmp_path, monkeypatch):
    """Review finding 1.2 (layer 2): startup rescan only catches orphans
    that died before the server came up. The lifespan rescan loop must
    reap a runner whose pid dies LATER — flipping the stale "running"
    record and releasing the project reservation a queued job waits on."""
    monkeypatch.setattr(app_mod, "RESCAN_INTERVAL", 0.05)
    app = create_app(tmp_path)

    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    orphan = subprocess.Popen(["sleep", "30"])
    jd = root / "jobs" / "20260901T000000Z-train-orphan"
    jd.mkdir(parents=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "train", "project": "proj", "params": {},
        "state": "running", "pid": orphan.pid}))
    (jd / "log.txt").touch()

    try:
        with TestClient(app) as client:
            # reservation held while the pid lives
            assert client.get(f"/api/jobs/{jd.name}").json()["state"] == "running"

            orphan.terminate()
            orphan.wait()
            # no explicit rescan call here — the loop must do the work
            deadline = time.time() + 5
            job = {"state": "running"}
            while time.time() < deadline:
                job = client.get(f"/api/jobs/{jd.name}").json()
                if job["state"] == "failed":
                    break
                time.sleep(0.05)
            assert job["state"] == "failed"
            assert job["error"] == "interrupted"
    finally:
        if orphan.poll() is None:
            orphan.kill()
            orphan.wait()
