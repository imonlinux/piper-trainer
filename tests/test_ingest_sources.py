"""Ingest source types (§2.5.2–2.5.4): url, media-site and hf-dataset
acquisition share the upload tail, so the pure helpers (command building,
content-type gate, filename derivation, metadata columns) are tested
without network, and in-process runs prove the runner wires fetch →
land → probe → metadata merge end to end. The train/preview endpoint
(§6.4) is exercised for its steps math, its 400s and its honest-basis
rule: no succeeded train job, no projected seconds."""
from __future__ import annotations

import json
import unittest.mock as mock
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from piper_trainer import ingest as ingest_mod
from piper_trainer import prepare
from piper_trainer.api import runner
from piper_trainer.api.app import create_app
from piper_trainer.api.jobs import _write_job


def write_wav(path, seconds=1.0, rate=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def make_job(tmp_path, kind, params=None):
    """A queued job on disk, exactly as the manager writes it."""
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    (root / "dataset").mkdir(exist_ok=True)
    job_dir = root / "jobs" / f"20260901T000000Z-{kind}-test"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps({
        "id": job_dir.name, "kind": kind, "project": "proj",
        "params": params or {}, "state": "queued", "pid": None,
    }))
    (job_dir / "log.txt").touch()
    return job_dir


PROBE = {"codec_name": "pcm_s16le", "sample_rate": "22050",
         "channels": "1", "duration": "1.0"}


# ------------------------------------------------------------ pure helpers

def test_content_type_gate():
    assert ingest_mod.content_type_ok("audio/mpeg")
    assert ingest_mod.content_type_ok("video/mp4")
    assert ingest_mod.content_type_ok("application/ogg")
    assert ingest_mod.content_type_ok("application/octet-stream")
    assert not ingest_mod.content_type_ok("text/html")
    assert not ingest_mod.content_type_ok("application/json")


def test_media_site_cmd_basics():
    dest = Path("/tmp/x")
    cmd = ingest_mod.media_site_cmd("https://v.example/watch/1", dest)
    assert cmd[0] == "yt-dlp"
    assert "-x" in cmd and cmd[cmd.index("--audio-format") + 1] == "wav"
    assert "--no-playlist" in cmd
    # -- ends option parsing, url is the only thing after it
    assert cmd[-2:] == ["--", "https://v.example/watch/1"]
    # playlist is opt-in; sections pass through verbatim
    pl = ingest_mod.media_site_cmd("https://v.example/watch/1", dest,
                                   playlist=True)
    assert "--no-playlist" not in pl
    sec = ingest_mod.media_site_cmd("https://v.example/watch/1", dest,
                                    sections="*:10-20")
    assert sec[sec.index("--download-sections") + 1] == "*:10-20"


def test_media_site_cmd_refuses_bad_urls():
    dest = Path("/tmp/x")
    # http/https only — no file:, no bare junk
    for url in ("u", "file:///etc/passwd", "", "ftp://h/x"):
        with pytest.raises(RuntimeError, match="unsupported url scheme"):
            ingest_mod.media_site_cmd(url, dest)
    # an option-looking URL must land after -- so yt-dlp reads it as text
    sneaky = ingest_mod.media_site_cmd("https://v.example/-o+--exec=touch", dest)
    assert sneaky[sneaky.index("--") + 1] == "https://v.example/-o+--exec=touch"


def test_url_filename_precedence():
    # Content-Disposition wins over the URL basename
    name = ingest_mod.url_filename(
        "https://h.example/a/clip3.bin",
        'attachment; filename="hal line 1.wav"', "audio/wav")
    assert name == "hal_line_1.wav"  # stored stems are sanitized
    # then the URL basename
    assert ingest_mod.url_filename("https://h.example/a/line.mp3", None,
                                   "audio/mpeg") == "line.mp3"
    # then a guess from the content type when the name has no audio ext
    assert ingest_mod.url_filename("https://h.example/a/track1", None,
                                   "audio/x-wav") == "track1.wav"


def test_url_filename_refuses_undeterminable():
    with pytest.raises(RuntimeError, match="audio extension"):
        ingest_mod.url_filename("https://h.example/a/page", None, "text/html")


def test_parse_metadata_columns(tmp_path):
    p = tmp_path / "metadata.csv"
    p.write_text("file_name,text\na.wav,hello world\nb.wav,second line\n")
    assert ingest_mod._parse_metadata(p) == [("a", "hello world"),
                                             ("b", "second line")]
    # unknown columns -> None, so the caller tries the next candidate
    p2 = tmp_path / "other.csv"
    p2.write_text("id,utterance\n1,hi\n")
    assert ingest_mod._parse_metadata(p2) is None


# ------------------------------------------------- runner wiring, url

def test_ingest_url_end_to_end(tmp_path, monkeypatch):
    """url source: fetch (mocked to land bytes) then the SAME tail as
    upload — sanitize into raw/, probe, identical job summary."""
    monkeypatch.setattr(prepare, "probe", lambda p: PROBE)
    jd = make_job(tmp_path, "ingest",
                  {"source_type": "url", "url": "https://f.example/hal 01.wav"})

    def fake_fetch(url, dest, emit):
        assert url == "https://f.example/hal 01.wav"
        out = dest / "hal 01.wav"
        write_wav(out)
        return [out]

    monkeypatch.setattr(ingest_mod, "fetch_url", fake_fetch)
    result = runner.execute(jd)
    assert result["source_type"] == "url"
    assert result["added"] == 1
    assert (tmp_path / "proj" / "raw" / "hal_01.wav").is_file()


def test_ingest_unknown_source_type(tmp_path):
    jd = make_job(tmp_path, "ingest", {"source_type": "carrier-pigeon"})
    with pytest.raises(RuntimeError, match="unknown source_type"):
        runner.execute(jd)


# ------------------------------------------- runner wiring, hf-dataset

def test_ingest_hf_dataset_merges_transcripts(tmp_path, monkeypatch):
    """hf-dataset: audio lands sanitized; transcripts keyed on ORIGINAL
    filenames survive collision renames and append to metadata.csv;
    transcripts_provided flips on in project.json."""
    snap = tmp_path / "snap"
    (snap / "audio").mkdir(parents=True)
    write_wav(snap / "audio" / "line one.wav")
    write_wav(snap / "audio" / "line two.wav")
    (snap / "metadata.csv").write_text(
        "file_name,text\n"
        "line one.wav,good morning\n"
        "line two.wav,i am completely operational\n")
    monkeypatch.setattr(prepare, "probe", lambda p: PROBE)
    monkeypatch.setattr(ingest_mod, "_snapshot_download",
                        lambda repo_id: snap)
    jd = make_job(tmp_path, "ingest",
                  {"source_type": "hf-dataset", "repo_id": "owner/corpus"})

    result = runner.execute(jd)

    assert result["source_type"] == "hf-dataset"
    assert result["transcripts_written"] == 2
    raw = tmp_path / "proj" / "raw"
    assert (raw / "line_one.wav").is_file()
    assert (raw / "line_two.wav").is_file()
    meta = (tmp_path / "proj" / "dataset" / "metadata.csv").read_text()
    assert "line_one|good morning" in meta
    assert "line_two|i am completely operational" in meta
    proj = json.loads((tmp_path / "proj" / "project.json").read_text())
    assert proj["transcripts_provided"] is True


def test_ingest_hf_dataset_existing_rows_win(tmp_path, monkeypatch):
    """Re-ingesting must not duplicate or clobber edited transcripts."""
    snap = tmp_path / "snap"
    snap.mkdir()
    write_wav(snap / "a.wav")
    (snap / "meta.csv").write_text("file_name,text\na.wav,fresh text\n")
    monkeypatch.setattr(prepare, "probe", lambda p: PROBE)
    monkeypatch.setattr(ingest_mod, "_snapshot_download", lambda r: snap)
    root = tmp_path / "proj"
    (root / "dataset").mkdir(parents=True)
    (root / "dataset" / "metadata.csv").write_text("a|human edited text\n")
    jd = make_job(tmp_path, "ingest",
                  {"source_type": "hf-dataset", "repo_id": "o/c"})

    result = runner.execute(jd)

    assert result["transcripts_written"] == 0
    assert (root / "dataset" / "metadata.csv").read_text() == \
        "a|human edited text\n"


def test_ingest_hf_dataset_parquet_refused(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "data.parquet").write_bytes(b"parquet")
    monkeypatch.setattr(ingest_mod, "_snapshot_download", lambda r: snap)
    jd = make_job(tmp_path, "ingest",
                  {"source_type": "hf-dataset", "repo_id": "o/pq"})
    with pytest.raises(RuntimeError, match="parquet"):
        runner.execute(jd)


# ------------------------------------------------------------ train preview

def make_client(tmp_path):
    app = create_app(tmp_path)
    return TestClient(app)


def test_train_preview_math_and_400s(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    (root / "dataset").mkdir()
    with make_client(tmp_path) as client:
        r = client.get("/api/projects/proj/train/preview",
                       params={"epochs": 0})
        assert r.status_code == 400
        r = client.get("/api/projects/proj/train/preview",
                       params={"epochs": 10, "batch_size": 0})
        assert r.status_code == 400

        # 8 clips, batch 3 -> 3 steps/epoch; no train history yet ->
        # honest nulls, not invented numbers
        (root / "dataset" / "metadata.csv").write_text(
            "a.wav|one\nb.wav|two\nc.wav|three\nd.wav|four\n"
            "e.wav|five\nf.wav|six\ng.wav|seven\nh.wav|eight\n")
        r = client.get("/api/projects/proj/train/preview",
                       params={"epochs": 10, "batch_size": 3})
        assert r.status_code == 200
    body = r.json()
    assert body["clips"] == 8
    assert body["steps_per_epoch"] == 3
    assert body["total_steps"] == 30
    assert body["sample_rate"] is not None
    assert body["projected_seconds"] is None
    assert body["basis"] is None


def test_train_preview_projection_from_history(tmp_path):
    """A succeeded train job plus a checkpoint at epoch 4 is the only
    honest basis: seconds/epoch = wall/4, projected = that * epochs."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    (root / "dataset").mkdir()
    (root / "dataset" / "metadata.csv").write_text("a.wav|one\n")
    ck = root / "runs-medium" / "lightning_logs" / "version_0" / \
        "checkpoints" / "epoch=4.ckpt"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x")

    import asyncio

    import piper_trainer.train as train_mod
    with make_client(tmp_path) as client, \
         mock.patch.object(train_mod, "latest_checkpoint", lambda p, t: ck), \
         mock.patch.object(train_mod, "checkpoint_epoch", lambda path: 4):
        job = asyncio.run(client.app.state.manager.submit(
            root, "train", run=False))
        jd = client.app.state.manager.job_dir(job["id"])
        _write_job(jd, state="succeeded",
                   started_at="2026-09-01T10:00:00Z",
                   finished_at="2026-09-01T12:00:00Z")
        r = client.get("/api/projects/proj/train/preview",
                       params={"epochs": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["seconds_per_epoch"] == 1800.0  # 7200 s wall / 4 epochs
    assert body["projected_seconds"] == 180000
    assert "epoch 0 -> 4" in body["basis"]  # no start_epoch recorded: 0


def test_train_preview_projection_counts_resume_epochs(tmp_path):
    """A resumed run's wall clock covers only the epochs IT trained. A
    job whose start_epoch is 40 ending at epoch 44 ran 4 epochs, not 44 —
    dividing by the absolute counter under-projected 11x."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    (root / "dataset").mkdir()
    (root / "dataset" / "metadata.csv").write_text("a.wav|one\n")
    ck = root / "runs-medium" / "lightning_logs" / "version_1" / \
        "checkpoints" / "epoch=44.ckpt"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"x")

    import asyncio

    import piper_trainer.train as train_mod
    with make_client(tmp_path) as client, \
         mock.patch.object(train_mod, "latest_checkpoint", lambda p, t: ck), \
         mock.patch.object(train_mod, "checkpoint_epoch", lambda path: 44):
        job = asyncio.run(client.app.state.manager.submit(
            root, "train", run=False))
        jd = client.app.state.manager.job_dir(job["id"])
        _write_job(jd, state="succeeded",
                   started_at="2026-09-01T10:00:00Z",
                   finished_at="2026-09-01T12:00:00Z",
                   result={"tier": "medium", "max_epochs": 44,
                           "start_epoch": 40, "checkpoint": "x"})
        r = client.get("/api/projects/proj/train/preview",
                       params={"epochs": 100})
    assert r.status_code == 200
    body = r.json()
    assert body["seconds_per_epoch"] == 1800.0  # 7200 s wall / 4 trained
    assert body["projected_seconds"] == 180000
    assert "epoch 40 -> 44" in body["basis"]


# -------------------------------------------------------------- fetch caps

class _FakeResp:
    """The seam surface fetch_url uses: headers + read() + context mgr."""
    def __init__(self, ctype="audio/wav", chunks=(), length=None):
        self._ctype = ctype
        self._chunks = list(chunks)
        self._hdrs = {"Content-Disposition": None,
                      "Content-Length": None if length is None
                      else str(length)}
        self.headers = type("H", (), {
            "get_content_type": (lambda s, c=self._ctype: c),
            "get": (lambda s, k, d=None: self._hdrs.get(k, d)),
        })()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n):
        return self._chunks.pop(0) if self._chunks else b""


def test_fetch_url_rejects_oversized_content_length(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest_mod, "urllib",
                        mock.Mock(request=mock.Mock(
                            urlopen=lambda req, timeout: _FakeResp(length=4 << 30))))
    with pytest.raises(RuntimeError, match="fetch cap"):
        ingest_mod.fetch_url("https://h.example/big.wav", tmp_path, lambda *a: None)
    assert list(tmp_path.iterdir()) == []   # nothing staged


def test_fetch_url_enforces_stream_budget(tmp_path, monkeypatch):
    """A missing/wrong Content-Length cannot smuggle an endless stream
    past the cap: the budget applies to the bytes that actually land,
    and the partial file is removed on failure."""
    big = b"x" * ingest_mod.CHUNK
    resp = _FakeResp(chunks=[big] * 10)   # 10 MiB, cap patched to 5 MiB
    monkeypatch.setattr(ingest_mod, "urllib",
                        mock.Mock(request=mock.Mock(
                            urlopen=lambda req, timeout: resp)))
    monkeypatch.setattr(ingest_mod, "MAX_FETCH_BYTES", 5 << 20)
    with pytest.raises(RuntimeError, match="fetch cap"):
        ingest_mod.fetch_url("https://h.example/long.wav", tmp_path,
                             lambda *a: None)
    assert list(tmp_path.iterdir()) == []   # partial removed


def test_fetch_url_success_under_cap(tmp_path, monkeypatch):
    resp = _FakeResp(chunks=[b"RIFFdata"])
    monkeypatch.setattr(ingest_mod, "urllib",
                        mock.Mock(request=mock.Mock(
                            urlopen=lambda req, timeout: resp)))
    got = ingest_mod.fetch_url("https://h.example/clip.wav", tmp_path,
                               lambda *a: None)
    assert [p.name for p in got] == ["clip.wav"]
    assert got[0].read_bytes() == b"RIFFdata"


def test_media_site_cmd_carries_caps():
    cmd = ingest_mod.media_site_cmd("https://v.example/watch/1", Path("/tmp/x"))
    assert cmd[cmd.index("--max-filesize") + 1] == str(ingest_mod.MAX_FETCH_BYTES)
    assert cmd[cmd.index("--socket-timeout") + 1] == str(ingest_mod.FETCH_TIMEOUT)


def test_fetch_media_site_watchdog_kills_hang(tmp_path, monkeypatch):
    """A hung extractor must not hold the job open forever: the watchdog
    kills the process and the failure names the timeout."""
    import subprocess as sp
    import time as time_mod

    monkeypatch.setattr(ingest_mod, "MEDIA_SITE_TIMEOUT", 0.2)

    class _HangingProc:
        def __init__(self):
            self.proc = sp.Popen(["sleep", "30"], stdout=sp.PIPE,
                                 stderr=sp.STDOUT, text=True)
            self.stdout = self.proc.stdout
            self.terminated = False
            self.killed = False

        def terminate(self):
            self.terminated = True
            self.proc.kill()   # sleep ignores TERM-ish nuances; force it

        def wait(self, timeout=None):
            return self.proc.wait(timeout=timeout)

        def kill(self):
            self.killed = True
            self.proc.kill()

    hp = _HangingProc()
    monkeypatch.setattr(ingest_mod.subprocess, "Popen",
                        lambda *a, **k: hp)
    start = time_mod.monotonic()
    with pytest.raises(RuntimeError, match="killed"):
        ingest_mod.fetch_media_site("https://v.example/1", tmp_path)
    assert time_mod.monotonic() - start < 10
    hp.proc.kill()
    hp.proc.wait()
