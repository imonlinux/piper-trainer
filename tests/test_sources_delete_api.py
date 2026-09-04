"""Source deletion: POST /sources/delete moves raw/ files to the
workspace .trash (§0), refusing while raw/-reading jobs are live."""
from __future__ import annotations

import asyncio
import json
import wave

import pytest
from fastapi.testclient import TestClient

from piper_trainer.api.app import create_app


def write_wav(path, seconds=1.0, rate=22050):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def make_project(tmp_path):
    root = tmp_path / "proj"
    root.mkdir(parents=True)
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    (root / "dataset").mkdir()
    (root / "raw").mkdir()
    return root


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as c:
        yield c


def test_delete_moves_to_trash(tmp_path, client):
    root = make_project(tmp_path)
    write_wav(root / "raw" / "a.wav")
    write_wav(root / "raw" / "b.wav")
    r = client.post("/api/projects/proj/sources/delete",
                    json={"names": ["a.wav"]})
    assert r.status_code == 200
    body = r.json()
    assert body == {"moved": ["a.wav"], "missing": []}
    assert not (root / "raw" / "a.wav").exists()
    assert (root / "raw" / "b.wav").is_file()
    # §0: recoverable, not destroyed
    trash = next((tmp_path / ".trash").iterdir())
    assert (trash / "a.wav").is_file()


def test_delete_bulk_reports_missing(tmp_path, client):
    root = make_project(tmp_path)
    write_wav(root / "raw" / "keep.wav")
    r = client.post("/api/projects/proj/sources/delete",
                    json={"names": ["keep.wav", "nope.wav", "keep.wav"]})
    assert r.status_code == 200
    assert r.json() == {"moved": ["keep.wav"], "missing": ["nope.wav"]}
    assert not (root / "raw" / "keep.wav").exists()


def test_delete_sanitizes_paths(tmp_path, client):
    """A name is a basename: ../escape.wav targets raw/escape.wav, and
    neither exists — both come back as missing, nothing outside raw/."""
    make_project(tmp_path)
    outside = tmp_path / "escape.wav"
    write_wav(outside)
    r = client.post("/api/projects/proj/sources/delete",
                    json={"names": ["../escape.wav", ".."]})
    assert r.status_code == 200
    assert r.json() == {"moved": [], "missing": ["../escape.wav", ".."]}
    assert outside.is_file()


def test_delete_skips_non_audio(tmp_path, client):
    """Only files sources() would list are eligible — a notes.txt in
    raw/ is reported missing, not moved."""
    root = make_project(tmp_path)
    (root / "raw" / "notes.txt").write_text("not audio")
    r = client.post("/api/projects/proj/sources/delete",
                    json={"names": ["notes.txt"]})
    assert r.status_code == 200
    assert r.json() == {"moved": [], "missing": ["notes.txt"]}
    assert (root / "raw" / "notes.txt").is_file()


def test_delete_empty_names_400(tmp_path, client):
    make_project(tmp_path)
    r = client.post("/api/projects/proj/sources/delete", json={"names": []})
    assert r.status_code == 400


def test_delete_unknown_project_404(client, tmp_path):
    make_project(tmp_path)
    r = client.post("/api/projects/ghost/sources/delete",
                    json={"names": ["a.wav"]})
    assert r.status_code == 404


def test_delete_refused_while_raw_jobs_active(tmp_path, client):
    """prepare/preview/ingest read raw/: with one queued, deletion must
    refuse (409) and leave the file in place."""
    root = make_project(tmp_path)
    write_wav(root / "raw" / "a.wav")
    asyncio.run(
        client.app.state.manager.submit(root, "prepare", run=False))
    r = client.post("/api/projects/proj/sources/delete",
                    json={"names": ["a.wav"]})
    assert r.status_code == 409
    assert "prepare" in r.json()["detail"]
    assert (root / "raw" / "a.wav").is_file()
