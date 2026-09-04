"""Audit dataset endpoints: GET rows (metadata × audit join, quarantine
state) and PATCH transcript editing (design doc §6.3)."""
from __future__ import annotations

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


def make_project(tmp_path, rows=None, audit=None, quarantine_manifest=None):
    """A project with metadata.csv, real WAVs, optional audit.csv."""
    root = tmp_path / "proj"
    root.mkdir(parents=True)
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    (root / "dataset").mkdir()
    rows = rows if rows is not None else [
        ("clip_a", "hello there"),
        ("clip_b", "second clip"),
    ]
    with (root / "dataset" / "metadata.csv").open("w", newline="") as fh:
        for cid, text in rows:
            fh.write(f"{cid}|{text}\n")
    wavs = root / "dataset" / "wavs"
    write_wav(wavs / "clip_a.wav", seconds=2.0)
    if len(rows) > 1:
        write_wav(wavs / "clip_b.wav", seconds=4.0)
    if audit is not None:
        with (root / "dataset" / "audit.csv").open("w", newline="") as fh:
            fh.write("file,duration,chars_per_sec,lang_prob,text\n")
            for row in audit:
                fh.write(",".join(str(c) for c in row) + "\n")
    if quarantine_manifest is not None:
        qdir = root / "dataset" / "quarantine"
        qdir.mkdir(parents=True)
        with (qdir / "manifest.csv").open("w", newline="") as fh:
            fh.write("timestamp,clip_id,action,reasons,text\n")
            for row in quarantine_manifest:
                fh.write(",".join(row) + "\n")
    return root


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as c:
        yield c


def seed(client, **kwargs):
    ws = client.app.state.workspace
    root = make_project(ws, **kwargs)
    return ws / "proj" / "dataset" / "metadata.csv"


# --------------------------------------------------------------------- rows

def test_rows_join_audit_scores(client):
    seed(client, audit=[
        ["clip_a.wav", 2.0, 5.5, 0.99, "hello there"],
    ])
    r = client.get("/api/projects/proj/dataset")
    assert r.status_code == 200
    rows = {row["id"]: row for row in r.json()["rows"]}
    assert rows["clip_a"]["duration"] == 2.0
    assert rows["clip_a"]["cps"] == 5.5
    assert rows["clip_a"]["lang_prob"] == 0.99
    assert rows["clip_a"]["text"] == "hello there"
    assert rows["clip_a"]["missing"] is False
    assert rows["clip_a"]["quarantined"] is False
    # clip_b has no audit row: duration probed from the WAV, cps derived
    assert rows["clip_b"]["duration"] == pytest.approx(4.0)
    assert rows["clip_b"]["lang_prob"] is None
    assert rows["clip_b"]["cps"] == pytest.approx(len("second clip") / 4.0, abs=0.1)


def test_rows_missing_wav_flagged(client):
    seed(client, rows=[("clip_a", "hello there"),
                       ("ghost", "no wav for me")])
    rows = {row["id"]: row
            for row in client.get("/api/projects/proj/dataset").json()["rows"]}
    assert rows["ghost"]["missing"] is True
    assert rows["ghost"]["duration"] is None
    assert rows["clip_a"]["missing"] is False


def test_rows_quarantine_flag_from_manifest(client):
    seed(client, quarantine_manifest=[
        ["20260904T000000Z", "clip_b", "quarantine", "short-clips", "second clip"],
    ])
    rows = {row["id"]: row
            for row in client.get("/api/projects/proj/dataset").json()["rows"]}
    assert rows["clip_b"]["quarantined"] is True
    assert rows["clip_a"]["quarantined"] is False


def test_quarantine_listing_most_recent_first(client):
    seed(client, quarantine_manifest=[
        ["20260903T000000Z", "clip_a", "quarantine", "cps-outliers", "hello there"],
        ["20260904T000000Z", "clip_b", "quarantine", "short-clips", "second clip"],
    ])
    r = client.get("/api/projects/proj/dataset")
    q = r.json()["quarantine"]
    assert [e["clip_id"] for e in q] == ["clip_b", "clip_a"]


def test_rows_empty_without_dataset(client):
    (client.app.state.workspace / "proj").mkdir(parents=True)
    (client.app.state.workspace / "proj" / "project.json").write_text(
        json.dumps({"name": "proj"}))
    r = client.get("/api/projects/proj/dataset")
    assert r.status_code == 200
    assert r.json() == {"rows": [], "quarantine": []}


# ------------------------------------------------------------------- PATCH

def test_patch_transcript_edits_one_row(client):
    path = seed(client)
    r = client.patch("/api/projects/proj/dataset/clip_a",
                     json={"text": "hello again"})
    assert r.status_code == 200
    assert r.json() == {"id": "clip_a", "text": "hello again"}
    lines = path.read_text().splitlines()
    assert lines == ["clip_a|hello again", "clip_b|second clip"]


def test_patch_transcript_preserves_malformed_lines(client):
    path = seed(client)
    path.write_text("clip_a|hello there\n\nclip_b|second clip\nbroken\n")
    r = client.patch("/api/projects/proj/dataset/clip_b",
                     json={"text": "rewritten"})
    assert r.status_code == 200
    out = path.read_text().splitlines()
    # the edit lands on its row; blank + malformed lines keep their positions
    assert out == ["clip_a|hello there", "", "clip_b|rewritten", "broken"]


def test_patch_transcript_strips_whitespace(client):
    path = seed(client)
    r = client.patch("/api/projects/proj/dataset/clip_a",
                     json={"text": "  padded text  "})
    assert r.status_code == 200
    assert "clip_a|padded text" in path.read_text()


def test_patch_transcript_404_unknown_clip(client):
    seed(client)
    r = client.patch("/api/projects/proj/dataset/nope",
                     json={"text": "x"})
    assert r.status_code == 404


def test_patch_transcript_400_empty_or_multiline(client):
    seed(client)
    for bad in ("", "   ", "two\nlines"):
        r = client.patch("/api/projects/proj/dataset/clip_a",
                         json={"text": bad})
        assert r.status_code == 400, bad


def test_patch_transcript_404_without_metadata(client):
    root = client.app.state.workspace / "proj"
    root.mkdir(parents=True)
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    r = client.patch("/api/projects/proj/dataset/clip_a",
                     json={"text": "x"})
    assert r.status_code == 404


def test_patch_unknown_project_404(client):
    r = client.patch("/api/projects/ghost/dataset/clip_a",
                     json={"text": "x"})
    assert r.status_code == 404
