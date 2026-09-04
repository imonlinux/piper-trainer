"""Voices screen (§6.5): list/patch/say/download over out/ exports, and
the say subprocess wrapper itself (piper CLI faked)."""
from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from piper_trainer import say
from piper_trainer.api.app import create_app
from piper_trainer.config import Project

# A stub .onnx can never sit inside export.verify's 20-200 MB window, so
# every listed voice carries exactly that one problem — anything else on
# top of it would be a real defect.
ONLY_SIZE = ["unexpected model size 0 MB"]


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as c:
        yield c


def make_voice(project: Project, stem: str = "proj-medium-voice",
               epoch: int | None = 42) -> None:
    """Write an exported-voice pair by hand: an .onnx stub and a COMPLETE
    .onnx.json with the name agreement export.py enforces."""
    project.out.mkdir(parents=True, exist_ok=True)
    (project.out / f"{stem}.onnx").write_bytes(b"RIFF" + b"\0" * 64)
    cfg = {
        "dataset": stem,
        "audio": {"quality": "medium", "sample_rate": 22050},
        "language": {"code": "en-us"},
        "espeak": {"voice": "en-us"},
        "inference": {"noise_scale": 0.667, "length_scale": 1.0,
                      "noise_w": 0.8},
        "num_symbols": 256, "num_speakers": 1,
        "phoneme_id_map": {"_": [0]},
    }
    if epoch is not None:
        cfg["checkpoint_epoch"] = epoch
    (project.out / f"{stem}.onnx.json").write_text(json.dumps(cfg))


@pytest.fixture
def proj(tmp_path):
    p = Project(root=tmp_path / "proj", name="proj")
    p.ensure()
    (tmp_path / "proj" / "project.json").write_text(json.dumps({"name": "proj"}))
    return p


# ------------------------------------------------------------------ list

def test_list_empty_then_one(client, tmp_path, proj):
    r = client.get("/api/projects/proj/voices")
    assert r.status_code == 200
    assert r.json() == []

    make_voice(proj)
    r = client.get("/api/projects/proj/voices")
    assert r.status_code == 200
    voices = r.json()
    assert len(voices) == 1
    v = voices[0]
    assert v["stem"] == "proj-medium-voice"
    assert v["quality"] == "medium"
    assert v["language"] == "en-us"
    assert v["checkpoint_epoch"] == 42
    assert v["inference"]["length_scale"] == 1.0
    # only the size problem is expected on a stub; anything else is real
    assert v["problems"] == ONLY_SIZE


def test_detail_and_404(client, tmp_path, proj):
    make_voice(proj)
    r = client.get("/api/projects/proj/voices/proj-medium-voice")
    assert r.status_code == 200
    assert r.json()["stem"] == "proj-medium-voice"
    assert client.get("/api/projects/proj/voices/nope").status_code == 404
    # path traversal is refused as a bad name, not a 404 leak
    r = client.get("/api/projects/proj/voices/..%2Fproj%2Fout%2Fx")
    assert r.status_code in (400, 404)


# ------------------------------------------------------------------ patch

def test_patch_updates_inference_only(client, tmp_path, proj):
    make_voice(proj)
    r = client.patch("/api/projects/proj/voices/proj-medium-voice",
                     json={"length_scale": 0.85, "noise_w": 1.1})
    assert r.status_code == 200
    v = r.json()
    assert v["inference"]["length_scale"] == 0.85
    assert v["inference"]["noise_w"] == 1.1
    assert v["inference"]["noise_scale"] == 0.667  # untouched

    cfg = json.loads(
        (proj.out / "proj-medium-voice.onnx.json").read_text())
    assert cfg["dataset"] == "proj-medium-voice"  # agreement preserved
    assert cfg["audio"]["quality"] == "medium"  # non-inference keys intact
    assert cfg["checkpoint_epoch"] == 42


def test_patch_rejects_out_of_range_and_empty(client, proj):
    make_voice(proj)
    r = client.patch("/api/projects/proj/voices/proj-medium-voice",
                     json={"length_scale": 9.9})
    assert r.status_code == 422
    r = client.patch("/api/projects/proj/voices/proj-medium-voice", json={})
    assert r.status_code == 400


# ------------------------------------------------------------------ say

@pytest.fixture
def fake_say(monkeypatch):
    """Capture the say.synthesize call; return a wav-looking payload."""
    seen = {}

    def fake(onnx, json_path, text, length_scale=None, noise_scale=None,
             noise_w=None, timeout=say.DEFAULT_TIMEOUT):
        seen["onnx"] = onnx.name
        seen["text"] = text
        seen["params"] = (length_scale, noise_scale, noise_w)
        return b"RIFF-fake-wav"

    monkeypatch.setattr(say, "synthesize", fake)
    return seen


def test_say_returns_wav_bytes(client, tmp_path, proj, fake_say):
    make_voice(proj)
    r = client.post("/api/projects/proj/voices/proj-medium-voice/say",
                    json={"text": "nine thousand credits",
                          "length_scale": 0.9})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert r.content == b"RIFF-fake-wav"
    assert fake_say["text"] == "nine thousand credits"
    assert fake_say["params"] == (0.9, None, None)


def test_say_rejects_empty_text(client, tmp_path, proj):
    make_voice(proj)
    r = client.post("/api/projects/proj/voices/proj-medium-voice/say",
                    json={"text": "   "})
    assert r.status_code == 400  # past pydantic (len 3); the module refuses
    r = client.post("/api/projects/proj/voices/proj-medium-voice/say",
                    json={"text": "hi" * 600})
    assert r.status_code == 422  # pydantic max_length


def test_say_surfaces_say_error(client, tmp_path, proj, monkeypatch):
    make_voice(proj)

    def boom(*a, **k):
        raise say.SayError("piper CLI not found")

    monkeypatch.setattr(say, "synthesize", boom)
    r = client.post("/api/projects/proj/voices/proj-medium-voice/say",
                    json={"text": "hello"})
    assert r.status_code == 502
    assert "piper CLI not found" in r.json()["detail"]


# ------------------------------------------------------------------ download

def test_download_onnx_and_json(client, tmp_path, proj):
    make_voice(proj)
    r = client.get(
        "/api/projects/proj/voices/proj-medium-voice/download?file=onnx")
    assert r.status_code == 200
    assert r.headers["content-disposition"].endswith(
        'filename="proj-medium-voice.onnx"')
    assert r.content.startswith(b"RIFF")

    r = client.get(
        "/api/projects/proj/voices/proj-medium-voice/download?file=json")
    assert r.status_code == 200
    assert json.loads(r.content)["dataset"] == "proj-medium-voice"

    r = client.get(
        "/api/projects/proj/voices/proj-medium-voice/download?file=nope")
    assert r.status_code == 400


# ------------------------------------------------------------------ say.py

def test_say_synthesize_builds_cli_command(monkeypatch):
    """The wrapper drives `python -m piper` with stdout as the wav sink,
    text on stdin, and the voice's own defaults for absent params."""
    called = {}

    class Proc:
        returncode = 0
        stdout = b"RIFF-wav"
        stderr = b""

    def fake_run(cmd, input=None, stdout=None, stderr=None, timeout=None):
        called["cmd"] = cmd
        called["stdin"] = input
        return Proc()

    monkeypatch.setattr(say.subprocess, "run", fake_run)
    json_path = types.SimpleNamespace(
        read_text=lambda: json.dumps({"inference": {"length_scale": 1.2}}))
    wav = say.synthesize(
        Path("/x/v.onnx"), json_path, "hello",
        noise_w=1.0)
    assert wav == b"RIFF-wav"
    assert called["stdin"] == b"hello"
    cmd = called["cmd"]
    assert cmd[1:3] == ["-m", "piper"]
    assert cmd[cmd.index("--length-scale") + 1] == "1.2"  # config default
    assert cmd[cmd.index("--noise-w") + 1] == "1.0"  # explicit override
    assert cmd[-1] == "-"  # wav to stdout


def test_say_synthesize_validates_text_and_output(monkeypatch):
    json_path = types.SimpleNamespace(read_text=lambda: "{}")

    with pytest.raises(ValueError):
        say.synthesize(Path("/x/v.onnx"), json_path, "  ")
    with pytest.raises(ValueError):
        say.synthesize(Path("/x/v.onnx"), json_path,
                       "x" * (say.MAX_TEXT + 1))

    class Bad:
        returncode = 1
        stdout = b""
        stderr = b"model load failed"

    monkeypatch.setattr(say.subprocess, "run",
                        lambda *a, **k: Bad())
    with pytest.raises(say.SayError, match="model load failed"):
        say.synthesize(Path("/x/v.onnx"), json_path,
                       "hi")

    class Empty:
        returncode = 0
        stdout = b"not a wav"
        stderr = b""

    monkeypatch.setattr(say.subprocess, "run", lambda *a, **k: Empty())
    with pytest.raises(say.SayError, match="no wav"):
        say.synthesize(Path("/x/v.onnx"), json_path,
                       "hi")


def test_say_cli_missing_is_actionable(monkeypatch):
    def no_piper(*a, **k):
        raise FileNotFoundError()

    monkeypatch.setattr(say.subprocess, "run", no_piper)
    json_path = types.SimpleNamespace(read_text=lambda: "{}")
    with pytest.raises(say.SayError, match="piper-trainer doctor"):
        say.synthesize(Path("/x/v.onnx"), json_path,
                       "hi")
