"""Tests for transcribe resumability (Whisper model is faked)."""
import math
import struct
import sys
import types
import wave
from pathlib import Path

import pytest

from piper_trainer import metadata, transcribe
from piper_trainer.config import Project


def write_tone(path: Path, rate: int = 16000, seconds: float = 1.0) -> None:
    frames = int(rate * seconds)
    data = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
        for i in range(frames))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)


@pytest.fixture
def fake_whisper(monkeypatch):
    """Every call transcribes with a fresh text suffix so carried-forward
    rows (old suffix) are distinguishable from fresh ones."""
    state = {"n": 0}

    class _Seg:
        def __init__(self, text):
            self.text = text

    class _Info:
        language_probability = 0.97

    class _FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, path, **k):
            state["n"] += 1
            stem = Path(path).stem
            return iter([_Seg(f"text-{state['n']} for {stem}")]), _Info()

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = _FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    return state


def test_first_run_transcribes_everything(tmp_path, fake_whisper):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    for stem in ("a", "b", "c"):
        write_tone(proj.wavs / f"{stem}.wav")
    stats = transcribe.transcribe(proj)
    assert stats["transcribed"] == 3
    assert stats["skipped"] == 0
    rows, _ = metadata.read(proj.metadata)
    assert [r[0] for r in rows] == ["a", "b", "c"]
    assert all("text-1" in r[1] or "text-" in r[1] for r in rows)


def test_second_run_skips_existing_and_carries_audit(tmp_path, fake_whisper):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    for stem in ("a", "b", "c"):
        write_tone(proj.wavs / f"{stem}.wav")
    transcribe.transcribe(proj)
    first_rows = dict(metadata.read(proj.metadata)[0])
    first_audit = proj.audit.read_text()

    write_tone(proj.wavs / "d.wav")
    stats = transcribe.transcribe(proj)

    assert stats["transcribed"] == 1
    assert stats["skipped"] == 3
    rows = dict(metadata.read(proj.metadata)[0])
    assert rows["a"] == first_rows["a"]  # original text, not re-transcribed
    assert "d" in rows
    # audit rows for skipped clips carried forward verbatim
    for line in first_audit.splitlines()[1:]:
        assert line in proj.audit.read_text()


def test_retranscribe_forces_full_pass(tmp_path, fake_whisper):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    for stem in ("a", "b"):
        write_tone(proj.wavs / f"{stem}.wav")
    transcribe.transcribe(proj)
    stats = transcribe.transcribe(proj, retranscribe=True)
    assert stats["transcribed"] == 2
    assert stats["skipped"] == 0


def test_on_progress_fires_for_every_clip(tmp_path, fake_whisper):
    """The live-progress contract: one callback per clip, in filename
    order, covering skipped clips too — a resumed run must still move
    the bar, not stall until RESULT."""
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    for stem in ("a", "b", "c"):
        write_tone(proj.wavs / f"{stem}.wav")

    seen: list[tuple[int, int, str]] = []
    transcribe.transcribe(
        proj, on_progress=lambda d, t, n: seen.append((d, t, n)))
    assert seen == [(1, 3, "a.wav"), (2, 3, "b.wav"), (3, 3, "c.wav")]

    # Resumable pass: 3 skipped + 1 fresh, callback still fires 4 times.
    write_tone(proj.wavs / "d.wav")
    seen.clear()
    transcribe.transcribe(
        proj, on_progress=lambda d, t, n: seen.append((d, t, n)))
    assert seen == [
        (1, 4, "a.wav"), (2, 4, "b.wav"), (3, 4, "c.wav"), (4, 4, "d.wav")]


def test_missing_inflect_is_loud_not_silent(tmp_path, fake_whisper,
                                            monkeypatch, capsys):
    """When inflect is absent the numbers pass falls back to digits — that
    fallback used to be invisible and cost a debugging session. The stats
    must flag it (engine: None) and the log must carry a WARNING line."""
    from piper_trainer import textnorm

    real_engine = textnorm.engine()
    monkeypatch.setattr(textnorm, "engine", lambda: None)
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    write_tone(proj.wavs / "a.wav")
    write_tone(proj.wavs / "b.wav")

    stats = transcribe.transcribe(proj)

    assert stats["normalized"]["engine"] is None
    out = capsys.readouterr().out
    assert "WARNING" in out and "inflect" in out

    # With the engine present the key reports it and no warning prints.
    monkeypatch.setattr(textnorm, "engine", lambda: real_engine)
    stats = transcribe.transcribe(proj, retranscribe=True)
    assert stats["normalized"]["engine"] == "inflect"
    assert "WARNING" not in capsys.readouterr().out


def test_malformed_lines_survive_only_missing_rewrite(tmp_path, fake_whisper):
    """A hand-edited malformed row must survive a --only-missing run that
    rewrites metadata.csv — appended at the end, counted in the result."""
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    write_tone(proj.wavs / "a.wav")
    write_tone(proj.wavs / "c.wav")
    proj.metadata.write_bytes(b"a|existing row\nno-delimiter-here\nb|\n")

    stats = transcribe.transcribe(proj)  # c is missing -> only c transcribed

    assert stats["malformed_preserved"] == 2
    data = proj.metadata.read_bytes()
    assert data.endswith(b"no-delimiter-here\nb|\n")  # appended, not dropped
    rows, problems = metadata.read(proj.metadata)
    assert [r[0] for r in rows] == ["a", "c"]
    assert len(problems) == 2
