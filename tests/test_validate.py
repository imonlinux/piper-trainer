"""Tests for validate: BAD_TEXT regex, validate_dataset, Finding.action."""
import math
import struct
import types
import wave
from pathlib import Path

from piper_trainer.config import Project, TIERS
from piper_trainer.validate import (BAD_TEXT, ACTIONS, Finding,
                                    validate_dataset)


def write_tone(path: Path, rate: int, seconds: float, freq: float = 440.0) -> None:
    frames = int(rate * seconds)
    data = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate)))
        for i in range(frames))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data)


# ------------------------------------------------------------------- BAD_TEXT

def test_bad_text_does_not_flag_sentence_final_i_and_a():
    assert not BAD_TEXT.search("and then I.")
    assert not BAD_TEXT.search("and then A.")
    assert not BAD_TEXT.search("I am here")


def test_bad_text_flags_known_abbreviations():
    assert BAD_TEXT.search("Mr. Smith was here")
    assert BAD_TEXT.search("Dr. Who arrived")
    assert BAD_TEXT.search("St. Andrews is far")
    assert BAD_TEXT.search("the company Inc. is large")
    assert BAD_TEXT.search("live on Ave. 5")


def test_bad_text_flags_symbols_and_digits():
    assert BAD_TEXT.search("I was 5 years old")
    assert BAD_TEXT.search("R&D department")
    assert BAD_TEXT.search("a 50% chance")


def test_bad_text_flags_e_g_and_ie():
    assert BAD_TEXT.search("use e.g. this one")
    assert BAD_TEXT.search("that is, i.e. the first")


# ------------------------------------------------------------ validate_dataset

def make_project(tmp_path: Path, rows: list[tuple[str, str]]) -> Project:
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    from piper_trainer import metadata
    metadata.write(proj.metadata, rows)
    return proj


def test_validate_dataset_codes(tmp_path):
    rate = TIERS["medium"]["sample_rate"]
    proj = make_project(tmp_path, [
        ("a", "alpha"),      # 2.0s @ 22050  -> clean
        ("b", "beta"),       # 2.0s @ 16000  -> sample-rate
        ("c", "gamma"),      # 0.5s @ 22050  -> short-clips
        ("d", "delta"),      # no WAV        -> missing-wav
    ])
    write_tone(proj.wavs / "a.wav", rate, 2.0)
    write_tone(proj.wavs / "b.wav", 16000, 2.0)
    write_tone(proj.wavs / "c.wav", rate, 0.5)
    write_tone(proj.wavs / "e.wav", rate, 2.0)  # orphan

    findings = validate_dataset(proj, tier="medium", batch_size=100)
    codes = {f.code: f for f in findings}

    assert "missing-wav" in codes
    assert codes["missing-wav"].ids == ["d"]
    assert codes["missing-wav"].level == "error"

    assert "orphan-wav" in codes
    assert codes["orphan-wav"].ids == ["e"]

    assert "sample-rate" in codes
    assert codes["sample-rate"].ids == ["b"]

    assert "short-clips" in codes
    assert codes["short-clips"].ids == ["c"]
    assert "long-clips" not in codes

    assert "crlf" not in codes
    assert "columns" not in codes
    assert "blank-row" not in codes

    # 100 >= int(4 * 0.98) == 3 -> one batch per epoch
    assert "batch-size" in codes
    assert codes["batch-size"].level == "error"


def test_validate_dataset_crlf_detected(tmp_path):
    proj = make_project(tmp_path, [("a", "ok")])
    proj.metadata.write_bytes(b"a|ok\r\n")
    findings = validate_dataset(proj, tier="medium")
    codes = {f.code for f in findings}
    assert "crlf" in codes


def test_validate_dataset_blank_and_columns(tmp_path):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    (proj.wavs).mkdir(parents=True, exist_ok=True)
    proj.metadata.write_bytes(b"a|ok\n\nbad\nb|\n")
    write_tone(proj.wavs / "a.wav", TIERS["medium"]["sample_rate"], 2.0)
    findings = validate_dataset(proj, tier="medium")
    codes = [f.code for f in findings]
    assert codes.count("blank-row") == 1
    assert codes.count("columns") == 1
    lines = {f for f in findings if f.code == "blank-row"}
    assert "line 2" in lines.pop().message


def test_validate_dataset_no_metadata(tmp_path):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    findings = validate_dataset(proj, tier="medium")
    assert [f.code for f in findings] == ["no-metadata"]


# -------------------------------------------------------------- Finding.action

def test_finding_action_mapping():
    assert Finding("error", "missing-wav", "x").action == "drop-row"
    assert Finding("warn", "orphan-wav", "x").action == "quarantine"
    assert Finding("error", "crlf", "x").action == "repair"
    assert Finding("error", "unspoken-text", "x").action == "repair"
    assert Finding("info", "duration", "x").action is None


def test_finding_str_contains_code():
    s = str(Finding("warn", "orphan-wav", "three wav files"))
    assert "orphan-wav" in s and "quarantine" in s


def test_actions_table_covers_known_codes():
    for code in ("crlf", "columns", "blank-row", "unspoken-text",
                 "missing-wav", "orphan-wav", "unreadable", "short-clips",
                 "long-clips", "cps-outliers", "sample-rate", "channels"):
        assert code in ACTIONS


# ------------------------------------------------------- 7c/7d additions

def test_validation_split_warns_at_zero_clip_boundary(tmp_path):
    """round(n * split) < 1 warns — unless the split is deliberately 0."""
    proj25 = make_project(tmp_path, [(f"c{i}", "text") for i in range(25)])
    codes = {f.code for f in validate_dataset(proj25, tier="medium",
                                              validation_split=0.02)}
    assert "validation-split" in codes  # round(0.5) == 0

    proj50 = make_project(tmp_path / "b", [(f"c{i}", "text") for i in range(50)])
    codes = {f.code for f in validate_dataset(proj50, tier="medium",
                                              validation_split=0.02)}
    assert "validation-split" not in codes  # round(1.0) == 1

    codes = {f.code for f in validate_dataset(proj25, tier="medium",
                                              validation_split=0.0)}
    assert "validation-split" not in codes  # deliberately off


def test_espeak_missing_finding_when_binary_absent(tmp_path, monkeypatch):
    proj = make_project(tmp_path, [("a", "ok")])

    def boom(*a, **k):
        raise FileNotFoundError("espeak-ng")

    monkeypatch.setattr("piper_trainer.validate.subprocess.run", boom)
    findings = validate_dataset(proj, tier="medium", espeak_voice="en-us")
    codes = {f.code for f in findings}
    assert "espeak-missing" in codes
    assert "espeak-voice" not in codes  # check skipped, not silently passed


def test_espeak_voice_check_runs_when_binary_present(tmp_path, monkeypatch):
    proj = make_project(tmp_path, [("a", "ok")])
    monkeypatch.setattr(
        "piper_trainer.validate.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(
            stdout=" 5 en-us +21/M 130 en_US\n 5 de +21/M 130 de_DE\n"))
    findings = validate_dataset(proj, tier="medium", espeak_voice="en-gb-x-rp")
    assert "espeak-voice" in {f.code for f in findings}
