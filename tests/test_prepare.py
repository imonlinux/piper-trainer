"""Tests for prepare: naming, stage manifests, skip/--force idempotency."""
import json
import sys
import types
from pathlib import Path

import pytest

from piper_trainer import prepare
from piper_trainer.config import Project


@pytest.fixture
def proj(tmp_path) -> Project:
    p = Project(root=tmp_path, name="t")
    p.ensure()
    return p


@pytest.fixture
def fake_tools(monkeypatch):
    """_run pretends ffmpeg/deep-filter wrote their outputs; auditok splits
    every input into two fixed events."""
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        if cmd[0] == "deep-filter":
            out = Path(cmd[cmd.index("-o") + 1])
            out.mkdir(parents=True, exist_ok=True)
            for a in cmd[cmd.index("-o") + 2:]:
                (out / Path(a).name).write_bytes(b"RIFF")
        elif cmd[-1].endswith(".wav"):
            dst = Path(cmd[-1])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"RIFF")

    class _Ev:
        def __init__(self, start, end):
            self.start, self.end = start, end

        def save(self, path):
            Path(path).write_bytes(b"RIFF")

    class _Region:
        def __init__(self, path):
            self._path = path

        def split(self, **kw):
            return [_Ev(0.0, 2.0), _Ev(2.5, 4.5)]

    fake_auditok = types.ModuleType("auditok")
    fake_auditok.load = lambda p: _Region(p)
    monkeypatch.setattr(prepare, "_run", fake_run)
    monkeypatch.setitem(sys.modules, "auditok", fake_auditok)
    return calls


# -------------------------------------------------------------------- naming

def test_sanitize_stem():
    assert prepare.sanitize_stem("plain-name_1.wav") == "plain-name_1.wav"
    assert prepare.sanitize_stem("my file?") == "my_file_"
    assert prepare.sanitize_stem("héllo   wörld") == "h_llo_w_rld"
    assert prepare.sanitize_stem("???") == "_"
    assert prepare.sanitize_stem("a...b") == "a...b"


def test_collision_group_gets_ext_suffix_for_every_member():
    srcs = [Path("foo.wav"), Path("foo.mp3"), Path("bar.wav")]
    names = prepare.assign_names(srcs)
    assert names[Path("foo.wav")] == "foo_wav"
    assert names[Path("foo.mp3")] == "foo_mp3"
    assert names[Path("bar.wav")] == "bar"


def test_collision_on_sanitized_names():
    """'foo bar' and 'foo?bar' sanitize to the same stem; the group must be
    resolved on the sanitized names, not silently overwritten."""
    srcs = [Path("foo bar.wav"), Path("foo?bar.wav")]
    names = prepare.assign_names(srcs)
    assert sorted(names.values()) == ["foo_bar_wav", "foo_bar_wav_2"]


def test_to_48k_reports_renamed_destinations(proj, fake_tools):
    (proj.raw / "foo.wav").write_bytes(b"a")
    (proj.raw / "foo.mp3").write_bytes(b"b")
    (proj.raw / "my file?.wav").write_bytes(b"c")
    n, renamed = prepare.to_48k(proj)
    assert n == 3
    assert renamed == {"foo.wav": "foo_wav.wav",
                       "foo.mp3": "foo_mp3.wav",
                       "my file?.wav": "my_file_.wav"}
    assert sorted(p.name for p in proj.work48k.glob("*.wav")) == \
        ["foo_mp3.wav", "foo_wav.wav", "my_file_.wav"]


# ------------------------------------------------------------ stage manifests

def read_manifest(d: Path) -> dict:
    return json.loads((d / ".stage.json").read_text())


def test_stage_manifest_round_trip(proj, fake_tools):
    (proj.raw / "a.wav").write_bytes(b"x")
    n, _ = prepare.to_48k(proj)
    assert n == 1
    mf = read_manifest(proj.work48k)
    assert mf["stage"] == "to_48k"
    assert mf["params"] == {"channel": "downmix"}
    assert mf["outputs"] == 1
    assert mf["input_fingerprint"] == prepare._fingerprint(
        [proj.raw / "a.wav"])
    assert "completed_at" in mf


def test_matching_fingerprint_skips_and_force_reruns(proj, fake_tools):
    (proj.raw / "a.wav").write_bytes(b"x")
    assert prepare.to_48k(proj)[0] == 1
    assert prepare.to_48k(proj)[0] == "skipped"
    assert prepare.to_48k(proj, force=True)[0] == 1
    assert prepare.to_48k(proj)[0] == "skipped"


def test_changed_parameter_does_not_skip(proj, fake_tools):
    (proj.raw / "a.wav").write_bytes(b"x")
    prepare.to_48k(proj)
    assert prepare.to_48k(proj, channel="left")[0] == 1


def test_changed_input_does_not_skip(proj, fake_tools):
    (proj.raw / "a.wav").write_bytes(b"x")
    prepare.to_48k(proj)
    (proj.raw / "a.wav").write_bytes(b"changed size")
    assert prepare.to_48k(proj)[0] == 1


def test_stage_clears_stale_outputs(proj, fake_tools):
    (proj.raw / "a.wav").write_bytes(b"x")
    prepare.to_48k(proj)
    stale = proj.work48k / "stale.wav"
    stale.write_bytes(b"old")
    (proj.raw / "a.wav").write_bytes(b"new size")
    prepare.to_48k(proj)
    assert not stale.exists()
    assert (proj.work48k / "a.wav").exists()


# ----------------------------------------------------------------- run_all

def test_run_all_then_skip_then_selective_rerun(proj, fake_tools):
    (proj.raw / "a.wav").write_bytes(b"x")
    stats = prepare.run_all(proj)
    assert stats == {"converted": 1, "denoised": 1, "clips": 2, "finalized": 2}

    # identical invocation: every stage skips
    stats = prepare.run_all(proj)
    assert stats == {"converted": "skipped", "denoised": "skipped",
                     "clips": "skipped", "finalized": "skipped"}

    # different segmentation parameter: only segment + finalize re-run
    stats = prepare.run_all(proj, energy_threshold=45)
    assert stats["converted"] == "skipped"
    assert stats["denoised"] == "skipped"
    assert stats["clips"] == 2
    assert stats["finalized"] == 2


def test_run_all_clears_previous_clips_from_wavs(proj, fake_tools):
    """The dataset must reflect one parameter set, not two."""
    (proj.raw / "a.wav").write_bytes(b"x")
    prepare.run_all(proj)
    stray = proj.wavs / "a_9999_0.000-9.999.wav"
    stray.write_bytes(b"stray")
    # re-segment with a different parameter; wavs must reflect ONLY that run
    prepare.run_all(proj, energy_threshold=45)
    names = sorted(p.name for p in proj.wavs.glob("*.wav"))
    assert stray.name not in names
    assert len(names) == 2


# ------------------------------------------------------ zero-clip visibility

def test_segment_reports_per_source_counts(proj, fake_tools, capsys):
    (proj.denoised / "a.wav").write_bytes(b"x")
    total, per_source = prepare.segment(proj)
    assert total == 2
    assert per_source == {"a.wav": 2}
    assert "segment: a.wav: 2 clips" in capsys.readouterr().out


def test_zero_clip_source_is_flagged_in_run_all(proj, fake_tools, monkeypatch):
    """A source the VAD rejects used to vanish silently: no clips, no
    transcripts, no training data, and nothing in the run's stats saying
    why. The run must now name it."""
    class _Region:
        def __init__(self, path):
            self._path = path

        def split(self, **kw):
            # the quiet file finds nothing above the threshold
            return [] if "quiet" in self._path else _EVENTS

    _EVENTS = [types.SimpleNamespace(start=0.0, end=2.0,
                                     save=lambda p: Path(p).write_bytes(b"x")),
               types.SimpleNamespace(start=2.5, end=4.5,
                                     save=lambda p: Path(p).write_bytes(b"x"))]
    fake_auditok = types.ModuleType("auditok")
    fake_auditok.load = lambda p: _Region(p)
    monkeypatch.setitem(sys.modules, "auditok", fake_auditok)

    (proj.raw / "a.wav").write_bytes(b"x")
    (proj.raw / "quiet.wav").write_bytes(b"x")
    stats = prepare.run_all(proj)
    assert stats["clips"] == 2
    assert stats["clips_zero_sources"] == ["quiet.wav"]
    assert stats["clips_per_source"] == {"a.wav": 2, "quiet.wav": 0}


# ----------------------------------------------------------------- levels

def test_wav_level_dbfs_reports_peak_rms_and_speech(tmp_path):
    import math
    import wave

    path = tmp_path / "lvl.wav"
    amp = 8000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        # 1 s of sine, then 1 s of digital silence: speech must reflect the
        # loud half only, while the overall RMS is pulled down by the gap
        frames = b"".join(
            int(amp * math.sin(2 * math.pi * 220 * i / 48000))
            .to_bytes(2, "little", signed=True) for i in range(48000))
        w.writeframes(frames + bytes(96000))

    lvl = prepare.wav_level_dbfs(path)
    expected_peak = 20 * math.log10(amp / 32768)
    assert lvl["peak_dbfs"] == pytest.approx(expected_peak, abs=0.1)
    assert lvl["speech_dbfs"] == pytest.approx(
        expected_peak - 3.0, abs=0.5)   # sine RMS = amp/sqrt(2)
    assert lvl["rms_dbfs"] < lvl["speech_dbfs"] - 2.5  # silence dilutes it


def test_wav_level_dbfs_rejects_non_16bit(tmp_path):
    import wave

    path = tmp_path / "w.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(48000)
        w.writeframes(bytes(4800))
    with pytest.raises(ValueError, match="16-bit mono"):
        prepare.wav_level_dbfs(path)
