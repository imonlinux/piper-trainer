"""Previews: job variants scoped to a sample, writing only to
work/preview/<stage>/<preview-id>/ (design doc §2, §4.5).

auditok and deep-filter are runtime-image dependencies, so the stage
primitives are monkeypatched; the manager/lifecycle behavior is covered by
test_jobs with stub runners, and the endpoints are tested against
hand-written preview.json files.
"""
from __future__ import annotations

import json
import shutil

import pytest

from piper_trainer import prepare
from piper_trainer.api import runner


def make_project(tmp_path, name="proj"):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(json.dumps({"name": name}))
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "take.wav").write_bytes(b"RIFF-fake")
    return root


def make_job(tmp_path, params):
    root = make_project(tmp_path)
    jd = root / "jobs" / "20260901T000000Z-preview-test"
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "preview", "project": "proj",
        "params": params, "state": "queued", "pid": None}))
    (jd / "log.txt").touch()
    return jd, root


@pytest.fixture
def segment_stubs(monkeypatch):
    """Deterministic 5-clip split: 3x ~2 s, 1x ~7 s, 1x ~12 s."""
    def fake_convert(src, dst, channel=None):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"RIFF-48k")

    def fake_denoise(src, dst_dir):
        dst_dir.mkdir(parents=True, exist_ok=True)
        out = dst_dir / src.name
        out.write_bytes(b"RIFF-denoised")
        return out

    def fake_excerpt(src, dst, seconds):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def fake_split(src, out_dir, stem=None, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        clips = []
        for i, (s, e) in enumerate(boundaries, start=1):
            name = f"{stem or src.stem}_{i:04d}_{s:.3f}-{e:.3f}.wav"
            (out_dir / name).write_bytes(b"RIFF-clip")
            clips.append({"clip": name, "start": s, "end": e})
        return clips

    boundaries = [(0.0, 2.0), (2.5, 4.5), (5.0, 7.0),
                  (8.0, 15.0), (16.0, 28.0)]
    monkeypatch.setattr(prepare, "convert_one", fake_convert)
    monkeypatch.setattr(prepare, "denoise_file", fake_denoise)
    monkeypatch.setattr(prepare, "excerpt", fake_excerpt)
    monkeypatch.setattr(prepare, "split_audio", fake_split)
    return boundaries

    boundaries = [(0.0, 2.0), (2.5, 4.5), (5.0, 7.0),
                  (8.0, 15.0), (16.0, 28.0)]

    def fake_split(src, out_dir, stem=None, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        clips = []
        for i, (s, e) in enumerate(boundaries, start=1):
            name = f"{stem or src.stem}_{i:04d}_{s:.3f}-{e:.3f}.wav"
            (out_dir / name).write_bytes(b"RIFF-clip")
            clips.append({"clip": name, "start": s, "end": e})
        return clips

    monkeypatch.setattr(prepare, "convert_one", fake_convert)
    monkeypatch.setattr(prepare, "denoise_file", fake_denoise)
    monkeypatch.setattr(prepare, "split_audio", fake_split)
    return boundaries


# ------------------------------------------------------------------ segment

def test_segment_preview_writes_preview_json(tmp_path, segment_stubs):
    jd, root = make_job(tmp_path, {
        "stage": "segment", "source": "take.wav", "channel": "left",
        "energy_threshold": 40, "clips_kept": 2})
    result = runner.execute(jd)
    pdir = root / "work" / "preview" / "segment" / jd.name

    assert result["clip_count"] == 5
    assert result["duration_total"] == pytest.approx(25.0)
    assert result["audio"] == [
        f"take_{i:04d}_{s:.3f}-{e:.3f}.wav"
        for i, (s, e) in enumerate(segment_stubs[:2], start=1)]
    assert [c["start"] for c in result["clips"]] == [0.0, 2.5, 5.0, 8.0, 16.0]
    hist = result["histogram"]
    assert {h["count"] for h in hist} == {3, 1, 1}   # 3 short, 7 s, 12 s

    on_disk = json.loads((pdir / "preview.json").read_text())
    assert on_disk["params"] == {"source": "take.wav", "channel": "left",
                                 "energy_threshold": 40, "clips_kept": 2}
    # only the kept clips remain; the scratch conversion dir is gone
    assert sorted(p.name for p in pdir.glob("*.wav")) == result["audio"]
    assert not (pdir / "_work").exists()


def test_segment_preview_without_denoise_skips_step(tmp_path, segment_stubs,
                                                    monkeypatch):
    called = []
    monkeypatch.setattr(prepare, "denoise_file",
                        lambda src, dst: called.append(1))
    jd, root = make_job(tmp_path, {"stage": "segment", "source": "take.wav",
                                   "denoise": False})
    runner.execute(jd)
    assert called == []


def test_preview_writes_only_to_preview_dir(tmp_path, segment_stubs):
    jd, root = make_job(tmp_path, {"stage": "segment", "source": "take.wav"})
    runner.execute(jd)
    # dataset/ and the real work dirs are untouched (§2.1 consistency rule)
    assert not (root / "dataset").exists()
    assert not (root / "work" / "clips").exists()


# ------------------------------------------------------------------ denoise

def test_denoise_preview_outputs_both_sides(tmp_path, segment_stubs):
    jd, root = make_job(tmp_path, {"stage": "denoise", "source": "take.wav",
                                   "seconds": 20})
    result = runner.execute(jd)
    pdir = root / "work" / "preview" / "denoise" / jd.name
    assert result["audio"] == ["original.wav", "denoised.wav"]
    assert result["seconds"] == 20
    assert (pdir / "original.wav").exists()
    assert (pdir / "denoised.wav").exists()
    assert not (pdir / "src-48k.wav").exists()


def test_denoise_preview_caps_seconds(tmp_path, segment_stubs):
    jd, _ = make_job(tmp_path, {"stage": "denoise", "source": "take.wav",
                                "seconds": 999})
    assert runner.execute(jd)["seconds"] == 60


# ------------------------------------------------------------------ errors

def test_preview_unknown_stage(tmp_path):
    jd, _ = make_job(tmp_path, {"stage": "audition"})
    with pytest.raises(RuntimeError, match="unknown preview stage"):
        runner.execute(jd)


def test_preview_unknown_source(tmp_path):
    jd, _ = make_job(tmp_path, {"stage": "segment", "source": "../escape"})
    with pytest.raises(RuntimeError, match="not found in raw/"):
        runner.execute(jd)
