"""Previews: job variants scoped to a sample, writing only to
work/preview/<stage>/<preview-id>/ (design doc §2, §4.5).

auditok and deep-filter are runtime-image dependencies, so the stage
primitives are monkeypatched; the manager/lifecycle behavior is covered by
test_jobs with stub runners, and the endpoints are tested against
hand-written preview.json files.
"""
from __future__ import annotations

import json
import math
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
    """Deterministic 5-clip split: 3x ~2 s, 1x ~7 s, 1x ~12 s. The converted
    and denoised outputs are real 16-bit WAVs (a small sine) so the preview's
    level measurement exercises the same code it runs in production."""
    def fake_wav(path, amp=1000):
        import math
        import wave

        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(b"".join(
                int(amp * math.sin(2 * math.pi * 220 * i / 48000))
                .to_bytes(2, "little", signed=True)
                for i in range(4800)))

    def fake_convert(src, dst, channel=None):
        fake_wav(dst)

    def fake_denoise(src, dst_dir):
        out = dst_dir / src.name
        fake_wav(out)
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

    # level of the real wav the stub wrote: a 1000/32768 sine
    assert result["level"]["peak_dbfs"] == pytest.approx(
        20 * math.log10(1000 / 32768), abs=0.2)
    assert result["level"]["speech_dbfs"] == pytest.approx(
        result["level"]["rms_dbfs"], abs=1.0)  # constant tone: no gaps

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


def test_segment_all_preview_reports_per_source(tmp_path, segment_stubs,
                                                monkeypatch):
    """Batch form: one row per raw source, zeros named, one bad file an
    error row instead of a dead job."""
    jd, root = make_job(tmp_path, {"stage": "segment-all",
                                   "energy_threshold": 40})
    names = ("take.wav", "quiet.wav", "bad.wav")
    for n in names:
        (root / "raw" / n).write_bytes(b"RIFF-fake")
    monkeypatch.setattr(prepare, "sources",
                        lambda project: [{"name": n} for n in names])

    stub_split = prepare.split_audio

    def fake_split(src, out_dir, stem=None, **kwargs):
        if "quiet" in src.name:  # split sees the converted quiet-48k.wav
            return []  # clear audio, but under the threshold cliff
        return stub_split(src, out_dir, stem=stem or src.stem, **kwargs)

    stub_convert = prepare.convert_one

    def fake_convert(src, dst, channel=None):
        if src.name == "bad.wav":
            raise RuntimeError("corrupt input")
        return stub_convert(src, dst, channel=channel)

    monkeypatch.setattr(prepare, "split_audio", fake_split)
    monkeypatch.setattr(prepare, "convert_one", fake_convert)

    result = runner.execute(jd)

    rows = {r["source"]: r for r in result["per_source"]}
    assert set(rows) == set(names)
    assert rows["take.wav"]["clips"] == 5
    assert rows["take.wav"]["error"] is None
    assert rows["take.wav"]["seconds"] == pytest.approx(25.0)
    # quiet.wav still gets a measured level: the UI can say WHY it was empty
    assert rows["quiet.wav"]["clips"] == 0
    # the stub's tone: RMS of a sine is peak/sqrt(2)
    assert rows["quiet.wav"]["level"]["speech_dbfs"] == pytest.approx(
        20 * math.log10(1000 / math.sqrt(2) / 32768), abs=0.5)
    assert rows["bad.wav"]["error"] == "corrupt input"
    assert result["clip_count"] == 5
    assert result["zeros"] == ["bad.wav", "quiet.wav"]
    assert result["audio"] == []
    # counts only, no playable clips: the scratch tree is gone
    assert not (root / "work" / "preview" / "segment-all" / jd.name
                / "_work").exists()
    on_disk = json.loads(
        (root / "work" / "preview" / "segment-all" / jd.name
         / "preview.json").read_text())
    assert on_disk["params"] == {"energy_threshold": 40}


def test_segment_all_preview_without_sources_fails(tmp_path, monkeypatch):
    jd, _root = make_job(tmp_path, {"stage": "segment-all"})
    monkeypatch.setattr(prepare, "sources", lambda project: [])
    with pytest.raises(RuntimeError, match="no sources in raw/"):
        runner.execute(jd)


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
    # audition ships now (§2.3); finalize is the next unshipped stage
    jd, _ = make_job(tmp_path, {"stage": "finalize"})
    with pytest.raises(RuntimeError, match="unknown preview stage"):
        runner.execute(jd)


def test_preview_unknown_source(tmp_path):
    jd, _ = make_job(tmp_path, {"stage": "segment", "source": "../escape"})
    with pytest.raises(RuntimeError, match="not found in raw/"):
        runner.execute(jd)


# -------------------------------------------------------------------- train

@pytest.fixture
def quiet_sleep(monkeypatch):
    # the rate parser retries the manager's log tee for up to a second;
    # tests do not need to actually wait those out
    monkeypatch.setattr(runner.time, "sleep", lambda _s: None)


def add_rows(root, n=8):
    root.joinpath("dataset").mkdir(exist_ok=True)
    root.joinpath("dataset", "metadata.csv").write_text(
        "".join(f"clip_{i:03d}| spoken row {i}\n" for i in range(n)))


@pytest.fixture
def train_stubs(monkeypatch):
    """No torch, no lightning, no validate here: capture the command a
    real run would get and always succeed."""
    commands: list[list[str]] = []
    monkeypatch.setattr(runner, "validate_dataset", lambda *a, **k: [])
    monkeypatch.setattr(runner.train_mod, "run",
                        lambda cmd: commands.append(cmd) or 0)
    return commands


def test_train_preview_runs_capped_isolated_command(
        tmp_path, train_stubs, quiet_sleep):
    jd, root = make_job(tmp_path, {
        "stage": "train", "warmstart": "fake.ckpt",
        "max_epochs": 100, "batch_size": 4, "steps": 50})
    (root / "fake.ckpt").write_bytes(b"ckpt")
    add_rows(root)
    # the manager tees the trainer's progress bar into the job log; a
    # pre-written bar stands in for it, so the preview reads its rate the
    # way it does in production
    (jd / "log.txt").write_text(
        "Epoch 0:  40%|████| 20/50 [00:09<00:14, 1.40it/s]\n"
        "Epoch 0: 100%|████| 50/50 [00:23<00:00, 2.17it/s]\n")
    result = runner.execute(jd)

    cmd = train_stubs[0]
    # capped at --trainer.max_steps; the ceiling stays the resolved target
    i = cmd.index("--trainer.max_steps")
    assert cmd[i + 1] == "50"
    # every output Lightning writes lands in the preview dir (§2.1)
    j = cmd.index("--trainer.default_root_dir")
    assert cmd[j + 1] == str(root / "work" / "preview" / "train" / jd.name)
    k = cmd.index("--trainer.logger.dict_kwargs")
    assert str(root / "work" / "preview" / "train" / jd.name) in cmd[k + 1]
    assert cmd[cmd.index("--model.warmstart_ckpt") + 1] \
        == str(root / "fake.ckpt")

    assert result["mode"] == "warmstart"
    assert result["steps_planned"] == 50
    assert result["clips"] == 8
    assert result["steps_per_epoch"] == 2
    # rate from the bar, so the projection is the steady-state one
    assert result["rate_source"] == "progress-bar"
    assert result["steps_per_sec"] == 2.17
    assert result["target_epochs"] == 100
    assert result["remaining_epochs"] == 100
    assert result["projected_seconds"] == round(100 * 2 / 2.17)
    assert "note" not in result

    # §2.1: nothing leaked into the real tree or project.json
    assert not (root / "runs-medium").exists()
    assert "target_epochs" not in json.loads(
        (root / "project.json").read_text())


def test_train_preview_resume_adds_global_steps(
        tmp_path, monkeypatch, quiet_sleep):
    jd, root = make_job(tmp_path, {
        "stage": "train", "resume": "auto", "add_epochs": 50, "steps": 20,
        "batch_size": 4, "skip_validate": True})
    ck = (root / "runs-medium" / "lightning_logs" / "version_0"
          / "checkpoints" / "last.ckpt")
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"ckpt")
    add_rows(root)

    import time

    commands: list[list[str]] = []
    monkeypatch.setattr(runner.train_mod, "latest_checkpoint",
                        lambda project, tier: ck)
    monkeypatch.setattr(runner.train_mod, "checkpoint_epoch",
                        lambda path: 100)
    monkeypatch.setattr(runner.train_mod, "checkpoint_global_step",
                        lambda path: 3100)

    def slow_run(cmd):
        # burn real wall clock so the no-bar fallback measures a sane rate
        commands.append(cmd)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.4:
            pass
        return 0

    monkeypatch.setattr(runner.train_mod, "run", slow_run)

    result = runner.execute(jd)
    # max_steps is a global counter: 3100 consumed + 20 preview steps
    i = commands[0].index("--trainer.max_steps")
    assert commands[0][i + 1] == "3120"
    assert result["mode"] == "resume"
    assert result["target_epochs"] == 150
    assert result["remaining_epochs"] == 50  # 150 ceiling - 100 at checkpoint
    # empty log -> wall-clock rate, and the result says so honestly
    assert result["rate_source"] == "wall-clock"
    assert 0 < result["steps_per_sec"] < 1000
    assert result["projected_seconds"] > 0
    assert "upper bound" in result["note"]


def test_train_preview_exit_code_raises(tmp_path, monkeypatch, quiet_sleep):
    jd, root = make_job(tmp_path, {"stage": "train", "max_epochs": 10,
                                   "batch_size": 4, "skip_validate": True})
    add_rows(root)
    monkeypatch.setattr(runner.train_mod, "run", lambda cmd: 1)
    with pytest.raises(RuntimeError,
                       match="the full run would have failed the same way"):
        runner.execute(jd)


def test_step_rate_parses_last_bar_frame(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        "Epoch 0:  40%|████| 20/50 [00:09<00:14, 1.40it/s]\n"
        "Epoch 0: 100%|████| 50/50 [00:23<00:00, 2.17it/s]\n")
    assert runner._step_rate_from_log(log) == 2.17
    assert runner._step_rate_from_log(tmp_path / "missing.txt") is None
