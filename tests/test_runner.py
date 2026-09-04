"""Runner stage execution, exercised in-process (no subprocess)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from piper_trainer import prepare
from piper_trainer.api import runner


def make_job(tmp_path, kind, params=None):
    """A queued job on disk, exactly as the manager writes it."""
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    (root / "project.json").write_text(json.dumps({"name": "proj"}))
    job_dir = root / "jobs" / f"20260901T000000Z-{kind}-test"
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(json.dumps({
        "id": job_dir.name, "kind": kind, "project": "proj",
        "params": params or {}, "state": "queued", "pid": None,
    }))
    (job_dir / "log.txt").touch()
    return job_dir


def test_ingest_sanitizes_moves_and_probes(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "probe", lambda p: {
        "codec_name": "pcm_s16le", "sample_rate": "48000",
        "channels": "1", "duration": "3.25"})
    jd = make_job(tmp_path, "ingest", {"source_type": "upload"})
    incoming = jd / "incoming"
    incoming.mkdir()
    (incoming / "my take (1).wav").write_bytes(b"a")
    (incoming / "clean.wav").write_bytes(b"b")

    result = runner.execute(jd)

    assert result["added"] == 2
    raw = tmp_path / "proj" / "raw"
    stored = sorted(p.name for p in raw.iterdir())
    assert stored == ["clean.wav", "my_take_1_.wav"]
    by_name = {r["original"]: r["stored_as"] for r in result["renamed"]}
    assert by_name == {"my take (1).wav": "my_take_1_.wav"}
    assert result["files"][0]["codec"] == "pcm_s16le"
    # staged files are consumed, not copied
    assert list(incoming.iterdir()) == []


def test_ingest_collision_gets_index(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "probe", lambda p: {})
    jd = make_job(tmp_path, "ingest", {"source_type": "upload"})
    incoming = jd / "incoming"
    incoming.mkdir()
    (incoming / "a b.wav").write_bytes(b"a")   # sanitizes to a_b.wav
    (incoming / "a_b.wav").write_bytes(b"b")   # same sanitized name

    result = runner.execute(jd)
    raw = tmp_path / "proj" / "raw"
    names = sorted(p.name for p in raw.iterdir())
    assert names == ["a_b-1.wav", "a_b.wav"]  # '-' sorts before '.'


def test_ingest_rejects_unknown_source_types(tmp_path):
    """Step 2 added url / media-site / hf-dataset (§2.5.2–2.5.4); what
    must still be refused is anything outside the known set."""
    jd = make_job(tmp_path, "ingest", {"source_type": "carrier-pigeon"})
    with pytest.raises(RuntimeError, match="unknown source_type"):
        runner.execute(jd)


def test_ingest_empty_staging_fails(tmp_path):
    jd = make_job(tmp_path, "ingest", {"source_type": "upload"})
    with pytest.raises(RuntimeError, match="no files"):
        runner.execute(jd)


def test_fetch_checkpoint_rejects_bad_paths(tmp_path):
    for bad in ("../../etc", "en", "en/en_US", "en/en_US/alan/medium/x"):
        jd = make_job(tmp_path, "fetch-checkpoint", {"catalog_path": bad})
        with pytest.raises(RuntimeError, match="catalog checkpoint path"):
            runner.execute(jd)


def test_validate_job_returns_findings_json(tmp_path):
    jd = make_job(tmp_path, "validate", {})
    result = runner.execute(jd)
    assert isinstance(result["findings"], list)
    assert result["errors"] == result["errors"]  # key present
    for f in result["findings"]:
        assert set(f) >= {"level", "code", "message", "action"}


def test_clean_dry_run_returns_plan(tmp_path):
    jd = make_job(tmp_path, "clean", {"apply": False})
    result = runner.execute(jd)
    assert "plan" in result
    assert "stats" not in result


def test_unknown_kind_fails(tmp_path):
    jd = make_job(tmp_path, "prepare")
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "explode", "project": "proj",
        "params": {}, "state": "running", "pid": None}))
    with pytest.raises(RuntimeError, match="unknown job kind"):
        runner.execute(jd)


def test_train_add_epochs_implies_resume_auto(tmp_path, monkeypatch):
    """A UI 'train N more' job (add_epochs, no resume) must resume from the
    latest checkpoint: the ceiling arithmetic needs its epoch counter.
    Regression: add_epochs alone hit '--add-epochs needs a checkpoint'."""
    jd = make_job(tmp_path, "train",
                  {"add_epochs": 10, "skip_validate": True})
    ckpt = tmp_path / "proj" / "fake-last.ckpt"
    ckpt.write_bytes(b"x")
    seen = {}
    monkeypatch.setattr(runner.train_mod, "latest_checkpoint",
                        lambda project, tier: ckpt)
    monkeypatch.setattr(runner.train_mod, "checkpoint_epoch",
                        lambda c: 100)
    monkeypatch.setattr(
        runner.train_mod, "build_command",
        lambda project, **kw: seen.update(kw) or ["echo", "train"])
    monkeypatch.setattr(runner.train_mod, "run", lambda cmd: 0)

    result = runner.execute(jd)

    assert seen["resume"] == ckpt
    assert seen["max_epochs"] == 110  # 100 in the checkpoint + 10 more
    assert result["max_epochs"] == 110


def test_process_entry_emits_result_and_exits_zero(tmp_path):
    """The real `python -m` entry, as the job manager spawns it."""
    jd = make_job(tmp_path, "validate", {})
    env = dict(os.environ)
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "piper_trainer.api.runner", str(jd)],
        capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "##RESULT " in proc.stdout
    payload = json.loads(proc.stdout.split("##RESULT ", 1)[1]
                         .splitlines()[0])
    assert "findings" in payload


def test_main_reports_failure_as_result(tmp_path, capsys):
    """main() turns a stage exception into a ##RESULT error + exit 1, so the
    jobs table shows the reason instead of a bare 'exited with code 1'."""
    jd = make_job(tmp_path, "export")   # no checkpoint exists -> RuntimeError
    rc = runner.main(["piper_trainer.api.runner", str(jd)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "##RESULT" in out
    assert "run a train job first" in out


def test_main_usage_error(tmp_path, capsys):
    assert runner.main([]) == 2
    assert "usage:" in capsys.readouterr().err
