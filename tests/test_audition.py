"""Audition preview (§2.3): one held-out sentence through N checkpoints.

The real handler runs a torch ONNX export plus the piper CLI per take —
minutes of work neither CI nor a unit test should do — so export and say
are stubbed at the seams the handler actually calls, and the tests pin
selection, naming, output placement, and the envelope instead.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from piper_trainer import say
from piper_trainer.api import runner
from piper_trainer.config import Project


def make_project(tmp_path, name="proj") -> Project:
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(json.dumps({"name": name}))
    return Project.load(root)


def make_job(tmp_path, params):
    root = make_project(tmp_path).root
    jd = root / "jobs" / "20260905T000000Z-preview-audition"
    jd.mkdir(parents=True, exist_ok=True)
    (jd / "job.json").write_text(json.dumps({
        "id": jd.name, "kind": "preview", "project": "proj",
        "params": params, "state": "queued", "pid": None}))
    (jd / "log.txt").touch()
    return jd


def make_ckpt(root, name, mtime):
    p = root / "runs-medium" / "lightning_logs" / "version_0" / \
        "checkpoints" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"ckpt-bytes")
    os.utime(p, (mtime, mtime))
    return p


@pytest.fixture
def synth_stubs(monkeypatch):
    """Export writes a stub pair wherever it is told (and records the
    out_dir); say returns recognizable wav bytes (and records the text)."""
    exports: list[dict] = []
    says: list[dict] = []

    def fake_export(project, tier, checkpoint, voice_name=None,
                    out_dir=None, **kwargs):
        assert out_dir is not None, "audition must not export into out/"
        out_dir.mkdir(parents=True, exist_ok=True)
        onnx = out_dir / f"{voice_name}.onnx"
        onnx.write_bytes(b"RIFF-stub-onnx")
        (out_dir / f"{voice_name}.onnx.json").write_text("{}")
        exports.append({"checkpoint": checkpoint, "stem": voice_name,
                        "out_dir": out_dir})
        return onnx, out_dir / f"{voice_name}.onnx.json"

    def fake_synth(onnx_path, json_path, text, **kwargs):
        says.append({"onnx": onnx_path, "text": text})
        return b"RIFF-take-wav"

    monkeypatch.setattr(runner.export_mod, "export", fake_export)
    monkeypatch.setattr(runner.say_mod, "synthesize", fake_synth)
    return exports, says


def test_default_takes_the_newest_three(tmp_path, synth_stubs):
    exports, says = synth_stubs
    proj = make_project(tmp_path)
    for i, mtime in enumerate([1000, 2000, 3000, 4000, 5000]):
        make_ckpt(proj.root, f"epoch={i}-step={i * 10}.ckpt", mtime)

    result = runner.execute(make_job(tmp_path, {"stage": "audition"}))

    assert len(result["takes"]) == 3
    # newest first, each a numbered take with the epoch from the filename
    assert [t["epoch"] for t in result["takes"]] == [4, 3, 2]
    assert [t["take"] for t in result["takes"]] == [1, 2, 3]
    assert [t["stem"] for t in result["takes"]] == [
        "take1-e4", "take2-e3", "take3-e2"]
    # every export went into the preview dir, never out/
    assert all(e["out_dir"] == proj.root / "work" / "preview" / "audition"
               / "20260905T000000Z-preview-audition" for e in exports)
    assert not (proj.out / "take1-e4.onnx").exists()
    assert not list(proj.out.glob("*.onnx"))
    # the same held-out sentence through every take
    assert {s["text"] for s in says} == {say.DEFAULT_TEXT}
    # wavs and the envelope land in the preview dir
    pdir = proj.root / "work" / "preview" / "audition" \
        / "20260905T000000Z-preview-audition"
    assert (pdir / "take1-e4.wav").read_bytes() == b"RIFF-take-wav"
    env = json.loads((pdir / "preview.json").read_text())
    assert env["stage"] == "audition"
    assert env["result"]["text"] == say.DEFAULT_TEXT


def test_explicit_checkpoints_and_text(tmp_path, synth_stubs):
    exports, says = synth_stubs
    proj = make_project(tmp_path)
    young = make_ckpt(proj.root, "epoch=7-step=70.ckpt", 2000)
    old = make_ckpt(proj.root, "epoch=1-step=10.ckpt", 1000)

    result = runner.execute(make_job(tmp_path, {
        "stage": "audition",
        "checkpoints": [str(old.relative_to(proj.root)),
                        str(young.relative_to(proj.root))],
        "text": "It can only be attributable to human error."}))

    assert [t["epoch"] for t in result["takes"]] == [1, 7]
    assert [e["checkpoint"] for e in exports] == [old, young]
    assert {s["text"] for s in says} == \
        {"It can only be attributable to human error."}


def test_missing_checkpoint_fails(tmp_path, synth_stubs):
    make_project(tmp_path)
    with pytest.raises(RuntimeError, match="checkpoint not found"):
        runner.execute(make_job(
            tmp_path, {"stage": "audition", "checkpoints": ["nope.ckpt"]}))


def test_no_checkpoints_at_all_fails(tmp_path, synth_stubs):
    proj = make_project(tmp_path)  # no runs-<tier> anywhere
    with pytest.raises(RuntimeError, match="run checkpoints to audition"):
        runner.execute(make_job(tmp_path, {"stage": "audition"}))
    assert not (proj.root / "work" / "preview" / "audition").exists()


def test_limit_clamps_and_empty_text_rejected(tmp_path, synth_stubs):
    proj = make_project(tmp_path)
    for i in range(6):
        make_ckpt(proj.root, f"epoch={i}-step={i}.ckpt", 1000 + i)

    result = runner.execute(make_job(
        tmp_path, {"stage": "audition", "limit": 99}))
    assert len(result["takes"]) == 5  # hard ceiling: five takes, not six

    with pytest.raises(RuntimeError, match="audition text is empty"):
        runner.execute(make_job(tmp_path,
                                {"stage": "audition", "text": "   "}))


def test_export_out_dir_redirect(tmp_path, monkeypatch):
    """export(out_dir=...) lands the pair in out_dir but still reads the
    training config from out/ — where the train run wrote it."""
    from piper_trainer import export as export_mod

    def fake_subprocess(cmd, **kwargs):
        # piper.train.export_onnx would create the .onnx; honor --output-file
        Path(cmd[cmd.index("--output-file") + 1]).write_bytes(b"RIFF-stub")

    monkeypatch.setattr(export_mod.subprocess, "run", fake_subprocess)
    proj = make_project(tmp_path)
    proj.ensure()  # out/ must exist for the training config
    generated = proj.out / "proj-medium.config.json"
    generated.write_text(json.dumps({
        "num_symbols": 256, "num_speakers": 1, "phoneme_id_map": {},
        "audio": {"sample_rate": 22050}}))

    scratch = tmp_path / "scratch" / "audition" / "x"
    onnx, js = export_mod.export(proj, "medium", tmp_path / "c.ckpt",
                                 voice_name="take1-e9", out_dir=scratch)
    assert onnx == scratch / "take1-e9.onnx"
    assert onnx.exists()
    cfg = json.loads(js.read_text())
    assert cfg["dataset"] == "take1-e9"  # patched fields still applied
    assert cfg["audio"]["quality"] == "medium"
