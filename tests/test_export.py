"""Tests for export: verify() and the .onnx.json config patch."""
import json
from pathlib import Path

from piper_trainer import export as export_mod
from piper_trainer.config import Project


def test_verify_rejects_stem_mismatch(tmp_path):
    onnx = tmp_path / "a.onnx"
    onnx.write_bytes(b"x")
    js = tmp_path / "b.onnx.json"
    js.write_text(json.dumps({"dataset": "b", "audio": {"quality": "medium"},
                              "language": {"code": "en_US"}}))
    problems = export_mod.verify(onnx, js)
    assert any("stem mismatch" in p for p in problems)
    assert any("dataset field" in p for p in problems)


def test_verify_rejects_missing_language_code(tmp_path):
    onnx = tmp_path / "a.onnx"
    onnx.write_bytes(b"x")
    js = tmp_path / "a.onnx.json"
    js.write_text(json.dumps({"dataset": "a", "audio": {"quality": "medium"},
                              "language": {}}))
    problems = export_mod.verify(onnx, js)
    assert any("language.code" in p for p in problems)


def test_verify_rejects_missing_embedding_fields(tmp_path):
    onnx = tmp_path / "a.onnx"
    onnx.write_bytes(b"x")
    js = tmp_path / "a.onnx.json"
    js.write_text(json.dumps({"dataset": "a", "audio": {"quality": "medium"},
                              "language": {"code": "en_US"}}))
    problems = export_mod.verify(onnx, js)
    for key in ("num_symbols", "num_speakers", "phoneme_id_map"):
        assert any(key in p for p in problems)


def test_export_patches_config_and_preserves_embedding_fields(
        tmp_path, monkeypatch):
    proj = Project(root=tmp_path, name="marvin")
    proj.ensure()

    # piper.train.export_onnx would create the .onnx; simulate its output
    monkeypatch.setattr(export_mod.subprocess, "run", lambda *a, **k: None)

    generated = proj.out / f"{proj.name}-medium.config.json"
    original = {
        "num_symbols": 256,
        "num_speakers": 1,
        "phoneme_id_map": {"P": 0, "H": 1},
        "audio": {"sample_rate": 22050},
    }
    generated.write_text(json.dumps(original))

    onnx_path, json_path = export_mod.export(
        proj, "medium", tmp_path / "ckpt.ckpt", espeak_voice="en-gb")

    assert onnx_path == proj.out / "marvin-medium.onnx"
    out = json.loads(json_path.read_text())

    # the three fields piper1-gpl omits are added
    assert out["dataset"] == "marvin-medium"
    assert out["audio"]["quality"] == "medium"
    assert out["audio"]["sample_rate"] == 22050
    assert out["language"]["code"] == "en_GB"

    # trained embedding metadata is untouched
    assert out["num_symbols"] == 256
    assert out["num_speakers"] == 1
    assert out["phoneme_id_map"] == {"P": 0, "H": 1}


def test_export_voice_name_overrides_stem(tmp_path, monkeypatch):
    proj = Project(root=tmp_path, name="marvin")
    proj.ensure()
    monkeypatch.setattr(export_mod.subprocess, "run", lambda *a, **k: None)
    generated = proj.out / f"{proj.name}-medium.config.json"
    generated.write_text(json.dumps({"num_symbols": 256, "num_speakers": 1,
                                     "phoneme_id_map": {},
                                     "audio": {"sample_rate": 22050}}))
    onnx_path, json_path = export_mod.export(
        proj, "medium", tmp_path / "c.ckpt", voice_name="en_gb-marvin-medium",
        espeak_voice="en-gb")
    assert onnx_path.name == "en_gb-marvin-medium.onnx"
    out = json.loads(json_path.read_text())
    assert out["dataset"] == "en_gb-marvin-medium"
