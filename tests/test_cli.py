"""CLI-level tests: tier persistence and other argument resolution."""
import json
import sys
from pathlib import Path

from piper_trainer import cli


def make_proj(root: Path, meta: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(json.dumps(
        {"name": root.name, **meta}))
    return root


def test_train_reads_saved_tier(tmp_path, capsys):
    root = make_proj(tmp_path / "p", {"tier": "low"})
    rc = cli.main(["train", str(root), "--dry-run", "--skip-validate"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--model.sample_rate" in out
    assert "16000" in out  # low tier, not the medium default


def test_explicit_tier_overrides_saved_and_persists(tmp_path, capsys):
    root = make_proj(tmp_path / "p", {"tier": "low"})
    rc = cli.main(["train", str(root), "--dry-run", "--skip-validate",
                   "--tier", "high"])
    assert rc == 0
    assert "tier changed" in capsys.readouterr().err
    assert json.loads((root / "project.json").read_text())["tier"] == "high"


def test_unsaved_tier_defaults_to_medium(tmp_path, capsys):
    root = make_proj(tmp_path / "p", {})  # pre-init project.json, no tier
    rc = cli.main(["train", str(root), "--dry-run", "--skip-validate"])
    assert rc == 0
    assert "22050" in capsys.readouterr().out  # medium sample rate


def test_prepare_reads_saved_tier_for_finalize(tmp_path, monkeypatch, capsys):
    """prepare without --tier must finalize at the saved tier's rate."""
    import sys
    import types as t
    root = make_proj(tmp_path / "p", {"tier": "low"})
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "a.wav").write_bytes(b"x")

    rates = []

    def fake_run(cmd):
        rates.append(cmd)
        if cmd[0] == "deep-filter":
            out = __import__("pathlib").Path(cmd[cmd.index("-o") + 1])
            out.mkdir(parents=True, exist_ok=True)
            for a in cmd[cmd.index("-o") + 2:]:
                (out / __import__("pathlib").Path(a).name).write_bytes(b"f")
        elif cmd[-1].endswith(".wav"):
            dst = __import__("pathlib").Path(cmd[-1])
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"f")

    class _Ev:
        def __init__(self, start, end):
            self.start, self.end = start, end

        def save(self, path):
            __import__("pathlib").Path(path).write_bytes(b"f")

    fake_auditok = t.ModuleType("auditok")
    fake_auditok.load = lambda p: t.SimpleNamespace(
        split=lambda **kw: [_Ev(0.0, 2.0), _Ev(2.5, 4.5)])
    monkeypatch.setitem(sys.modules, "auditok", fake_auditok)

    import piper_trainer.prepare as prepare_mod
    monkeypatch.setattr(prepare_mod, "_run", fake_run)
    rc = cli.main(["prepare", str(root)])
    assert rc == 0
    finalize = [c for c in rates
                if any("loudnorm" in part for part in c)]
    assert finalize and "16000" in finalize[0]  # low tier's rate


def test_validate_uses_saved_tier(tmp_path):
    root = make_proj(tmp_path / "p", {"tier": "low"})
    rc = cli.main(["validate", str(root)])
    assert rc == 1  # no metadata: error either way, but it must not crash
    # a low-tier project with a 22050 wav flags sample-rate only if the
    # saved tier is respected — covered by the rate check in validate tests


def test_serve_runs_uvicorn(monkeypatch):
    import uvicorn
    calls = {}
    monkeypatch.setattr(uvicorn, "run",
                        lambda app, host, port: calls.update(
                            app=app, host=host, port=port))
    rc = cli.main(["serve", "--host", "0.0.0.0", "--port", "8123"])
    assert rc == 0
    assert calls["host"] == "0.0.0.0"
    assert calls["port"] == 8123
    assert calls["app"] is not None  # a real ASGI app from create_app()


def test_serve_without_api_extra(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    rc = cli.main(["serve"])
    assert rc == 2
    assert "api extra" in capsys.readouterr().err
