"""settings: env-first workspace resolution with a bare-host fallback."""
from __future__ import annotations

from piper_trainer.api import settings


def test_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPER_WORKSPACE", str(tmp_path / "ws"))
    assert settings.workspace() == tmp_path / "ws"


def test_default_when_it_exists(tmp_path, monkeypatch):
    """The container layout: /workspace exists, so it stays the default."""
    monkeypatch.delenv("PIPER_WORKSPACE", raising=False)
    ws = tmp_path / "container-like"
    ws.mkdir()
    monkeypatch.setattr(settings, "DEFAULT_WORKSPACE", str(ws))
    assert settings.workspace() == ws


def test_bare_host_falls_back_to_cwd(tmp_path, monkeypatch):
    """Review finding 14: on a pip-installed host without /workspace, the
    old default pointed at a directory that does not exist and the UI
    showed an empty project list with no explanation."""
    monkeypatch.delenv("PIPER_WORKSPACE", raising=False)
    monkeypatch.setattr(settings, "DEFAULT_WORKSPACE",
                        str(tmp_path / "missing"))
    monkeypatch.chdir(tmp_path)
    assert settings.workspace() == tmp_path / "workspace"
