"""Checkpoint catalog: bundled snapshot, live-first with fallback."""
from __future__ import annotations

import pytest

from piper_trainer.api import catalog


@pytest.fixture(autouse=True)
def fresh_cache():
    catalog._cache = {"at": 0.0, "data": None}
    yield
    catalog._cache = {"at": 0.0, "data": None}


def test_snapshot_loads():
    snap = catalog.snapshot()
    assert snap["repo"] == catalog.REPO
    assert snap["generated"]
    assert isinstance(snap["languages"], dict)
    assert isinstance(snap["files"], dict)


def test_snapshot_has_expected_shape():
    snap = catalog.snapshot()
    # en -> en_GB -> alan exists with at least one quality
    assert "alan" in snap["languages"]["en"]["en_GB"]
    quals = snap["languages"]["en"]["en_GB"]["alan"]
    assert quals == sorted(quals)
    for q in quals:
        assert f"en/en_GB/alan/{q}" in snap["files"]


def test_fallback_to_snapshot_when_live_fails():
    def boom(url, timeout):
        raise RuntimeError("offline")
    cat = catalog.catalog(fetch=boom)
    assert cat["source"] == "snapshot"
    assert cat["generated"] == catalog.snapshot()["generated"]
    assert "alan" in cat["languages"]["en"]["en_GB"]


def test_live_wins_when_available():
    def fake(url, timeout):
        assert "recursive=true" in url
        return [
            {"type": "directory", "path": "en"},
            {"type": "directory", "path": "en/en_US"},
            {"type": "directory", "path": "en/en_US/alan"},
            {"type": "directory", "path": "en/en_US/alan/medium"},
            {"type": "file", "path": "en/en_US/alan/medium/config.json"},
        ]
    cat = catalog.catalog(fetch=fake)
    assert cat["source"] == "live"
    assert cat["languages"]["en"]["en_US"]["alan"] == ["medium"]


def test_catalog_is_cached():
    calls = []

    def fake(url, timeout):
        calls.append(url)
        raise RuntimeError("offline")
    clock = {"t": 100.0}
    catalog.catalog(fetch=fake, clock=lambda: clock["t"])
    catalog.catalog(fetch=fake, clock=lambda: clock["t"])
    assert len(calls) == 1
    clock["t"] = 100.0 + catalog._CACHE_TTL + 1
    catalog.catalog(fetch=fake, clock=lambda: clock["t"])
    assert len(calls) == 2


def test_detail_from_snapshot():
    def boom(url, timeout):
        raise RuntimeError("offline")
    snap = catalog.snapshot()
    path = next(iter(snap["files"]))
    d = catalog.detail(path, fetch=boom)
    assert d["source"] == "snapshot"
    assert d["files"] == sorted(snap["files"][path])


def test_detail_rejects_bad_paths():
    with pytest.raises(ValueError):
        catalog.detail("../etc/passwd", fetch=lambda u, t: [])
    with pytest.raises(ValueError):
        catalog.detail("only-two", fetch=lambda u, t: [])


def test_detail_unknown_path_is_keyerror():
    def boom(url, timeout):
        raise RuntimeError("offline")
    with pytest.raises(KeyError):
        catalog.detail("en/en_US/does-not-exist/medium", fetch=boom)
