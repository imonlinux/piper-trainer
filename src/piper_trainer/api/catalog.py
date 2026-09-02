"""Checkpoint catalog (design doc §3): the HF tree, fetched live with a
short timeout, falling back to a snapshot bundled with the package so the
picker works offline and on first run.

The snapshot is refreshed deliberately with
`python scripts/build_catalog_snapshot.py` — a maintenance action, not an
automatic pull (resolved decision #6).
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from importlib import resources

REPO = "rhasspy/piper-checkpoints"
_TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
_CACHE_TTL = 600.0  # seconds; the repo changes rarely, sessions shorter

_PATH_RE = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+){2,3}$")


def _http_json(url: str, timeout: float):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def snapshot() -> dict:
    """The bundled snapshot: {generated, repo, languages, files}."""
    ref = resources.files("piper_trainer").joinpath(
        "api/catalog_snapshot.json")
    return json.loads(ref.read_text())


def live_languages(timeout: float, fetch=_http_json) -> dict:
    """{family: {locale: {voice: [qualities]}}} from the live HF tree."""
    entries = fetch(f"{_TREE_URL}?recursive=true", timeout)
    languages: dict[str, dict] = {}
    for e in entries:
        parts = e.get("path", "").split("/")
        if e.get("type") == "directory" and len(parts) == 4:
            fam, loc, voice, qual = parts
            quals = languages.setdefault(fam, {}).setdefault(loc, {}) \
                             .setdefault(voice, [])
            quals.append(qual)
    for fam in languages.values():
        for loc in fam.values():
            for voice, quals in loc.items():
                loc[voice] = sorted(quals)
    if not languages:
        raise RuntimeError("HF tree listing came back empty")
    return languages


_cache: dict = {"at": 0.0, "data": None}


def catalog(timeout: float = 3.0, fetch=_http_json,
            clock=time.monotonic) -> dict:
    """Catalog for the picker: live first, snapshot on any failure."""
    now = clock()
    if _cache["data"] is not None and now - _cache["at"] < _CACHE_TTL:
        return _cache["data"]
    try:
        data = {"source": "live", "generated": None,
                "languages": live_languages(timeout, fetch)}
    except Exception:  # noqa: BLE001 — any failure falls back
        snap = snapshot()
        data = {"source": "snapshot", "generated": snap["generated"],
                "languages": snap["languages"], "files": snap["files"]}
    _cache["at"] = now
    _cache["data"] = data
    return data


def detail(path: str, timeout: float = 3.0, fetch=_http_json) -> dict:
    """One node of the catalog: the qualities under a voice, or the files
    under a quality directory (config.json, MODEL_CARD, checkpoints)."""
    if not _PATH_RE.match(path) or ".." in path:
        raise ValueError(f"not a catalog path: {path!r}")
    try:
        entries = fetch(f"{_TREE_URL}/{path}", timeout)
        directories = [e["path"].rsplit("/", 1)[1]
                       for e in entries if e.get("type") == "directory"]
        files = [e["path"].rsplit("/", 1)[1]
                 for e in entries if e.get("type") == "file"]
        return {"source": "live", "path": path,
                "directories": sorted(directories), "files": sorted(files)}
    except Exception:  # noqa: BLE001
        snap = snapshot()
        files = snap["files"].get(path)
        if files is None:
            raise KeyError(path) from None
        return {"source": "snapshot", "path": path, "directories": [],
                "files": sorted(files)}
