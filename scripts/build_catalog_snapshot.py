"""Fetch the HF checkpoint catalog and write the bundled snapshot.

Run inside a container or any environment with network access:

    python scripts/build_catalog_snapshot.py

The snapshot is committed as package data so the checkpoint picker works
offline and on first run (design doc §3.5: fetch live, fall back to a
bundled snapshot). Re-run it deliberately when the catalog should be
refreshed — it is a build/maintenance action, not an automatic one.
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = "rhasspy/piper-checkpoints"
OUT = Path(__file__).resolve().parents[1] / "src" / "piper_trainer" / "api" / "catalog_snapshot.json"


def fetch(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def main() -> None:
    entries = fetch(
        f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true")

    languages: dict[str, dict] = {}
    files: dict[str, list[str]] = {}
    for e in entries:
        parts = e["path"].split("/")
        if e["type"] == "directory" and len(parts) == 4:
            fam, loc, voice, qual = parts
            quals = languages.setdefault(fam, {}).setdefault(loc, {}) \
                             .setdefault(voice, [])
            quals.append(qual)
        elif e["type"] == "file" and len(parts) == 5:
            files.setdefault("/".join(parts[:4]), []).append(parts[4])

    for fam in languages.values():
        for loc in fam.values():
            for voice, quals in loc.items():
                loc[voice] = sorted(quals)

    snap = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "repo": REPO, "languages": languages, "files": files}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    print(f"  languages: {len(languages)}  voice+quality dirs: {len(files)}")


if __name__ == "__main__":
    main()
