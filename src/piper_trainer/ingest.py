"""Source acquisition for ingest jobs (design doc §2.5): url, media-site
(yt-dlp) and hf-dataset. Upload (§2.5.1) is different — the API stages
those bytes itself — but every fetched variant lands files in the job's
staged incoming/ dir, where the shared _ingest tail sanitizes, moves to
raw/ and ffprobes. That keeps the job summary identical across source
types; only acquisition differs.

The subprocess/network seams (urlopen, yt-dlp, snapshot_download) sit
behind small functions so tests can exercise the pure parts — command
building, content-type gating, filename derivation, metadata column
detection — without network or yt-dlp installed.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from .prepare import AUDIO_EXT, sanitize_stem

CHUNK = 1 << 20

# Content types a direct URL may serve. text/html is the important
# refusal: a 404 landing page must not land in raw/ wearing a .mp3 name.
ALLOWED_TYPE_PREFIXES = ("audio/", "video/")
ALLOWED_TYPES_EXACT = ("application/octet-stream", "application/ogg")

TEXT_COLUMNS = ("text", "transcript", "sentence",
                "normalized_transcription", "target")
FILE_COLUMNS = ("file_name", "filename", "file", "path", "audio")


def content_type_ok(ctype: str) -> bool:
    return (ctype.startswith(ALLOWED_TYPE_PREFIXES)
            or ctype in ALLOWED_TYPES_EXACT)


def url_filename(url: str, disposition: str | None, ctype: str) -> str:
    """Pick the stored filename: Content-Disposition wins, then the URL
    basename, then a guess from the content type. Raises when nothing
    yields an audio extension we process."""
    name = ""
    if disposition:
        m = re.search(r'filename\*?=(?:"([^"]+)"|([^;]+))', disposition)
        if m:
            name = (m.group(1) or m.group(2)).strip()
    if not name:
        name = Path(urlsplit(url).path).name
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    if suffix not in AUDIO_EXT:
        suffix = ""
        guessed = _guessed_ext(ctype)
        if guessed:
            suffix = guessed
    if not suffix or suffix not in AUDIO_EXT:
        raise RuntimeError(
            f"cannot determine an audio extension for {url!r} "
            f"(content-type {ctype!r}); name a file with a known audio "
            f"extension or serve a recognisable content type")
    return sanitize_stem(stem) + suffix


def _guessed_ext(ctype: str) -> str:
    import mimetypes
    ext = mimetypes.guess_extension(ctype.split(";")[0].strip())
    return ext.lower() if ext else ""


def fetch_url(url: str, dest: Path, emit) -> list[Path]:
    """Plain HTTP(S) GET of one media file (§2.5.2). Identical to upload
    once the bytes land; streams so large files never buffer whole."""
    scheme = urlsplit(url).scheme
    if scheme not in ("http", "https"):
        raise RuntimeError(f"unsupported url scheme {scheme!r} (http/https only)")
    req = urllib.request.Request(url, headers={"User-Agent": "piper-trainer-ingest"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except OSError as exc:
        raise RuntimeError(f"fetch failed: {exc}") from exc
    with resp:
        ctype = resp.headers.get_content_type()
        if not content_type_ok(ctype):
            raise RuntimeError(
                f"refusing content-type {ctype!r} — not a media file. "
                "(An HTML error page lands here when the URL is wrong.)")
        name = url_filename(
            url, resp.headers.get("Content-Disposition"), ctype)
        total = int(resp.headers.get("Content-Length") or 0)
        out = dest / name
        done = 0
        with out.open("wb") as fh:
            while chunk := resp.read(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    emit("PROGRESS", {"current": round(done / total * 100),
                                      "total": 100, "unit": "percent"})
    return [out]


def media_site_cmd(url: str, dest: Path, sections: str | None = None,
                   playlist: bool = False) -> list[str]:
    """yt-dlp argv (§2.5.3). Always -x --audio-format wav: video is dead
    weight and prepare normalizes anyway, so no format negotiation exists.
    Playlists need the explicit opt-in flag or --no-playlist pins the
    download to one video."""
    cmd = ["yt-dlp", "-x", "--audio-format", "wav",
           "-o", str(dest / "%(title)s.%(ext)s")]
    if not playlist:
        cmd.append("--no-playlist")
    if sections:
        cmd += ["--download-sections", sections]
    cmd.append(url)
    return cmd


def fetch_media_site(url: str, dest: Path, sections: str | None = None,
                     playlist: bool = False, emit=None) -> list[Path]:
    cmd = media_site_cmd(url, dest, sections, playlist)
    tail: list[str] = []
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp is not installed in this image; media-site ingest "
            "is unavailable") from None
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        tail.append(line)
        tail = tail[-30:]
        m = re.search(r"\[download\]\s+(\d+(?:\.\d+)?)%", line)
        if m and emit:
            emit("PROGRESS", {"current": float(m.group(1)),
                              "total": 100, "unit": "percent"})
    code = proc.wait()
    if code != 0:
        # Surface the extractor's own words: it is usually the fix (§2.5.3)
        raise RuntimeError("yt-dlp failed:\n" + "\n".join(tail[-10:]))
    got = sorted(p for p in dest.iterdir() if p.is_file()
                 and p.suffix.lower() in AUDIO_EXT)
    if not got:
        raise RuntimeError("yt-dlp produced no audio files")
    return got


def _snapshot_download(repo_id: str) -> Path:
    """The network seam (§2.5.4): module-level so tests can monkeypatch
    it; huggingface_hub stays a lazy import so the image needs it only
    when hf-dataset ingest is actually used."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed in this image; hf-dataset "
            "ingest is unavailable") from None
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))


def fetch_hf_dataset(repo_id: str, dest: Path,
                     split: str | None = None
                     ) -> tuple[list[Path], list[tuple[str, str]] | None]:
    """HuggingFace dataset (§2.5.4). Supports the audio-directory layout:
    audio files (any processed container) plus a csv/tsv/jsonl metadata
    file carrying transcripts. Returns (audio files, rows) where rows key
    on the STAGED incoming filename — the only name the caller's landing
    pass knows — so collision renames in raw/ are absorbed by the
    caller's landed map. Parquet-embedded audio is refused with a plain
    explanation rather than half-imported."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo_id):
        raise RuntimeError(f"not a dataset repo_id: {repo_id!r} "
                           "(expected owner/name)")
    snap = _snapshot_download(repo_id)
    everything = sorted(p for p in snap.rglob("*") if p.is_file())
    if split:
        narrowed = [p for p in everything
                    if split in p.parts or split in p.name]
        if narrowed:
            everything = narrowed
    audio = [p for p in everything if p.suffix.lower() in AUDIO_EXT]
    meta = [p for p in everything
            if p.suffix.lower() in (".csv", ".tsv", ".jsonl")]
    parquet = [p for p in everything if p.suffix.lower() == ".parquet"]
    if not audio:
        if parquet:
            raise RuntimeError(
                f"{repo_id} ships as parquet shards with embedded audio — "
                "this image cannot unpack those yet. Convert to a wav + "
                "metadata layout outside, or extend ingest.")
        raise RuntimeError(f"no audio files found in {repo_id}")

    # Land audio under sanitized stems now; the raw/ pass will agree
    # (sanitize is idempotent) unless a collision renames there — which
    # the rows re-mapping in the caller absorbs.
    stem_map: dict[str, str] = {}
    out_files: list[Path] = []
    used: set[str] = set()
    for p in audio:
        stem = sanitize_stem(p.stem)
        final = stem
        n = 1
        while final in used:
            final = f"{stem}-{n}"
            n += 1
        used.add(final)
        dst = dest / f"{final}{p.suffix.lower()}"
        shutil.copy2(p, dst)
        stem_map[p.name] = dst.name
        out_files.append(dst)

    rows: list[tuple[str, str]] | None = None
    for mf in meta:
        parsed = _parse_metadata(mf)
        if parsed is None:
            continue
        rows = parsed
        break
    if rows is not None:
        # Re-key rows from the original dataset filename to the staged
        # incoming name — the identity the caller's landed map carries
        # through the raw/ landing pass.
        by_stem = {Path(orig).stem: staged for orig, staged in stem_map.items()}
        keyed = []
        for stem, text in rows:
            staged = by_stem.get(stem)
            if staged is not None:
                keyed.append((staged, text))
        if keyed:
            return out_files, keyed
    return out_files, None


def _parse_metadata(path: Path) -> list[tuple[str, str]] | None:
    """Read a transcript file into (stem, text) pairs, or None when its
    columns do not match a known layout."""
    try:
        if path.suffix.lower() == ".jsonl":
            records = [json.loads(line) for line in
                       path.read_text().splitlines() if line.strip()]
        else:
            delim = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open(newline="") as fh:
                records = list(csv.DictReader(fh, delimiter=delim))
    except (OSError, json.JSONDecodeError, csv.Error):
        return None
    if not records:
        return None
    cols = set(records[0].keys() or [])
    fcol = next((c for c in FILE_COLUMNS if c in cols), None)
    tcol = next((c for c in TEXT_COLUMNS if c in cols), None)
    if not fcol or not tcol:
        return None
    out = []
    for rec in records:
        fval, tval = rec.get(fcol), rec.get(tcol)
        if not isinstance(fval, str) or not isinstance(tval, str):
            continue
        out.append((Path(fval).stem, tval))
    return out or None
