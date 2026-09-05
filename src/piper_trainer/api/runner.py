"""One job, executed as a process.

    python -m piper_trainer.api.runner <job-dir>

Calls the same `piper_trainer.*` functions the CLI does (design doc §0: the
API layer never re-implements the pipeline) and reports through stdout:

    ##TARGET {json}     progress denominator, e.g. {"total": 4000, "unit": "epoch"}
    ##PROGRESS {json}   progress updates; merged into job.json by the manager
    ##RESULT {json}     final stage summary, captured into job.json

When spawned by the manager, directives carry the per-job nonce from
$PIPER_DIRECTIVE_NONCE (`##<nonce> TARGET {json}`); only those are
honored, so echoed pipeline output cannot forge one.

The exit code decides succeeded/failed; every other line is log output.

`execute()` is importable so tests can run a stage in-process; `main()` is
the process entry the job manager spawns.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .. import clean as clean_mod
from .. import export as export_mod
from .. import ingest as ingest_mod
from .. import metadata, metrics as metrics_mod, prepare, train as train_mod
from .. import say as say_mod
from .. import transcribe as transcribe_mod
from .. import validate as validate_mod
from ..config import Project, TIERS
from ..lock import project_lock
from ..validate import validate_dataset


def _emit(tag: str, obj: dict) -> None:
    # The manager gives every job a nonce in the environment (review
    # finding 13): prefixing directives with it means a transcript or
    # filename echoed on stdout can never pose as one. Bare (no env) keeps
    # hand-run runners legible.
    nonce = os.environ.get("PIPER_DIRECTIVE_NONCE", "")
    prefix = f"##{nonce} " if nonce else "##"
    print(f"{prefix}{tag} {json.dumps(obj)}", flush=True)


def _tier(project: Project, params: dict) -> str:
    tier = params.get("tier") or project.get("tier") or "medium"
    if tier not in TIERS:
        raise RuntimeError(f"unknown tier {tier!r}")
    return tier


def _espeak_voice(project: Project, params: dict) -> str:
    """Same resolution order as the CLI: explicit > saved > en-us (with the
    same warning when an explicit value disagrees with the saved one)."""
    given = params.get("espeak_voice")
    saved = project.get("espeak_voice")
    if given:
        if saved and saved != given:
            print(f"! espeak voice changed: project has {saved!r}, "
                  f"using {given!r}", file=sys.stderr)
        project.set(espeak_voice=given)
        return given
    if saved:
        return saved
    print("! no espeak voice set for this project; defaulting to 'en-us'",
          file=sys.stderr)
    return "en-us"


# -------------------------------------------------------------------- stages

def _prepare(project: Project, params: dict, emit) -> dict:
    stats = prepare.run_all(
        project,
        tier=_tier(project, params),
        channel=params.get("channel"),
        denoise_enabled=params.get("denoise", True),
        force=params.get("force", False),
        energy_threshold=params.get("energy_threshold", 55),
        min_dur=params.get("min_dur", 1.5),
        max_dur=params.get("max_dur", 10.0),
        max_silence=params.get("max_silence", 0.4),
        max_leading_silence=params.get("pad", 0.15),
        max_trailing_silence=params.get("pad", 0.15),
        on_stage=lambda i, name: emit(
            "PROGRESS", {"current": i, "total": 4, "unit": "stage"}),
    )
    clips = stats.get("clips")
    if isinstance(clips, int):
        # current rides along so the union-merge in jobs.py doesn't leave
        # the last stage count ("stage 4/4") bleeding into "clip N/M"
        emit("TARGET", {"total": clips, "unit": "clip", "current": clips})
    return {"stats": stats}


def _transcribe(project: Project, params: dict, emit) -> dict:
    # Per-clip reporting: without it a large-v3 batch over hundreds of
    # clips is a silent log and a frozen bar for however long the model
    # needs, then a 100% jump the moment RESULT lands.
    total = len(list(project.wavs.glob("*.wav")))
    emit("TARGET", {"total": total, "unit": "clip"})

    def on_progress(done: int, total: int, name: str) -> None:
        emit("PROGRESS", {"current": done, "total": total, "unit": "clip"})
        print(f"clip {done}/{total}: {name}", flush=True)

    stats = transcribe_mod.transcribe(
        project,
        model_size=params.get("model"),
        language=params.get("language", "en"),
        device=params.get("device", "cpu"),
        retranscribe=params.get("retranscribe", False),
        normalize=params.get("normalize", True),
        on_progress=on_progress,
    )
    return {"stats": stats}


def _findings_json(findings) -> list[dict]:
    return [{"level": f.level, "code": f.code, "message": f.message,
             "ids": f.ids, "action": f.action} for f in findings]


def _validate(project: Project, params: dict, emit) -> dict:
    findings = validate_dataset(
        project,
        tier=_tier(project, params),
        batch_size=params.get("batch_size"),
        espeak_voice=params.get("espeak_voice"),
    )
    out = _findings_json(findings)
    for f in findings:
        print(f)  # lands in the job log
    return {"findings": out,
            "errors": sum(1 for f in findings if f.level == "error")}


def _clean(project: Project, params: dict, emit) -> dict:
    tier = _tier(project, params)
    findings = validate_dataset(project, tier=tier,
                                espeak_voice=params.get("espeak_voice"))
    plan = clean_mod.build_plan(
        project, findings,
        only=set(params["only"].split(",")) if params.get("only") else None,
        exclude=set(params["exclude"].split(","))
        if params.get("exclude") else None)
    total = len(metadata.read(project.metadata)[0]) \
        if project.metadata.exists() else 0
    endings = metadata.line_endings(project.metadata) \
        if project.metadata.exists() else None
    if not params.get("apply", False):
        return {"plan": clean_mod.describe(plan, total, endings=endings)}
    # clean.apply enforces the one-third guard; RuntimeError fails the job
    # with the guard's own message
    return {"stats": clean_mod.apply(project, plan,
                                     force=params.get("force", False))}


def _restore(project: Project, params: dict, emit) -> dict:
    ids = params.get("ids")
    return {"stats": clean_mod.restore(
        project,
        clip_ids=ids.split(",") if isinstance(ids, str) else ids,
        files_only=params.get("files_only", False))}


def _train(project: Project, params: dict, emit) -> dict:
    tier = _tier(project, params)
    espeak_voice = _espeak_voice(project, params)
    batch_size = params.get("batch_size", 32)
    validation_split = params.get("validation_split", 0.02)

    if not params.get("skip_validate", False):
        findings = validate_dataset(project, tier=tier,
                                    batch_size=batch_size,
                                    espeak_voice=espeak_voice,
                                    validation_split=validation_split)
        for f in findings:
            print(f)
        errors = [f for f in findings if f.level == "error"]
        if errors:
            raise RuntimeError(
                "validation failed; fix the errors before training: "
                + "; ".join(f"[{f.code}] {f.message}" for f in errors))

    # warmstart (§6.4 "start from this voice"): weights-only, epoch count
    # starts at zero. Accepts absolute paths (fetched catalog checkpoints
    # record one) or project-relative ones (the /checkpoints listing's
    # run entries), so the UI can echo either back verbatim.
    warmstart = None
    if params.get("warmstart"):
        warmstart = Path(params["warmstart"])
        if not warmstart.is_absolute():
            warmstart = project.root / warmstart
        if not warmstart.is_file():
            raise RuntimeError(f"warmstart checkpoint not found: {warmstart}")

    resume = params.get("resume")
    # "N more" is a resume by definition: the epoch arithmetic needs the
    # checkpoint's counter, so add_epochs with no explicit checkpoint
    # resumes from the latest one. (The CLI asks for --resume auto
    # explicitly; the API makes it implicit so a "train N more" button
    # cannot produce "--add-epochs needs a checkpoint" by omission.)
    if resume is None and params.get("add_epochs") is not None:
        resume = "auto"
    ckpt_epoch = None
    if resume:
        if resume == "auto":
            resume = train_mod.latest_checkpoint(project, tier)
            if resume is None:
                raise RuntimeError("--resume auto found no checkpoint")
        else:
            resume = Path(resume)
            if not resume.exists():
                raise RuntimeError(f"checkpoint not found: {resume}")
        ckpt_epoch = train_mod.checkpoint_epoch(resume)
        print(f"resuming from {resume}"
              + (f" (epoch {ckpt_epoch})" if ckpt_epoch is not None else ""))

    # Epoch arithmetic shared with the CLI (design doc §1.4): max_epochs is
    # an absolute ceiling; the UI submits "N more" as completed + N.
    max_epochs = train_mod.resolve_max_epochs(
        ckpt_epoch, params.get("add_epochs"), params.get("max_epochs"))
    if resume is not None:
        train_mod.check_resume_ceiling(ckpt_epoch, max_epochs)

    targets = project.get("target_epochs") or {}
    if resume is None or tier not in targets:
        targets[tier] = max_epochs
        project.set(target_epochs=targets)

    emit("TARGET", {"total": max_epochs, "unit": "epoch"})

    cmd = train_mod.build_command(
        project, tier=tier, espeak_voice=espeak_voice,
        batch_size=batch_size, max_epochs=max_epochs,
        num_workers=params.get("num_workers", 8),
        validation_split=validation_split,
        warmstart=warmstart,
        resume=resume,
        accelerator=params.get("accelerator", "gpu"),
        precision=params.get("precision", "32-true"))
    # Live loss curve (§6.4): the trainer's progress bar carries no loss
    # values, so a thread tails the CSV logger's metrics.csv and prints
    # one line per epoch — the log is the data path, same as the
    # directives. Set only after the subprocess exits so the final drain
    # catches the last epoch's line.
    stop = threading.Event()
    tailer = threading.Thread(
        target=metrics_mod.tail_metrics,
        args=(project.runs(tier), stop, lambda line: print(line, flush=True)),
        daemon=True)
    tailer.start()
    try:
        # run() prints the command itself; printing here too doubled it in the log
        code = train_mod.run(cmd)
    finally:
        stop.set()
        tailer.join(timeout=10)
    if code != 0:
        raise RuntimeError(f"training exited with code {code}")
    latest = train_mod.latest_checkpoint(project, tier)
    return {"tier": tier, "max_epochs": max_epochs,
            "checkpoint": str(latest) if latest else None}


def _export(project: Project, params: dict, emit) -> dict:
    tier = _tier(project, params)
    ckpt = params.get("checkpoint") or train_mod.latest_checkpoint(project, tier)
    if not ckpt:
        raise RuntimeError(
            f"no {tier} checkpoint found to export — run a train job first")
    # Same convention as _train's warmstart: absolute (catalog fetch) or
    # project-relative (the /checkpoints listing) both resolve.
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project.root / ckpt_path
    onnx_path, json_path = export_mod.export(
        project, tier, ckpt_path,
        voice_name=params.get("voice_name"),
        espeak_voice=_espeak_voice(project, params),
        length_scale=params.get("length_scale"),
        noise_scale=params.get("noise_scale"),
        noise_w=params.get("noise_w"))
    problems = export_mod.verify(onnx_path, json_path)
    if problems:
        raise RuntimeError("export verification failed: "
                           + "; ".join(problems))
    return {"artifacts": [str(onnx_path), str(json_path)]}


def _merge_transcripts(project: Project, rows: list[tuple[str, str]],
                       landed: dict[str, str]) -> int:
    """Append hf-dataset transcripts to dataset/metadata.csv (§2.5.4).
    Rows key on the original audio filename; `landed` maps that to the
    name it actually got in raw/ (collision renames). Existing rows win —
    re-running an ingest must never duplicate or clobber edits."""
    existing, problems = ([], [])
    if project.metadata.exists():
        existing, problems = metadata.read(project.metadata)
    have = {cid for cid, _ in existing}
    fresh = [(Path(landed[orig]).stem, text) for orig, text in rows
             if orig in landed]
    added = [(cid, text) for cid, text in fresh if cid not in have]
    if not added:
        return 0
    merged = existing + added
    raw_lines = {p.line_no: p.raw for p in problems}
    metadata.write(project.metadata, merged, raw_lines=raw_lines)
    return len(added)


def _ingest(project: Project, params: dict, emit) -> dict:
    """Ingest (design doc §2.5): every source_type lands files in the
    job's staged incoming/, then one shared pass sanitizes into raw/,
    ffprobes, and reports the same summary. Only acquisition differs."""
    source_type = params.get("source_type", "upload")
    job_dir = params["_job_dir"]
    incoming = job_dir / "incoming"
    files = sorted(p for p in incoming.iterdir()) if incoming.exists() else []
    hf_rows: list[tuple[str, str]] | None = None

    if source_type == "upload":
        if not files:
            raise RuntimeError("no files were staged for this ingest job")
    elif source_type == "url":
        incoming.mkdir(parents=True, exist_ok=True)
        files = ingest_mod.fetch_url(params.get("url") or "", incoming, emit)
    elif source_type == "media-site":
        incoming.mkdir(parents=True, exist_ok=True)
        files = ingest_mod.fetch_media_site(
            params.get("url") or "", incoming,
            sections=params.get("sections"),
            playlist=bool(params.get("playlist")), emit=emit)
    elif source_type == "hf-dataset":
        incoming.mkdir(parents=True, exist_ok=True)
        files, hf_rows = ingest_mod.fetch_hf_dataset(
            params.get("repo_id") or "", incoming,
            split=params.get("split"))
    else:
        raise RuntimeError(f"unknown source_type {source_type!r}")

    project.raw.mkdir(parents=True, exist_ok=True)
    renamed: list[dict] = []
    probed = []
    landed: dict[str, str] = {}
    for f in files:
        stem = prepare.sanitize_stem(f.stem)
        dst = project.raw / f"{stem}{f.suffix.lower()}"
        n = 1
        while dst.exists():
            dst = project.raw / f"{stem}-{n}{f.suffix.lower()}"
            n += 1
        shutil.move(str(f), dst)
        landed[f.name] = dst.name
        if dst.name != f.name:
            renamed.append({"original": f.name, "stored_as": dst.name})
        info = prepare.probe(dst)
        probed.append({"name": dst.name,
                       "codec": info.get("codec_name"),
                       "sample_rate": info.get("sample_rate"),
                       "channels": info.get("channels"),
                       "duration": info.get("duration")})

    result: dict = {"added": len(files), "renamed": renamed,
                    "files": probed, "source_type": source_type}
    if hf_rows is not None:
        written = _merge_transcripts(project, hf_rows, landed)
        project.set(transcripts_provided=True)
        result["transcripts_written"] = written
    return result


def _fetch_checkpoint(project: Project, params: dict, emit) -> dict:
    """Download one catalog checkpoint into base_checkpoints/ (§3.5), with
    '=' stripped from filenames and the mapping recorded in project.json."""
    catalog_path = params.get("catalog_path") or ""
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", catalog_path) \
            or ".." in catalog_path or len(catalog_path.split("/")) != 4:
        raise RuntimeError(f"not a catalog checkpoint path: {catalog_path!r} "
                           f"(expected family/locale/voice/quality)")
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed in this image; checkpoint "
            "fetch is unavailable") from exc

    from . import catalog
    wanted = sorted(
        f for f in list_repo_files(catalog.REPO, repo_type="dataset")
        if f.startswith(catalog_path + "/") and f.count("/") == 4)
    if not wanted:
        raise RuntimeError(f"no files found at {catalog_path}")

    local_dir = project.checkpoints / catalog_path.replace("/", "-")
    local_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for i, rel in enumerate(wanted, start=1):
        emit("PROGRESS", {"current": i, "total": len(wanted), "unit": "file"})
        dl = hf_hub_download(catalog.REPO, rel, repo_type="dataset")
        original = rel.rsplit("/", 1)[1]
        # '=' breaks shells, env vars and container args (§3.5)
        safe = original.replace("=", "_")
        shutil.copy2(dl, local_dir / safe)
        mapping[original] = safe

    saved = project.get("base_checkpoints") or {}
    saved[catalog_path] = {
        "dir": str(local_dir),
        "files": mapping,
        "fetched_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    project.set(base_checkpoints=saved)
    return {"catalog_path": catalog_path, "dir": str(local_dir),
            "files": mapping}


def _preview_source(project: Project, params: dict) -> Path:
    name = Path(params.get("source") or "").name
    src = project.raw / name
    if not name or not src.is_file():
        raise RuntimeError(f"source not found in raw/: {name!r}")
    return src


def _preview_dir(project: Project, params: dict, stage: str) -> Path:
    """Previews write ONLY to work/preview/<stage>/<preview-id>/ (§2.1);
    nothing enters dataset/ except via a full run."""
    d = project.root / "work" / "preview" / stage / params["_job_dir"].name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_preview(pdir: Path, stage: str, params: dict, result: dict) -> dict:
    """Write preview.json (the envelope records the winning parameters so
    promote can replay them as a full run, §2.1) and return the result
    itself, which becomes the job's ##RESULT payload."""
    from .jobs import _now
    envelope = {"id": pdir.name, "stage": stage,
                "params": {k: v for k, v in params.items()
                           if k not in ("stage", "_job_dir")},
                "created_at": _now(), "result": result}
    (pdir / "preview.json").write_text(json.dumps(envelope, indent=2) + "\n")
    return result


def _histogram(durations: list[float], width: float = 1.0) -> list[dict]:
    """1 s duration bins: the sweep's at-a-glance shape comparison."""
    bins: dict[int, int] = {}
    for d in durations:
        bins[min(int(d / width), 30)] = bins.get(min(int(d / width), 30), 0) + 1
    return [{"from": b * width, "to": (b + 1) * width, "count": n}
            for b, n in sorted(bins.items())]


def _preview_segment(project: Project, params: dict, emit) -> dict:
    """§2.2 segment preview: one source through the tuner's VAD parameters.
    Returns clip count, duration histogram, boundary timestamps and the
    first clips as playable audio — judge whether clips start/end on word
    boundaries and whether the count is plausible for the duration."""
    src = _preview_source(project, params)
    pdir = _preview_dir(project, params, "segment")
    channel = params.get("channel")
    keep = max(1, min(int(params.get("clips_kept", 5)), 20))

    emit("PROGRESS", {"current": 1, "total": 3, "unit": "step"})
    workdir = pdir / "_work"
    work = workdir / "src-48k.wav"
    prepare.convert_one(src, work, channel=channel)
    if params.get("denoise", True):
        # the full pipeline denoises before segmenting, so the preview must
        # judge boundaries on the audio a full run would actually split
        emit("PROGRESS", {"current": 2, "total": 3, "unit": "step"})
        prepare.denoise_file(work, pdir / "_dn")
        (pdir / "_dn" / work.name).replace(work)
        (pdir / "_dn").rmdir()

    emit("PROGRESS", {"current": 3, "total": 3, "unit": "step"})
    threshold = float(params.get("energy_threshold", 55))
    clips = prepare.split_audio(
        work, pdir, stem=src.stem,
        energy_threshold=threshold,
        min_dur=float(params.get("min_dur", 1.5)),
        max_dur=float(params.get("max_dur", 10.0)),
        max_silence=float(params.get("max_silence", 0.4)),
        max_leading_silence=float(params.get("pad", 0.15)),
        max_trailing_silence=float(params.get("pad", 0.15)))
    # Measured on the exact audio that was just split, so the UI can say WHY
    # a file produced nothing (quiet source vs threshold) instead of only
    # advising blind dial-turning.
    level = prepare.wav_level_dbfs(work)
    shutil.rmtree(workdir, ignore_errors=True)

    for extra in clips[keep:]:  # previews stay small; keep the first N
        (pdir / extra["clip"]).unlink(missing_ok=True)
    durs = [c["end"] - c["start"] for c in clips]
    result = {"clip_count": len(clips),
              "duration_total": round(sum(durs), 2),
              "histogram": _histogram(durs),
              "clips": clips[:200],       # boundaries for the overlay
              "clips_truncated": len(clips) > 200,
              "audio": [c["clip"] for c in clips[:keep]],
              "level": level}
    if not clips:
        print(f"! {src.name}: 0 clips — speech level {level['speech_dbfs']} "
              f"dBFS vs threshold {threshold:g} (rejects below "
              f"{threshold - prepare.INT16_FULL_SCALE_DB:.1f} dBFS); lower "
              "the energy threshold below the speech level", flush=True)
    emit("TARGET", {"total": len(clips), "unit": "clip"})
    return _write_preview(pdir, "segment", params, result)


def _preview_segment_all(project: Project, params: dict, emit) -> dict:
    """Batch form of the segment preview: every raw source through one dial
    set, one row each. After a handful of single-file previews the winning
    dials are usually obvious; this proves them against the whole set —
    and names every source they would leave at zero — before promote
    commits a full run. No playable clips are kept (counts and levels are
    the product here; the single-source preview stays the listening tool),
    and one bad source records an error row instead of killing the batch.
    """
    srcs = [project.raw / s["name"] for s in prepare.sources(project)]
    if not srcs:
        raise RuntimeError("no sources in raw/ to preview")
    emit("TARGET", {"total": len(srcs), "unit": "source"})
    pdir = _preview_dir(project, params, "segment-all")
    workdir = pdir / "_work"
    threshold = float(params.get("energy_threshold", 55))
    dials = dict(
        energy_threshold=threshold,
        min_dur=float(params.get("min_dur", 1.5)),
        max_dur=float(params.get("max_dur", 10.0)),
        max_silence=float(params.get("max_silence", 0.4)),
        max_leading_silence=float(params.get("pad", 0.15)),
        max_trailing_silence=float(params.get("pad", 0.15)))
    denoise = params.get("denoise", True)

    per_source: list[dict] = []
    all_durs: list[float] = []
    for i, src in enumerate(srcs, start=1):
        workdir.mkdir(parents=True, exist_ok=True)
        work = workdir / f"{src.stem}-48k.wav"
        try:
            prepare.convert_one(src, work, channel=params.get("channel"))
            if denoise:
                # same rule as the single preview: judge the audio a full
                # run would actually split
                prepare.denoise_file(work, workdir / "_dn")
                (workdir / "_dn" / work.name).replace(work)
                (workdir / "_dn").rmdir()
            clips = prepare.split_audio(work, workdir / "clips",
                                        stem=src.stem, **dials)
            level = prepare.wav_level_dbfs(work)
            durs = [c["end"] - c["start"] for c in clips]
            all_durs.extend(durs)
            row = {"source": src.name, "clips": len(clips),
                   "seconds": round(sum(durs), 2), "level": level,
                   "error": None}
            print(f"segment-all: {src.name}: {len(clips)} clips", flush=True)
            if not clips:
                print(f"! segment-all: {src.name}: 0 clips — speech "
                      f"{level['speech_dbfs']} dBFS vs threshold "
                      f"{threshold:g} (rejects below "
                      f"{threshold - prepare.INT16_FULL_SCALE_DB:.1f} dBFS)",
                      flush=True)
        except Exception as exc:  # one bad file must not kill the batch
            row = {"source": src.name, "clips": 0, "seconds": 0,
                   "level": None, "error": str(exc)}
            print(f"! segment-all: {src.name}: failed: {exc}", flush=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
            emit("PROGRESS", {"current": i, "total": len(srcs),
                              "unit": "source"})
        per_source.append(row)
    zeros = sorted(r["source"] for r in per_source if r["clips"] == 0)
    result = {"clip_count": sum(r["clips"] for r in per_source),
              "duration_total": round(sum(all_durs), 2),
              "histogram": _histogram(all_durs),
              "per_source": per_source,
              "zeros": zeros,
              "audio": []}
    emit("TARGET", {"total": result["clip_count"], "unit": "clip"})
    return _write_preview(pdir, "segment-all", params, result)


def _preview_denoise(project: Project, params: dict, emit) -> dict:
    """§2.2 denoise preview: an excerpt, original vs denoised. Judge
    sibilants, breaths, plosives — metallic or gated means back off."""
    src = _preview_source(project, params)
    pdir = _preview_dir(project, params, "denoise")
    seconds = min(max(float(params.get("seconds", 25)), 1.0), 60.0)

    emit("PROGRESS", {"current": 1, "total": 3, "unit": "step"})
    loud = pdir / "src-48k.wav"
    prepare.convert_one(src, loud, channel=params.get("channel"))
    original = pdir / "original.wav"
    prepare.excerpt(loud, original, seconds)
    loud.unlink()
    emit("PROGRESS", {"current": 2, "total": 3, "unit": "step"})
    prepare.denoise_file(original, pdir / "_dn")
    (pdir / "_dn" / original.name).replace(pdir / "denoised.wav")
    (pdir / "_dn").rmdir()
    emit("PROGRESS", {"current": 3, "total": 3, "unit": "step"})
    return _write_preview(pdir, "denoise", params,
                          {"seconds": seconds,
                           "audio": ["original.wav", "denoised.wav"]})


def _preview_espeak(project: Project, params: dict) -> str:
    """Read-only twin of _espeak_voice: a preview must not write an
    explicit override into project.json (§2.1)."""
    return (params.get("espeak_voice") or project.get("espeak_voice")
            or "en-us")


_IT_PER_S = re.compile(r"(\d+(?:\.\d+)?)\s*it/s")


def _step_rate_from_log(log_path: Path) -> float | None:
    """The last `it/s` in Lightning's progress bar — the steady-state step
    rate the trainer measured itself, excluding startup and preprocessing.
    The manager tees runner stdout into log.txt through a pipe, so give
    the drain a moment before concluding the bar isn't there."""
    for _ in range(4):
        if log_path.exists():
            hits = _IT_PER_S.findall(log_path.read_text(errors="replace"))
            if hits:
                return float(hits[-1])
        time.sleep(0.25)
    return None


def _human_seconds(sec: float) -> str:
    s = int(sec)
    if s < 90:
        return f"~{s}s"
    m, h = s // 60, s // 3600
    if h == 0:
        return f"~{m}m"
    return f"~{h}h {m % 60:02d}m"


def _preview_train(project: Project, params: dict, emit) -> dict:
    """§2.2 train preview, the one the design doc calls highest-value: a
    real ~N-step run (default 50) that answers "does it start, how fast,
    and how long would the full run take" in under a minute. It builds the
    exact command a full train builds — same validation gate, same
    warmstart/resume resolution, same phonemization cache — then caps
    Lightning with --trainer.max_steps and redirects every output it
    writes into work/preview/train/<id>/ so runs-<tier>/ and project.json
    stay untouched (§2.1). A failure here is the product: a broken
    warmstart, an oversized batch or a mis-set espeak voice dies here
    instead of three hours into a real run."""
    tier = _tier(project, params)
    espeak_voice = _preview_espeak(project, params)
    batch_size = int(params.get("batch_size", 32))
    validation_split = params.get("validation_split", 0.02)
    steps = max(10, min(int(params.get("steps", 50)), 500))

    if not params.get("skip_validate", False):
        findings = validate_dataset(project, tier=tier,
                                    batch_size=batch_size,
                                    espeak_voice=espeak_voice,
                                    validation_split=validation_split)
        for f in findings:
            print(f)
        errors = [f for f in findings if f.level == "error"]
        if errors:
            raise RuntimeError(
                "validation failed; fix the errors before training: "
                + "; ".join(f"[{f.code}] {f.message}" for f in errors))

    # Same warmstart/resume resolution as _train — the preview only means
    # something if it exercises the mode the real run will use.
    warmstart = None
    if params.get("warmstart"):
        warmstart = Path(params["warmstart"])
        if not warmstart.is_absolute():
            warmstart = project.root / warmstart
        if not warmstart.is_file():
            raise RuntimeError(f"warmstart checkpoint not found: {warmstart}")

    resume = params.get("resume")
    if resume is None and params.get("add_epochs") is not None:
        resume = "auto"
    if resume == "auto":
        resume = train_mod.latest_checkpoint(project, tier)
        if resume is None:
            raise RuntimeError("--resume auto found no checkpoint")
    elif resume is not None:
        resume = Path(resume)
        if not resume.exists():
            raise RuntimeError(f"checkpoint not found: {resume}")

    rows = len(metadata.read(project.metadata)[0]) \
        if project.metadata.exists() else 0
    if rows == 0:
        raise RuntimeError("no dataset rows to train on — transcribe first")
    steps_per_epoch = (rows + batch_size - 1) // batch_size

    # max_steps is a global counter, so a resume must add its N on top of
    # the steps the checkpoint already consumed (see checkpoint_global_step).
    ckpt_epoch = None
    global_step = 0
    if resume is not None:
        ckpt_epoch = train_mod.checkpoint_epoch(resume)
        gs = train_mod.checkpoint_global_step(resume)
        if gs is not None:
            global_step = gs
        elif ckpt_epoch:
            global_step = ckpt_epoch * steps_per_epoch
        print(f"preview resumes from {resume}"
              + (f" (epoch {ckpt_epoch}, step {global_step})"
                 if ckpt_epoch is not None else ""), flush=True)

    max_epochs = train_mod.resolve_max_epochs(
        ckpt_epoch, params.get("add_epochs"), params.get("max_epochs"))
    if resume is not None:
        train_mod.check_resume_ceiling(ckpt_epoch, max_epochs)
    remaining_epochs = max_epochs - (ckpt_epoch or 0)

    pdir = _preview_dir(project, params, "train")
    cmd = train_mod.build_command(
        project, tier=tier, espeak_voice=espeak_voice,
        batch_size=batch_size, max_epochs=max_epochs,
        num_workers=params.get("num_workers", 8),
        validation_split=validation_split,
        warmstart=warmstart,
        resume=resume,
        accelerator=params.get("accelerator", "gpu"),
        precision=params.get("precision", "32-true"),
        runs_dir=pdir)
    cmd += ["--trainer.max_steps", str(global_step + steps)]

    mode = ("resume" if resume is not None
            else "warmstart" if warmstart is not None else "scratch")
    emit("TARGET", {"total": steps, "unit": "step"})
    print(f"train preview ({mode}): {steps} steps, batch {batch_size}, "
          f"{rows} clips; all outputs -> {pdir}", flush=True)
    t0 = time.monotonic()
    code = train_mod.run(cmd)
    elapsed = time.monotonic() - t0
    if code != 0:
        raise RuntimeError(
            f"train preview exited with code {code} after {elapsed:.0f}s — "
            "the full run would have failed the same way; the reason is in "
            "the log above")

    rate = _step_rate_from_log(params["_job_dir"] / "log.txt")
    rate_source = "progress-bar"
    if rate is None:
        # no bar in the log: fall back to wall clock, which silently folds
        # startup and first-run phonemization into the step rate
        rate = steps / elapsed if elapsed > 0 else 0.0
        rate_source = "wall-clock"
    seconds_per_epoch = steps_per_epoch / rate if rate > 0 else None
    projected = remaining_epochs * seconds_per_epoch \
        if seconds_per_epoch else None

    result: dict = {
        "mode": mode,
        "tier": tier,
        "batch_size": batch_size,
        "clips": rows,
        "steps_planned": steps,
        "steps_per_epoch": steps_per_epoch,
        "elapsed_seconds": round(elapsed, 1),
        "steps_per_sec": round(rate, 3) if rate > 0 else None,
        "rate_source": rate_source,
        "seconds_per_epoch": (round(seconds_per_epoch, 1)
                              if seconds_per_epoch else None),
        "target_epochs": max_epochs,
        "remaining_epochs": remaining_epochs,
        "projected_seconds": round(projected) if projected else None,
        "projected_human": _human_seconds(projected) if projected else None,
    }
    if rate_source == "wall-clock" and rate > 0:
        result["note"] = ("rate includes startup and first-run "
                          "phonemization; treat the projection as an "
                          "upper bound")
    print(f"train preview: {steps} steps in {elapsed:.1f}s "
          f"({rate:.2f} it/s, {rate_source}); full run "
          f"({remaining_epochs} epochs x {steps_per_epoch} steps) "
          f"projected {result['projected_human'] or 'n/a'}", flush=True)
    return _write_preview(pdir, "train", params, result)


def _recent_checkpoints(project: Project, tier: str,
                        limit: int) -> list[Path]:
    """The last N saved run checkpoints, newest first. Same glob as the
    /checkpoints listing; mtime order because Lightning writes each save
    as it goes and the operator's question is always about the newest."""
    runs = project.runs(tier)
    if not runs.exists():
        return []
    cands = sorted(runs.glob("lightning_logs/version_*/checkpoints/*.ckpt"),
                   key=lambda p: p.stat().st_mtime)
    return list(reversed(cands[-limit:]))


def _preview_audition(project: Project, params: dict, emit) -> dict:
    """§2.3 audition: one held-out sentence rendered through N checkpoints,
    presented A/B/C. The only cheap answer to "are more epochs helping?" —
    if the newest take is not clearly better than its predecessor, the run
    has plateaued and the effort belongs in more audio, not more epochs.
    Each take is a REAL export (torch load + ONNX conversion, minutes on
    CPU) followed by the same say subprocess the voices screen uses; the
    take models land in work/preview/audition/<id>/ so out/ only ever
    holds deliberately exported voices."""
    tier = _tier(project, params)
    limit = max(1, min(int(params.get("limit", 3)), 5))
    text = str(params.get("text") or say_mod.DEFAULT_TEXT).strip()
    if not text:
        raise RuntimeError("audition text is empty")
    if len(text) > say_mod.MAX_TEXT:
        raise RuntimeError(
            f"audition text too long (max {say_mod.MAX_TEXT} characters)")

    # Explicit checkpoint list (absolute or project-relative, the same
    # convention _export accepts) or the default: the last N saved.
    wanted: list[Path] = []
    for c in (params.get("checkpoints") or []):
        p = Path(c)
        if not p.is_absolute():
            p = project.root / p
        if not p.is_file():
            raise RuntimeError(f"checkpoint not found: {c}")
        wanted.append(p)
    if not wanted:
        wanted = _recent_checkpoints(project, tier, limit)
        if not wanted:
            raise RuntimeError(
                f"no {tier} run checkpoints to audition — run a train job "
                "first, or pass checkpoints explicitly")
    wanted = wanted[:limit]

    pdir = _preview_dir(project, params, "audition")
    emit("TARGET", {"total": len(wanted), "unit": "take"})
    print(f"audition: {len(wanted)} checkpoint(s), one held-out sentence "
          f"through each (each take runs a real ONNX export — this takes "
          f"minutes)", flush=True)

    takes = []
    for i, ckpt in enumerate(wanted, start=1):
        # Lightning names checkpoints epoch=N-step=M; the filename is the
        # torch-free epoch source (same parse export.py uses for
        # provenance). last.ckpt carries no epoch -> plain take number.
        m = re.search(r"epoch=(\d+)", ckpt.name)
        epoch = int(m.group(1)) if m else None
        stem = f"take{i}-e{epoch}" if epoch is not None else f"take{i}"
        print(f"take {i}/{len(wanted)}: exporting {ckpt.name} -> "
              f"{stem}.onnx", flush=True)
        onnx_path, json_path = export_mod.export(
            project, tier, ckpt, voice_name=stem, out_dir=pdir)
        wav_path = pdir / f"{stem}.wav"
        wav_path.write_bytes(say_mod.synthesize(onnx_path, json_path, text))
        emit("PROGRESS", {"current": i, "total": len(wanted), "unit": "take"})
        takes.append({
            "take": i,
            "stem": stem,
            "checkpoint": str(ckpt.relative_to(project.root))
            if ckpt.is_relative_to(project.root) else str(ckpt),
            "epoch": epoch,
            "wav": wav_path.name,
        })
        print(f"take {i}/{len(wanted)}: synthesized {wav_path.name}",
              flush=True)

    return _write_preview(pdir, "audition", params,
                          {"text": text, "tier": tier, "takes": takes})


def _preview(project: Project, params: dict, emit) -> dict:
    stage = params.get("stage")
    if stage == "segment":
        return _preview_segment(project, params, emit)
    if stage == "segment-all":
        return _preview_segment_all(project, params, emit)
    if stage == "denoise":
        return _preview_denoise(project, params, emit)
    if stage == "train":
        return _preview_train(project, params, emit)
    if stage == "audition":
        return _preview_audition(project, params, emit)
    raise RuntimeError(f"unknown preview stage {stage!r} "
                       "(segment, segment-all, denoise, train and audition "
                       "ship today; finalize and transcribe later)")


HANDLERS = {
    "prepare": _prepare,
    "transcribe": _transcribe,
    "validate": _validate,
    "clean": _clean,
    "restore": _restore,
    "train": _train,
    "export": _export,
    "ingest": _ingest,
    "fetch-checkpoint": _fetch_checkpoint,
    "preview": _preview,
}

# validate is read-only; the CLI does not lock it either
UNLOCKED = frozenset({"validate"})


def execute(job_dir: Path) -> dict:
    """Run the stage for one job dir; returns the RESULT payload."""
    job = json.loads((job_dir / "job.json").read_text())
    kind = job["kind"]
    if kind not in HANDLERS:
        raise RuntimeError(f"unknown job kind {kind!r}")
    project = Project.load(job_dir.parents[1])
    params = dict(job.get("params") or {})
    params["_job_dir"] = job_dir
    handler = HANDLERS[kind]
    if kind in UNLOCKED:
        return handler(project, params, _emit)
    with project_lock(project, command=kind):
        return handler(project, params, _emit)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m piper_trainer.api.runner <job-dir>",
              file=sys.stderr)
        return 2
    try:
        result = execute(Path(argv[1]))
    except Exception as exc:
        traceback.print_exc()  # full detail lands in log.txt via the tee
        # The manager reads this as job["error"], so the jobs table shows
        # the reason and not a bare "exited with code 1".
        _emit("RESULT", {"error": str(exc) or type(exc).__name__})
        return 1
    _emit("RESULT", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
