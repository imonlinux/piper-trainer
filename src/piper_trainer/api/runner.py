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
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .. import clean as clean_mod
from .. import export as export_mod
from .. import ingest as ingest_mod
from .. import metadata, prepare, train as train_mod
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
    stats = transcribe_mod.transcribe(
        project,
        model_size=params.get("model"),
        language=params.get("language", "en"),
        device=params.get("device", "cpu"),
        retranscribe=params.get("retranscribe", False),
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
    # run() prints the command itself; printing here too doubled it in the log
    code = train_mod.run(cmd)
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
    onnx_path, json_path = export_mod.export(
        project, tier, Path(ckpt),
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
    clips = prepare.split_audio(
        work, pdir, stem=src.stem,
        energy_threshold=float(params.get("energy_threshold", 55)),
        min_dur=float(params.get("min_dur", 1.5)),
        max_dur=float(params.get("max_dur", 10.0)),
        max_silence=float(params.get("max_silence", 0.4)),
        max_leading_silence=float(params.get("pad", 0.15)),
        max_trailing_silence=float(params.get("pad", 0.15)))
    shutil.rmtree(workdir, ignore_errors=True)

    for extra in clips[keep:]:  # previews stay small; keep the first N
        (pdir / extra["clip"]).unlink(missing_ok=True)
    durs = [c["end"] - c["start"] for c in clips]
    result = {"clip_count": len(clips),
              "duration_total": round(sum(durs), 2),
              "histogram": _histogram(durs),
              "clips": clips[:200],       # boundaries for the overlay
              "clips_truncated": len(clips) > 200,
              "audio": [c["clip"] for c in clips[:keep]]}
    emit("TARGET", {"total": len(clips), "unit": "clip"})
    return _write_preview(pdir, "segment", params, result)


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


def _preview(project: Project, params: dict, emit) -> dict:
    stage = params.get("stage")
    if stage == "segment":
        return _preview_segment(project, params, emit)
    if stage == "denoise":
        return _preview_denoise(project, params, emit)
    raise RuntimeError(f"unknown preview stage {stage!r} "
                       "(step 2 ships segment and denoise; finalize, "
                       "transcribe, train and audition arrive later)")


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
