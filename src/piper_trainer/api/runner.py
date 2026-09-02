"""One job, executed as a process.

    python -m piper_trainer.api.runner <job-dir>

Calls the same `piper_trainer.*` functions the CLI does (design doc §0: the
API layer never re-implements the pipeline) and reports through stdout:

    ##TARGET {json}     progress denominator, e.g. {"total": 4000, "unit": "epoch"}
    ##PROGRESS {json}   progress updates; merged into job.json by the manager
    ##RESULT {json}     final stage summary, captured into job.json

The exit code decides succeeded/failed; every other line is log output.

`execute()` is importable so tests can run a stage in-process; `main()` is
the process entry the job manager spawns.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import clean as clean_mod
from .. import export as export_mod
from .. import metadata, prepare, train as train_mod
from .. import transcribe as transcribe_mod
from .. import validate as validate_mod
from ..config import Project, TIERS
from ..lock import project_lock
from ..validate import validate_dataset


def _emit(tag: str, obj: dict) -> None:
    print(f"##{tag} {json.dumps(obj)}", flush=True)


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
    )
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

    resume = params.get("resume")
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
        warmstart=Path(params["warmstart"]) if params.get("warmstart") else None,
        resume=resume,
        accelerator=params.get("accelerator", "gpu"),
        precision=params.get("precision", "32-true"))
    print(" \\\n  ".join(cmd), flush=True)
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
        raise RuntimeError("no checkpoint found to export")
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


def _ingest(project: Project, params: dict, emit) -> dict:
    """Upload ingest (design doc §2.5.1): staged files -> raw/, sanitized,
    probed. Other source types (url, media-site, hf-dataset) are step 2."""
    source_type = params.get("source_type", "upload")
    if source_type != "upload":
        raise RuntimeError(
            f"source_type {source_type!r} arrives in step 2; only 'upload' "
            f"is implemented")
    job_dir = params["_job_dir"]
    incoming = job_dir / "incoming"
    files = sorted(p for p in incoming.iterdir()) if incoming.exists() else []
    if not files:
        raise RuntimeError("no files were staged for this ingest job")

    project.raw.mkdir(parents=True, exist_ok=True)
    renamed: list[dict] = []
    probed = []
    for f in files:
        stem = prepare.sanitize_stem(f.stem)
        dst = project.raw / f"{stem}{f.suffix.lower()}"
        n = 1
        while dst.exists():
            dst = project.raw / f"{stem}-{n}{f.suffix.lower()}"
            n += 1
        shutil.move(str(f), dst)
        if dst.name != f.name:
            renamed.append({"original": f.name, "stored_as": dst.name})
        info = prepare.probe(dst)
        probed.append({"name": dst.name,
                       "codec": info.get("codec_name"),
                       "sample_rate": info.get("sample_rate"),
                       "channels": info.get("channels"),
                       "duration": info.get("duration")})
    return {"added": len(files), "renamed": renamed, "files": probed}


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
    result = execute(Path(argv[1]))
    _emit("RESULT", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
