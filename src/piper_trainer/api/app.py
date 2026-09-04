"""FastAPI application — Bones, design doc §6.1 / §9.1.

A thin layer over the existing `piper_trainer.*` functions plus the job
manager. Everything long-running is a job; validation is synchronous
because it is fast and the UI wants it inline.

Run: uvicorn piper_trainer.api.app:app --host 127.0.0.1 --port 8000
(PIPER_WORKSPACE selects the volume; bind localhost only — no auth in v1,
design doc §7.)
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (FastAPI, File, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import doctor, peaks, prepare
from ..config import Project, TIERS
from ..lock import LockBusy
from . import catalog, dataset as dataset_mod, settings
from .jobs import JobError, JobManager

NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
UI_DIR = Path(__file__).resolve().parents[1] / "ui"


class ProjectCreate(BaseModel):
    name: str
    espeak_voice: str | None = None
    tier: str | None = None
    catalog_path: str | None = None


class JobCreate(BaseModel):
    kind: str
    stage: str | None = None
    params: dict = {}


class FetchCreate(BaseModel):
    catalog_path: str


class PreviewCreate(BaseModel):
    stage: str
    params: dict = {}


class TranscriptEdit(BaseModel):
    text: str


class SourceDelete(BaseModel):
    names: list[str]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_app(workspace: Path | None = None,
               runner_cmd=None) -> FastAPI:
    """runner_cmd is a test seam: the command the job manager executes for
    one job. Production uses `python -m piper_trainer.api.runner <job-dir>`."""
    ws_root = Path(workspace) if workspace else settings.workspace()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        kwargs = {"allow_parallel": settings.allow_parallel(),
                  "cancel_grace": settings.cancel_grace()}
        if runner_cmd is not None:
            kwargs["runner_cmd"] = runner_cmd
        manager = JobManager(ws_root, **kwargs)
        manager.rescan()
        app.state.manager = manager
        app.state.workspace = ws_root
        yield
        for t in list(manager._tasks):
            t.cancel()

    app = FastAPI(title="piper-trainer", version=settings.version(),
                  lifespan=lifespan)

    # ------------------------------------------------------------ helpers

    def manager() -> JobManager:
        return app.state.manager

    def project_or_404(project_id: str) -> Project:
        if not NAME_RE.match(project_id):
            raise HTTPException(400, f"invalid project id {project_id!r}")
        root = ws_root / project_id
        if not (root / "project.json").exists():
            raise HTTPException(404, f"no such project: {project_id}")
        return Project.load(root)

    def job_or_404(job_id: str) -> dict:
        try:
            return manager().get(job_id)
        except JobError:
            raise HTTPException(404, f"no such job: {job_id}") from None

    def project_stats(root: Project) -> dict:
        clips = len(list(root.wavs.glob("*.wav"))) if root.wavs.exists() else 0
        minutes = None
        if root.audit.exists():
            total = 0.0
            try:
                with root.audit.open(newline="") as fh:
                    rdr = csv.reader(fh)
                    next(rdr, None)  # header: clip, duration, cps, prob, text
                    for row in rdr:
                        try:
                            total += float(row[1])
                        except (ValueError, IndexError):
                            continue
                minutes = round(total / 60, 1)
            except OSError:
                pass
        tiers_trained = sorted(
            t for t in TIERS if (root.root / f"runs-{t}").exists())
        jobs = manager().list_for_project(root.root)
        last = {"id": jobs[0]["id"], "kind": jobs[0]["kind"],
                "state": jobs[0]["state"]} if jobs else None
        return {"name": root.name, "path": str(root.root), "clips": clips,
                "minutes": minutes, "tiers_trained": tiers_trained,
                "last_job": last,
                "prepare_params": root.get("prepare_params")}

    # -------------------------------------------------------------- system

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": settings.version()}

    @app.get("/api/doctor")
    def doctor_json():
        lines, ok = doctor.check()
        checks = []
        for line in lines:
            if line.startswith("✓ "):
                status, message = "ok", line[2:]
            elif line.startswith("✗ "):
                status, message = "error", line[2:]
            else:
                status, message = "info", line[2:] if line[:2] in ("· ",
                                                                   "  ") \
                    else line
            checks.append({"status": status, "message": message})
        # §3.8: report the transcription capability so the UI does not have
        # to infer it (CTranslate2 has a CUDA backend and no ROCm backend)
        devices = ["cpu"]
        try:
            import torch
            if torch.cuda.is_available() and not getattr(torch.version,
                                                         "hip", None):
                devices.append("cuda")
        except Exception:  # noqa: BLE001
            pass
        return {"ok": ok, "checks": checks,
                "transcribe_devices": devices}

    @app.get("/api/espeak-voices")
    def espeak_voices(prefix: str = ""):
        try:
            return doctor.espeak_voices(prefix)
        except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError):
            return []  # doctor reports the missing or failing binary

    @app.get("/api/tiers")
    def tiers():
        return TIERS

    # ------------------------------------------------------------ projects

    @app.get("/api/projects")
    def list_projects():
        out = []
        for d in manager().projects():
            try:
                out.append(project_stats(Project.load(d)))
            except Exception:  # noqa: BLE001 — one bad dir must not kill the list
                continue
        return out

    @app.post("/api/projects", status_code=201)
    def create_project(body: ProjectCreate):
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", body.name).strip("._")
        if not name:
            raise HTTPException(400, "project name normalizes to empty")
        root = ws_root / name
        if root.exists():
            raise HTTPException(409, f"already exists: {name}")
        proj = Project(root=root, name=name)
        proj.ensure()
        proj.set(name=name, espeak_voice=body.espeak_voice,
                 tier=body.tier, catalog_path=body.catalog_path)
        return project_stats(proj)

    @app.get("/api/projects/{project_id}")
    def project_detail(project_id: str):
        proj = project_or_404(project_id)

        def count(d: Path) -> int:
            return len(list(d.glob("*"))) if d.exists() else 0

        def count_audio(d: Path) -> int:
            # the raw/ card counts sources, not stray files (peaks used to
            # write caches into raw/ and triple the number — finding 8)
            if not d.exists():
                return 0
            return sum(1 for p in d.iterdir()
                       if p.is_file()
                       and p.suffix.lower() in prepare.AUDIO_EXT)

        rows, problems = [], []
        endings = None
        if proj.metadata.exists():
            from .. import metadata as metadata_mod
            rows, problems = metadata_mod.read(proj.metadata)
            endings = metadata_mod.line_endings(proj.metadata)
        return {
            **project_stats(proj),
            "config": {k: proj.get(k) for k in
                       ("espeak_voice", "tier", "catalog_path",
                        "target_epochs", "transcripts_provided")},
            "directories": {
                "raw": count_audio(proj.raw),
                "work/48k": count(proj.work48k),
                "work/denoised": count(proj.denoised),
                "work/clips": count(proj.clips),
                "dataset/wavs": count(proj.wavs),
                "dataset/quarantine": count(
                    proj.dataset / "quarantine"),
            },
            "dataset": {
                "rows": len(rows),
                "malformed_lines": len(problems),
                "line_endings": endings,
            },
            "voices": sorted(p.stem for p in proj.out.glob("*.onnx"))
            if proj.out.exists() else [],
            "checkpoints": local_checkpoints(proj),
            "jobs": manager().list_for_project(proj.root)[:10],
        }

    @app.delete("/api/projects/{project_id}")
    def delete_project(project_id: str):
        proj = project_or_404(project_id)
        running = [j for j in manager().list_for_project(proj.root)
                   if j["state"] in ("running", "queued")]
        if running:
            raise HTTPException(409, "project has active jobs: "
                                + ", ".join(j["id"] for j in running))
        trash = ws_root / ".trash"
        trash.mkdir(exist_ok=True)
        dest = trash / f"{project_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        # §0: nothing is destroyed — the directory is moved aside, never rm'd
        shutil.move(str(proj.root), dest)
        return {"moved_to": str(dest)}

    @app.get("/api/projects/{project_id}/files/{path:path}")
    def project_file(project_id: str, path: str):
        proj = project_or_404(project_id)
        full = (proj.root / path).resolve()
        root = proj.root.resolve()
        if not str(full).startswith(str(root) + "/"):
            raise HTTPException(400, "path escapes the project")
        if full.suffix.lower() not in prepare.PLAYABLE_EXT or not full.is_file():
            raise HTTPException(404, "not a project audio file")
        return FileResponse(full)

    @app.get("/api/projects/{project_id}/sources")
    def project_sources(project_id: str):
        proj = project_or_404(project_id)
        return prepare.sources(proj)

    @app.get("/api/projects/{project_id}/sources/{filename}/peaks")
    def source_peaks(project_id: str, filename: str,
                     channel: str = "downmix", buckets: int = 2000):
        """Waveform envelope for the tuner canvas (§8 decision 1). Basename
        only: peaks are for raw/ sources, never a path traversal target."""
        proj = project_or_404(project_id)
        if channel not in peaks.CHANNELS:
            raise HTTPException(400, f"channel must be one of {peaks.CHANNELS}")
        src = proj.raw / Path(filename).name
        if not peaks.is_audio(src) or not src.is_file():
            raise HTTPException(404, "not a source audio file")
        try:
            return peaks.compute_peaks(src, channel=channel, buckets=buckets)
        except subprocess.CalledProcessError as exc:
            raise HTTPException(422, "ffmpeg could not decode this file") from exc

    @app.post("/api/projects/{project_id}/sources/delete")
    def delete_sources(project_id: str, body: SourceDelete):
        """Bulk-remove raw/ sources by moving them to .trash (§0: nothing
        is destroyed). Refuses while a job that reads raw/ is live."""
        proj = project_or_404(project_id)
        if not body.names:
            raise HTTPException(400, "no sources named")
        busy = [j["id"] for j in manager().list_for_project(proj.root)
                if j["state"] in ("running", "queued")
                and j["kind"] in ("ingest", "prepare", "preview")]
        if busy:
            raise HTTPException(409, "jobs reading raw/ are active: "
                                + ", ".join(busy))
        try:
            return prepare.delete_sources(proj, body.names)
        except LockBusy as exc:
            raise HTTPException(409, str(exc)) from exc

    # -------------------------------------------------------------- ingest

    @app.post("/api/projects/{project_id}/ingest", status_code=202)
    async def ingest(project_id: str,
                     files: list[UploadFile] = File(...)):
        proj = project_or_404(project_id)
        if not files:
            raise HTTPException(400, "no files uploaded")
        # refuse what the pipeline can never recognise, before any byte is
        # staged: an unchecked upload landed in raw/, vanished from
        # sources() and still inflated the directory card (finding 9)
        rejected = [f.filename for f in files
                    if Path(f.filename or "").suffix.lower()
                    not in prepare.AUDIO_EXT]
        if rejected:
            raise HTTPException(400, "not a recognized audio type: "
                                + ", ".join(rejected))
        # Stage every byte before a runner can exist. submit(run=False)
        # creates the job record without enqueueing it; the runner starts
        # only after the last chunk lands. Submitting first used to let the
        # supervisor's first await race the upload loop, and _ingest moved
        # half-written files into raw/ (review finding 1 — silent data
        # loss on any upload larger than one read chunk).
        job = await manager().submit(proj.root, "ingest",
                                     params={"source_type": "upload"},
                                     run=False)
        job_dir = manager().job_dir(job["id"])
        incoming = job_dir / "incoming"
        incoming.mkdir()
        for f in files:
            dest = incoming / Path(f.filename or "unnamed").name
            with dest.open("wb") as out:
                while chunk := await f.read(1 << 20):
                    out.write(chunk)
        try:
            return await manager().start(job["id"])
        except JobError:
            # cancelled while the upload was still streaming — the record
            # already says so; report that state instead of failing
            return manager().get(job["id"])

    # ---------------------------------------------------------------- jobs

    @app.post("/api/projects/{project_id}/jobs", status_code=202)
    async def create_job(project_id: str, body: JobCreate):
        proj = project_or_404(project_id)
        params = dict(body.params or {})
        if body.kind == "prepare":
            # promote saves the tuner's winning dials as prepare_params;
            # a plain "run prepare" starts from those, explicit params win
            params = {**(proj.get("prepare_params") or {}), **params}
        try:
            return await manager().submit(proj.root, body.kind,
                                          stage=body.stage,
                                          params=params)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/projects/{project_id}/jobs")
    def list_jobs(project_id: str):
        proj = project_or_404(project_id)
        return manager().list_for_project(proj.root)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return job_or_404(job_id)

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, tail: int | None = None):
        job_or_404(job_id)
        return PlainTextResponse(manager().log(job_id, tail_bytes=tail))

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        job_or_404(job_id)
        try:
            return await manager().cancel(job_id)
        except JobError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.post("/api/jobs/{job_id}/start")
    async def start_job(job_id: str):
        job_or_404(job_id)
        try:
            return await manager().start(job_id)
        except JobError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.websocket("/api/jobs/{job_id}/stream")
    async def stream_job(ws: WebSocket, job_id: str):
        try:
            manager().job_dir(job_id)
        except JobError:
            await ws.close(code=4404)
            return
        await ws.accept()
        q = manager().subscribe(job_id)
        try:
            await ws.send_json({"type": "state", "job": manager().get(job_id)})
            # bounded: a refresh on a multi-hour training run must not
            # re-read the whole log into one frame (review finding 6)
            await ws.send_json({"type": "log_reset",
                                "text": manager().log(job_id, tail_bytes=262144)})
            while True:
                event = await q.get()
                await ws.send_json(event)
        except WebSocketDisconnect:
            pass
        finally:
            manager().unsubscribe(job_id, q)

    # ---------------------------------------------------------- checkpoints

    @app.get("/api/checkpoints/catalog")
    def checkpoints_catalog(refresh: bool = False):
        return catalog.catalog(force=refresh)

    @app.get("/api/checkpoints/catalog/{path:path}")
    def checkpoints_detail(path: str):
        try:
            return catalog.detail(path)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        except KeyError:
            raise HTTPException(404, f"not in catalog: {path}") from None

    @app.post("/api/projects/{project_id}/checkpoints/fetch",
              status_code=202)
    async def fetch_checkpoint(project_id: str, body: FetchCreate):
        proj = project_or_404(project_id)
        return await manager().submit(
            proj.root, "fetch-checkpoint",
            params={"catalog_path": body.catalog_path})

    def local_checkpoints(proj: Project) -> list[dict]:
        out = []
        for path, entry in (proj.get("base_checkpoints") or {}).items():
            out.append({"source": "catalog", "catalog_path": path,
                        "dir": entry.get("dir"),
                        "files": entry.get("files"),
                        "fetched_at": entry.get("fetched_at")})
        for tier in TIERS:
            runs = proj.root / f"runs-{tier}"
            if not runs.exists():
                continue
            for ckpt in sorted(runs.glob(
                    "lightning_logs/version_*/checkpoints/*.ckpt")):
                epoch = None
                try:
                    from .. import train as train_mod
                    epoch = train_mod.checkpoint_epoch(ckpt)
                except Exception:  # noqa: BLE001 — torch may be absent
                    pass
                out.append({"source": "run", "tier": tier,
                            "path": str(ckpt.relative_to(proj.root)),
                            "name": ckpt.name, "epoch": epoch,
                            "mtime": datetime.fromtimestamp(
                                ckpt.stat().st_mtime,
                                tz=timezone.utc).strftime(
                                "%Y-%m-%dT%H:%M:%SZ")})
        return out

    @app.get("/api/projects/{project_id}/checkpoints")
    def project_checkpoints(project_id: str):
        return local_checkpoints(project_or_404(project_id))

    @app.get("/api/projects/{project_id}/train/preview")
    def train_preview(project_id: str, epochs: int = 0, batch_size: int = 32):
        """Wall-clock projection for the Train screen (§6.4). Honest by
        construction: steps math from the dataset, and a seconds/epoch
        figure only when a previous train job actually ran here —
        otherwise the field stays null and the UI says so."""
        proj = project_or_404(project_id)
        if epochs < 1:
            raise HTTPException(400, "epochs must be >= 1")
        if batch_size < 1:
            raise HTTPException(400, "batch_size must be >= 1")
        clips = 0
        if proj.metadata.exists():
            from .. import metadata as metadata_mod
            clips = len(metadata_mod.read(proj.metadata)[0])
        tier = proj.get("tier") or "medium"
        steps = (clips + batch_size - 1) // batch_size if clips else None
        out: dict = {"clips": clips, "epochs": epochs,
                     "batch_size": batch_size,
                     "sample_rate": TIERS.get(tier, {}).get("sample_rate"),
                     "steps_per_epoch": steps,
                     "total_steps": steps * epochs if steps else None,
                     "seconds_per_epoch": None,
                     "projected_seconds": None, "basis": None}
        done = [j for j in manager().list_for_project(proj.root)
                if j["kind"] == "train" and j["state"] == "succeeded"
                and j["started_at"] and j["finished_at"]]
        if done:
            last = max(done, key=lambda j: j["finished_at"])
            started = datetime.fromisoformat(last["started_at"])
            wall = (datetime.fromisoformat(last["finished_at"])
                    - started).total_seconds()
            epoch_now = None
            try:
                from .. import train as train_mod
                ck = train_mod.latest_checkpoint(proj, tier)
                epoch_now = (train_mod.checkpoint_epoch(ck) if ck else None)
            except Exception:  # noqa: BLE001 — torch may be absent
                pass
            if epoch_now:
                out["seconds_per_epoch"] = round(wall / epoch_now, 1)
                out["projected_seconds"] = round(wall / epoch_now * epochs)
                out["basis"] = (f"last run took {wall / 3600:.1f} h "
                                f"to reach epoch {epoch_now}")
        return out

    # ------------------------------------------------------------- previews
    # §2/§4.5: a preview is a job variant — same lifecycle, same log
    # streaming — but short-lived, non-destructive and scoped to a sample.
    # It writes only to work/preview/<stage>/<preview-id>/ (the runner
    # enforces this); preview.json records the parameters so promote can
    # replay them as a full run.

    @app.post("/api/projects/{project_id}/preview", status_code=202)
    async def create_preview(project_id: str, body: PreviewCreate):
        proj = project_or_404(project_id)
        try:
            return await manager().submit(
                proj.root, "preview",
                params={**body.params, "stage": body.stage})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/projects/{project_id}/previews")
    def list_previews(project_id: str):
        proj = project_or_404(project_id)
        out = []
        base = proj.root / "work" / "preview"
        if base.exists():
            for mf in sorted(base.glob("*/*/preview.json")):
                try:
                    data = json.loads(mf.read_text())
                except (json.JSONDecodeError, OSError):
                    continue  # a running preview has no preview.json yet
                data["dir"] = str(mf.parent.relative_to(proj.root))
                out.append(data)
        # job ids are UTC-stamped, so lexicographic sort is newest-first
        return sorted(out, key=lambda p: p.get("id", ""), reverse=True)

    @app.post("/api/projects/{project_id}/previews/{preview_id}/promote",
              status_code=202)
    async def promote_preview(project_id: str, preview_id: str):
        proj = project_or_404(project_id)
        if not NAME_RE.match(preview_id):
            raise HTTPException(400, "bad preview id")
        matches = sorted(
            (proj.root / "work" / "preview").glob(f"*/{preview_id}/preview.json"))
        if not matches:
            raise HTTPException(404, "no such preview")
        data = json.loads(matches[0].read_text())
        params = dict(data.get("params") or {})
        params.pop("stage", None)
        # _prepare maps the tuner's parameters (pad -> leading/trailing
        # silence, denoise -> denoise_enabled) exactly as the preview ran them
        proj.set(prepare_params=params)
        # saved so the project page's "run prepare" (empty params) replays
        # the promoted settings instead of falling back to code defaults —
        # a later re-prepare at defaults would otherwise re-segment at
        # energy 55 and quietly drop every quiet source again
        return await manager().submit(proj.root, "prepare", params=params)

    @app.delete("/api/projects/{project_id}/previews")
    def prune_previews(project_id: str):
        proj = project_or_404(project_id)
        shutil.rmtree(proj.root / "work" / "preview", ignore_errors=True)
        return {"pruned": True}

    # --------------------------------------------------------------- dataset
    # Audit screen (§6.3): the dataset table and inline transcript editing.
    # Validation findings and clean/restore are job kinds — the UI reads the
    # latest validate/clean job's result and submits clean jobs like any other.

    @app.get("/api/projects/{project_id}/dataset")
    def project_dataset(project_id: str):
        proj = project_or_404(project_id)
        return {"rows": dataset_mod.rows(proj),
                "quarantine": dataset_mod.quarantine(proj)}

    @app.patch("/api/projects/{project_id}/dataset/{clip_id}")
    def edit_transcript(project_id: str, clip_id: str,
                        body: TranscriptEdit):
        proj = project_or_404(project_id)
        try:
            return dataset_mod.set_text(proj, clip_id, body.text)
        except dataset_mod.ClipNotFound as exc:
            raise HTTPException(404, str(exc)) from None
        except dataset_mod.BadText as exc:
            raise HTTPException(400, str(exc)) from None
        except LockBusy as exc:
            raise HTTPException(409, str(exc)) from None

    # --------------------------------------------------------------- static

    @app.get("/")
    def index():
        return RedirectResponse("/ui/")

    if UI_DIR.exists():
        # Pre-cutover bookmarks pointed at /ui/app/#/...; the hash fragment
        # is client-side only, so a 308 to /ui/ keeps it (browsers re-attach
        # the original fragment when Location has none).
        @app.get("/ui/app", include_in_schema=False)
        @app.get("/ui/app/{rest:path}", include_in_schema=False)
        def legacy_ui_app(rest: str = ""):
            return RedirectResponse("/ui/", status_code=308)

        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True),
                  name="ui")

        # Heuristic browser caching served stale JS after pulls, so a fixed
        # UI looked unfixed. no-cache keeps the ETag 304 but forces a
        # revalidation round-trip on every load.
        @app.middleware("http")
        async def revalidate_ui(request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/ui"):
                response.headers["Cache-Control"] = "no-cache"
            return response

    return app


app = create_app()
