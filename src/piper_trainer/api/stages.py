"""Project readiness (workorder-04 A2): the GET /stages view.

Derived entirely from the project directory and the job records that
already exist — no subprocess, no re-validation (spec principle 4).
Requirement texts name their numbers, and `blocked_by` is always the
first unmet requirement's text, never a hardcoded stage name.

`compute()` is a pure function of (project, job records): the endpoint
supplies live jobs, the tests supply hand-written ones.
"""
from __future__ import annotations

from .. import config, metadata as metadata_mod, prepare as prepare_mod

STAGES = ("sources", "prepare", "transcribe", "audit", "train", "voices")

# Job kind → pipeline stage. Preview jobs carry theirs in params.stage
# (segmentation and denoise ARE preparation work); utility kinds map to
# no stage at all — a checkpoint fetch must not light up a pipeline dot.
KIND_STAGE = {
    "ingest": "sources",
    "prepare": "prepare",
    "transcribe": "transcribe",
    "validate": "audit",
    "clean": "audit",
    "restore": "audit",
    "train": "train",
    "export": "voices",
}
PREVIEW_STAGE = {
    "segment": "prepare",
    "segment-all": "prepare",
    "denoise": "prepare",
    "train": "train",
    "audition": "voices",
}
UTILITY_KINDS = frozenset({"fetch-checkpoint"})
ACTIVE_STATES = frozenset({"queued", "running"})


def _ts(job: dict) -> str:
    """Sortable ordering key; _now() writes ISO-8601 UTC, so plain
    lexicographic comparison is correct."""
    return job.get("finished_at") or job.get("created_at") or ""


def _ref(job: dict | None) -> dict | None:
    if job is None:
        return None
    return {"id": job.get("id"), "kind": job.get("kind"),
            "state": job.get("state"), "progress": job.get("progress"),
            "error": job.get("error"),
            "finished_at": job.get("finished_at")}


def stage_of(job: dict) -> str | None:
    kind = job.get("kind")
    if kind == "preview":
        return PREVIEW_STAGE.get((job.get("params") or {}).get("stage"))
    if kind in UTILITY_KINDS:
        return None
    return KIND_STAGE.get(kind)


def _audit_findings(audit_jobs: list[dict]) -> dict | None:
    """The audit read set, honest about freshness: `errors`/`warnings`
    come only from the newest succeeded VALIDATE job — clean apply and
    restore results carry repair stats, not error counts — and a later
    mutate marks the counts stale instead of recomputing them."""
    vals = [j for j in audit_jobs
            if j.get("kind") == "validate" and j.get("state") == "succeeded"
            and isinstance(j.get("result"), dict)]
    if not vals:
        return None
    v = max(vals, key=lambda j: _ts(j))
    res = v["result"]
    findings = res.get("findings") or []
    errors = int(res.get("errors") or 0)
    out: dict = {"errors": errors,
                 "warnings": max(0, len(findings) - errors),
                 "stale": False}
    mutators = [j for j in audit_jobs
                if j.get("kind") in ("clean", "restore")
                and j.get("state") == "succeeded"
                and (j.get("kind") == "restore"
                     or (j.get("params") or {}).get("apply"))
                and j.get("finished_at")]
    if mutators:
        m = max(mutators, key=lambda j: _ts(j))
        if m["finished_at"] > (v.get("finished_at") or ""):
            out["stale"] = True
            out["stale_reason"] = (
                "a clip was restored after this validation"
                if m.get("kind") == "restore"
                else "clean applied after this validation")
    return out


def compute(proj: config.Project, jobs: list[dict]) -> dict:
    tier = proj.get("tier") or "medium"
    rows: list[tuple[str, str]] = []
    if proj.metadata.exists():
        rows, _ = metadata_mod.read(proj.metadata)
    n_rows = len(rows)
    # transcribe's INPUT is the prepared clips, not metadata.csv — that
    # file is transcribe's OUTPUT. Gating on rows locked the stage right
    # after a successful prepare.
    n_wavs = (sum(1 for p in proj.wavs.glob("*.wav"))
              if proj.wavs.exists() else 0)
    n_src = (sum(1 for p in proj.raw.iterdir()
                 if p.is_file() and p.suffix.lower() in prepare_mod.AUDIO_EXT)
             if proj.raw.exists() else 0)
    n_quarantine = (len(list((proj.dataset / "quarantine").glob("*")))
                    if (proj.dataset / "quarantine").exists() else 0)
    has_ckpt = bool(list(proj.root.glob(
        "runs-*/lightning_logs/version_*/checkpoints/*.ckpt")))
    voices = sorted(p.stem for p in proj.out.glob("*.onnx")) \
        if proj.out.exists() else []

    stage_jobs: dict[str, list[dict]] = {s: [] for s in STAGES}
    for j in jobs:
        s = stage_of(j)
        if s is not None:
            stage_jobs[s].append(j)

    findings = _audit_findings(stage_jobs["audit"])

    def requirements(stage: str) -> list[dict]:
        if stage == "sources":
            return [{"met": n_src > 0,
                     "text": f"{n_src} source files in raw/"}]
        if stage == "prepare":
            return [{"met": n_src > 0,
                     "text": f"{n_src} source files to segment"}]
        if stage == "transcribe":
            return [{"met": n_wavs > 0,
                     "text": f"{n_wavs} clips in dataset/wavs"}]
        if stage == "audit":
            text = f"{n_rows} clips in the dataset"
            if n_quarantine:
                text += f" · {n_quarantine} quarantined"
            return [{"met": n_rows > 0, "text": text}]
        if stage == "train":
            return [{"met": n_rows > 0,
                     "text": f"{n_rows} clips to train on"}]
        return [{"met": has_ckpt,
                 "text": (f"checkpoint ready in runs-{tier}" if has_ckpt
                          else "no checkpoint yet — run training first")}]

    # last terminal job and active jobs per stage; supersede check: a
    # succeeded stage is only "done" while no earlier stage finished
    # after it. Previews are prepare-stage experiments: a running one
    # still shows the stage as active, but done, failed-attention and
    # supersede are decided by real pipeline jobs only, because a
    # preview writes nothing to the dataset.
    last: dict[str, dict | None] = {}
    active: dict[str, list[dict]] = {}
    superseded: dict[str, bool] = {}
    for i, stage in enumerate(STAGES):
        sj = stage_jobs[stage]
        real = [j for j in sj if j.get("kind") != "preview"]
        done = [j for j in real if j.get("finished_at")]
        last[stage] = max(done, key=_ts) if done else None
        active[stage] = sorted(
            (j for j in sj if j.get("state") in ACTIVE_STATES),
            key=_ts)
        sup = False
        if last[stage] and last[stage]["state"] == "succeeded":
            t = last[stage]["finished_at"] or ""
            for earlier in STAGES[:i]:
                for j in stage_jobs[earlier]:
                    if (j.get("kind") != "preview"
                            and j.get("state") == "succeeded"
                            and (j.get("finished_at") or "") > t):
                        sup = True
                        break
                if sup:
                    break
        superseded[stage] = sup

    out_stages: dict[str, dict] = {}
    for stage in STAGES:
        reqs = requirements(stage)
        if active[stage]:
            status = "active"
        elif stage == "audit" and findings and (
                findings["errors"] > 0 or findings["stale"]):
            status = "attn"
        elif last[stage] and last[stage]["state"] == "failed":
            status = "attn"
        elif (last[stage] and last[stage]["state"] == "succeeded"
                and not superseded[stage]):
            status = "done"
        elif stage == "sources":
            # The entry point never locks: with no files yet it is the
            # state the next card sends you to fix; once raw/ holds
            # audio (a manual drop, no ingest job needed) the stage's
            # work product exists and it reads as done.
            status = "done" if n_src > 0 else "ready"
        elif all(r["met"] for r in reqs):
            status = "ready"
        else:
            status = "locked"
        out_stages[stage] = {
            "status": status,
            "requirements": reqs,
            "blocked_by": (next((r["text"] for r in reqs if not r["met"]),
                                None)
                           if status == "locked" else None),
            "active_job": _ref(active[stage][-1] if active[stage] else None),
            "last_job": _ref(last[stage]),
        }

    # next: running first, then the first thing that needs you, then the
    # first runnable thing, then audition.
    nxt: dict = {"stage": "voices", "why": "audition your voice"}
    for stage in STAGES:
        if out_stages[stage]["status"] == "active":
            nxt = {"stage": stage, "why": "running now"}
            break
    else:
        for stage in STAGES:
            if out_stages[stage]["status"] != "attn":
                continue
            if stage == "audit" and findings and findings["stale"]:
                why = "dataset changed — run validation to refresh"
            elif stage == "audit" and findings:
                why = f"validation found {findings['errors']} errors"
            else:
                why = f"last {last[stage]['kind']} failed" if last[stage] \
                    else "needs attention"
            nxt = {"stage": stage, "why": why}
            break
        else:
            for stage in STAGES:
                if out_stages[stage]["status"] != "ready":
                    continue
                if stage == "sources" and n_src == 0:
                    why = "add your first source"
                elif superseded[stage]:
                    why = "an earlier stage re-ran — this stage is stale"
                else:
                    why = f"{stage} has not run yet"
                nxt = {"stage": stage, "why": why}
                break
            else:
                # Nothing actionable: point at the frontier (first
                # locked stage) and say what is blocking it, instead of
                # skipping ahead to audition.
                for stage in STAGES:
                    if out_stages[stage]["status"] == "locked":
                        nxt = {"stage": stage,
                               "why": out_stages[stage]["blocked_by"]
                               or "earlier stages first"}
                        break

    running = sorted((j for j in jobs
                      if j.get("state") in ACTIVE_STATES), key=_ts)
    return {
        "project": {"name": proj.name,
                    "voice": config.voice_stem(proj.name, tier)},
        "stages": out_stages,
        "next": nxt,
        "findings": findings,
        "running": {"job": _ref(running[-1]) if running else None,
                    "more": max(0, len(running) - 1)},
        "chain": None,  # Phase B (workorder-04 B1)
    }
