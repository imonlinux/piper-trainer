"""Workorder-04 A2: the /stages readiness contract.

compute() is pure, so most tests drive it with hand-written job
records over a real on-disk Project. One smoke test exercises the
endpoint itself.
"""
from __future__ import annotations

import json
from fastapi.testclient import TestClient

from piper_trainer.api.app import create_app
from piper_trainer.api import stages
from piper_trainer.config import Project


def make_proj(tmp_path, name="hal"):
    proj = Project(root=tmp_path / name, name=name)
    proj.ensure()
    (proj.root / "project.json").write_text(
        json.dumps({"name": name, "tier": "medium"}))
    return proj


def job(kind, state, finished_at=None, created_at="2026-09-01T00:00:00Z",
        params=None, result=None, jid="j1"):
    return {"id": jid, "kind": kind, "state": state, "params": params or {},
            "created_at": created_at, "finished_at": finished_at,
            "result": result, "progress": None, "error": None}


def stage(data, name):
    return data["stages"][name]


def test_fresh_project_sources_ready_rest_locked(tmp_path):
    data = stages.compute(make_proj(tmp_path), [])
    assert stage(data, "sources")["status"] == "ready"
    assert stage(data, "sources")["requirements"] == [
        {"met": False, "text": "0 source files in raw/"}]
    for s in ("prepare", "transcribe", "audit", "train", "voices"):
        assert stage(data, s)["status"] == "locked", s
    assert stage(data, "transcribe")["blocked_by"] == "0 clips in dataset/wavs"
    assert stage(data, "voices")["blocked_by"] == \
        "no checkpoint yet — run training first"
    assert data["next"] == {"stage": "sources", "why": "add your first source"}
    assert data["running"] == {"job": None, "more": 0}
    assert data["chain"] is None
    assert data["project"]["voice"] == "hal-medium"


def test_sources_unlock_prepare_only(tmp_path):
    proj = make_proj(tmp_path)
    (proj.raw / "a.wav").write_bytes(b"x")
    data = stages.compute(proj, [])
    # files exist without an ingest job -> the entry point reads as done
    assert stage(data, "sources")["status"] == "done"
    assert stage(data, "prepare")["status"] == "ready"
    assert stage(data, "prepare")["blocked_by"] is None
    assert stage(data, "transcribe")["status"] == "locked"


def test_wavs_unlock_transcribe_rows_unlock_audit_train(tmp_path):
    proj = make_proj(tmp_path)
    (proj.raw / "a.wav").write_bytes(b"x")
    proj.metadata.write_text("c1|hello world\n")
    data = stages.compute(proj, [])
    # metadata rows are audit/train's input, not transcribe's: that
    # stage consumes dataset/wavs and CREATES metadata.csv
    for s in ("audit", "train"):
        assert stage(data, s)["status"] == "ready", s
    assert stage(data, "transcribe")["status"] == "locked", "transcribe"
    assert stage(data, "transcribe")["blocked_by"] == "0 clips in dataset/wavs"
    (proj.wavs / "c1.wav").write_bytes(b"x")
    data = stages.compute(proj, [])
    assert stage(data, "transcribe")["status"] == "ready", "transcribe"
    assert stage(data, "voices")["status"] == "locked"
    assert data["next"]["stage"] == "prepare"


def test_transcribe_gate_counts_wavs_not_metadata(tmp_path):
    # Regression (James, 2026-09-09): after a successful prepare the
    # transcribe page said "not ready: 0 clips in the dataset" while the
    # run button worked fine — the gate counted metadata.csv rows, the
    # file transcription itself creates.
    proj = make_proj(tmp_path)
    (proj.wavs / "c1.wav").write_bytes(b"x")
    assert not proj.metadata.exists()
    data = stages.compute(proj, [])
    st = stage(data, "transcribe")
    assert st["status"] == "ready"
    assert st["requirements"] == [
        {"met": True, "text": "1 clips in dataset/wavs"}]


def test_precedence_active_over_attn_over_done(tmp_path):
    proj = make_proj(tmp_path)
    proj.metadata.write_text("c1|hello\n")
    jobs = [
        job("prepare", "succeeded", finished_at="2026-09-01T01:00:00Z"),
        job("transcribe", "failed", finished_at="2026-09-01T02:00:00Z",
            jid="j2"),
    ]
    data = stages.compute(proj, jobs)
    assert stage(data, "prepare")["status"] == "done"
    assert stage(data, "transcribe")["status"] == "attn"
    assert data["next"] == {"stage": "transcribe",
                            "why": "last transcribe failed"}
    # a queued job outranks the failure and moves `next`
    jobs.append(job("transcribe", "queued", created_at="2026-09-01T03:00:00Z",
                    jid="j3"))
    data = stages.compute(proj, jobs)
    assert stage(data, "transcribe")["status"] == "active"
    assert stage(data, "transcribe")["active_job"]["id"] == "j3"
    assert data["next"] == {"stage": "transcribe", "why": "running now"}
    assert data["running"]["job"]["id"] == "j3"


def test_earlier_stage_rerun_supersedes_done(tmp_path):
    proj = make_proj(tmp_path)
    (proj.raw / "a.wav").write_bytes(b"x")
    (proj.wavs / "c1.wav").write_bytes(b"x")
    proj.metadata.write_text("c1|hello\n")
    jobs = [
        job("transcribe", "succeeded", finished_at="2026-09-01T01:00:00Z"),
        job("prepare", "succeeded", finished_at="2026-09-01T02:00:00Z",
            jid="j2"),
    ]
    data = stages.compute(proj, jobs)
    assert stage(data, "transcribe")["status"] == "ready"
    assert data["next"] == {
        "stage": "transcribe",
        "why": "an earlier stage re-ran — this stage is stale"}


def test_audit_findings_fresh_and_stale(tmp_path):
    proj = make_proj(tmp_path)
    proj.metadata.write_text("c1|hello\n")
    findings = [{"level": "error"}, {"level": "error"}, {"level": "warn"}]
    jobs = [job("validate", "succeeded",
                finished_at="2026-09-01T01:00:00Z",
                result={"findings": findings, "errors": 2})]
    data = stages.compute(proj, jobs)
    assert data["findings"] == {"errors": 2, "warnings": 1, "stale": False}
    assert stage(data, "audit")["status"] == "attn"
    assert data["next"] == {"stage": "audit",
                            "why": "validation found 2 errors"}
    # a later clean apply makes the counts stale, not wrong
    jobs.append(job("clean", "succeeded", jid="j2",
                    finished_at="2026-09-01T02:00:00Z",
                    params={"apply": True}, result={"stats": {}}))
    data = stages.compute(proj, jobs)
    assert data["findings"]["stale"] is True
    assert data["findings"]["stale_reason"] == \
        "clean applied after this validation"
    assert data["next"]["why"] == \
        "dataset changed — run validation to refresh"
    # a clean PLAN (apply falsy) must not stale the counts
    jobs[-1]["params"] = {}
    data = stages.compute(proj, jobs)
    assert data["findings"]["stale"] is False


def test_restore_marks_audit_stale_with_its_own_reason(tmp_path):
    proj = make_proj(tmp_path)
    proj.metadata.write_text("c1|hello\n")
    jobs = [
        job("validate", "succeeded", finished_at="2026-09-01T01:00:00Z",
            result={"findings": [], "errors": 0}),
        job("restore", "succeeded", finished_at="2026-09-01T02:00:00Z",
            jid="j2", result={"stats": {}}),
    ]
    data = stages.compute(proj, jobs)
    assert data["findings"]["stale"] is True
    assert data["findings"]["stale_reason"] == \
        "a clip was restored after this validation"


def test_preview_jobs_map_by_params_stage(tmp_path):
    proj = make_proj(tmp_path)
    data = stages.compute(proj, [
        job("preview", "running", params={"stage": "train"}),
        job("preview", "running", params={"stage": "audition"}, jid="j2"),
        job("preview", "running", params={"stage": "denoise"}, jid="j3"),
    ])
    assert stage(data, "train")["status"] == "active"
    assert stage(data, "voices")["status"] == "active"
    assert stage(data, "prepare")["status"] == "active"


def test_utility_job_never_lights_a_stage(tmp_path):
    proj = make_proj(tmp_path)
    data = stages.compute(proj, [
        job("fetch-checkpoint", "running"),
    ])
    assert all(stage(data, s)["status"] != "active" for s in stages.STAGES)
    assert data["running"]["job"]["kind"] == "fetch-checkpoint"
    assert data["running"]["more"] == 0


def test_checkpoint_unlocks_voices_and_done_projects_audition(tmp_path):
    proj = make_proj(tmp_path)
    proj.metadata.write_text("c1|hello\n")
    ck = proj.root / "runs-medium" / "lightning_logs" / "version_0" \
        / "checkpoints"
    ck.mkdir(parents=True)
    (ck / "epoch=3-step=12.ckpt").write_bytes(b"x")
    data = stages.compute(proj, [])
    assert stage(data, "voices")["status"] == "ready"
    assert stage(data, "voices")["requirements"][0]["text"] == \
        "checkpoint ready in runs-medium"
    # everything done -> next points at audition
    done = "2026-09-01T09:00:00Z"
    jobs = [
        job("ingest", "succeeded", finished_at=done),
        job("prepare", "succeeded", finished_at=done, jid="j2"),
        job("transcribe", "succeeded", finished_at=done, jid="j3"),
        job("validate", "succeeded", finished_at=done, jid="j4",
            result={"findings": [], "errors": 0}),
        job("train", "succeeded", finished_at=done, jid="j5"),
        job("export", "succeeded", finished_at=done, jid="j6"),
    ]
    data = stages.compute(proj, jobs)
    assert all(stage(data, s)["status"] == "done" for s in stages.STAGES)
    assert data["next"] == {"stage": "voices", "why": "audition your voice"}


def test_stages_endpoint_smoke(tmp_path):
    app = create_app(tmp_path, runner_cmd=lambda jd: ["true"])
    with TestClient(app) as c:
        r = c.post("/api/projects", json={"name": "smoke"})
        assert r.status_code in (200, 201), r.text
        r = c.get("/api/projects/smoke/stages")
        assert r.status_code == 200, r.text
        body = r.json()
        assert list(body["stages"].keys()) == list(stages.STAGES)
        assert body["next"]["stage"] == "sources"
        assert body["chain"] is None


def test_preview_never_marks_prepare_done(tmp_path):
    # A succeeded preview is an experiment, not the stage's work
    # product: prepare stays ready and the next card keeps pointing at
    # it. last_job reports real pipeline jobs only.
    proj = make_proj(tmp_path)
    (proj.raw / "a.wav").write_bytes(b"x")
    jobs = [job("preview", "succeeded", finished_at="2026-09-01T00:00:10Z",
                params={"stage": "segment"}, jid="p1")]
    data = stages.compute(proj, jobs)
    assert stage(data, "prepare")["status"] == "ready"
    assert stage(data, "prepare")["last_job"] is None


def test_failed_preview_does_not_raise_attention(tmp_path):
    proj = make_proj(tmp_path)
    (proj.raw / "a.wav").write_bytes(b"x")
    jobs = [job("preview", "failed", finished_at="2026-09-01T00:00:10Z",
                params={"stage": "segment"}, jid="p1")]
    data = stages.compute(proj, jobs)
    assert stage(data, "prepare")["status"] == "ready"


def test_next_falls_back_to_locked_frontier(tmp_path):
    # All stages behind the frontier read done and nothing is runnable:
    # the next card points at the first locked stage with its blocker,
    # not at audition.
    proj = make_proj(tmp_path)
    (proj.raw / "a.wav").write_bytes(b"x")
    jobs = [job("prepare", "succeeded", finished_at="2026-09-01T00:00:10Z",
                jid="j1")]
    data = stages.compute(proj, jobs)
    assert stage(data, "prepare")["status"] == "done"
    assert data["next"] == {"stage": "transcribe",
                            "why": "0 clips in dataset/wavs"}


def test_later_preview_does_not_supersede_downstream(tmp_path):
    # Previews write nothing, so a preview after a full transcribe must
    # not mark the dataset stale.
    proj = make_proj(tmp_path)
    (proj.raw / "a.wav").write_bytes(b"x")
    proj.metadata.write_text("c1|hello world\n")
    jobs = [
        job("prepare", "succeeded", finished_at="2026-09-01T00:00:10Z",
            jid="j1"),
        job("transcribe", "succeeded", finished_at="2026-09-01T00:01:00Z",
            jid="j2"),
        job("preview", "succeeded", finished_at="2026-09-01T00:02:00Z",
            params={"stage": "segment"}, jid="p1"),
    ]
    data = stages.compute(proj, jobs)
    assert stage(data, "transcribe")["status"] == "done"
