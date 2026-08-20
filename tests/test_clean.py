"""Tests for clean: text repair, plan building, and apply."""
from pathlib import Path

import pytest

from piper_trainer import metadata
from piper_trainer.clean import (ABBREV, SYMBOLS, Plan, apply, build_plan,
                                 repair_text)
from piper_trainer.config import Project
from piper_trainer.validate import Finding


def make_project(tmp_path: Path, rows: list[tuple[str, str]],
                 wavs: list[str] | None = None) -> Project:
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    metadata.write(proj.metadata, rows)
    proj.wavs.mkdir(parents=True, exist_ok=True)
    for w in wavs or []:
        (proj.wavs / f"{w}.wav").write_bytes(b"RIFF..WAVE")
    return proj


# ---------------------------------------------------------------- repair_text

def test_repair_each_abbreviation():
    for pattern, replacement in ABBREV.items():
        # materialize the regex into plain text: r"\bMr\." -> "Mr."
        sample = pattern.replace(r"\b", "").replace(r"\\.", ".").replace("\\", "")
        out = repair_text(f"hello {sample} world")
        assert replacement in out, (pattern, out)


def test_repair_each_symbol():
    for sym, rep in SYMBOLS.items():
        out = repair_text(f"x {sym} y")
        assert rep.strip() in out, (sym, out)


def test_single_digits_expand():
    assert repair_text("room 7") == "room seven"
    assert repair_text("a 1 and 9") == "a one and nine"


def test_multi_digit_numbers_left_alone():
    assert repair_text("year 1984") == "year 1984"
    assert repair_text("the 2026 model") == "the 2026 model"


def test_whitespace_collapsed():
    assert repair_text("a   b") == "a b"
    assert repair_text("R&D") == "R and D"


def test_no_repair_needed_unchanged():
    text = "plain text with no problems at all"
    assert repair_text(text) == text


# --------------------------------------------------------------- build_plan

def test_build_plan_quarantine_implies_drop_rows(tmp_path):
    proj = make_project(tmp_path, [("a", "x"), ("b", "y")])
    findings = [Finding("warn", "orphan-wav", "orphan", ids=["a"])]
    plan = build_plan(proj, findings)
    assert plan.quarantine == {"a"}
    assert "a" in plan.drop_rows


def test_build_plan_only_filters(tmp_path):
    proj = make_project(tmp_path, [("a", "x"), ("b", "y")])
    findings = [Finding("error", "missing-wav", "", ids=["a"]),
                Finding("warn", "orphan-wav", "", ids=["b"])]
    plan = build_plan(proj, findings, only={"orphan-wav"})
    assert plan.quarantine == {"b"}
    assert plan.drop_rows == {"b"}


def test_build_plan_exclude_filters(tmp_path):
    proj = make_project(tmp_path, [("a", "x"), ("b", "y")])
    findings = [Finding("error", "missing-wav", "", ids=["a"]),
                Finding("warn", "orphan-wav", "", ids=["b"])]
    plan = build_plan(proj, findings, exclude={"orphan-wav"})
    assert plan.drop_rows == {"a"}
    assert plan.quarantine == set()


def test_build_plan_no_ids_no_actions(tmp_path):
    proj = make_project(tmp_path, [("a", "x")])
    findings = [Finding("error", "missing-wav", "no ids given")]
    plan = build_plan(proj, findings)
    assert plan.drop_rows == set()
    assert plan.quarantine == set()


def test_build_plan_informational_finding_no_action(tmp_path):
    proj = make_project(tmp_path, [("a", "x")])
    findings = [Finding("info", "duration", "5 clips", ids=["a"])]
    plan = build_plan(proj, findings)
    assert plan.drop_rows == set()
    assert plan.quarantine == set()
    assert plan.touched == set()


def test_build_plan_unspoken_text_repairs(tmp_path):
    proj = make_project(tmp_path, [("a", "room 7"), ("b", "clean")])
    findings = [Finding("error", "unspoken-text", "digits", ids=["a"])]
    plan = build_plan(proj, findings)
    assert plan.repairs == {"a": ("room 7", "room seven")}
    assert plan.unresolved == []


# --------------------------------------------------------------------- apply

def test_apply_max_fraction_guard(tmp_path):
    rows = [(f"c{i}", f"text {i}") for i in range(10)]
    proj = make_project(tmp_path, rows, wavs=[f"c{i}" for i in range(10)])
    findings = [Finding("warn", "short-clips", "short",
                        ids=[f"c{i}" for i in range(4)])]  # 40% > 0.34
    plan = build_plan(proj, findings)
    with pytest.raises(RuntimeError, match="refusing to remove"):
        apply(proj, plan)
    stats = apply(proj, plan, force=True)
    assert stats["quarantined"] == 4


def test_apply_moves_quarantined_files_and_rewrites_metadata(tmp_path):
    rows = [(f"c{i}", f"text {i}") for i in range(10)]
    proj = make_project(tmp_path, rows, wavs=[f"c{i}" for i in range(10)])
    findings = [Finding("warn", "orphan-wav", "orphan", ids=["c0", "c1"])]
    plan = build_plan(proj, findings)

    stats = apply(proj, plan)

    assert stats["rows_remaining"] == 8
    qdir = proj.dataset / "quarantine"
    assert (qdir / "c0.wav").exists()
    assert (qdir / "c1.wav").exists()
    assert not (proj.wavs / "c0.wav").exists()
    rows_after, problems = metadata.read(proj.metadata)
    ids = [cid for cid, _ in rows_after]
    assert "c0" not in ids and "c1" not in ids
    assert len(ids) == 8
    assert problems == []
    # clean log records the moves
    manifest = qdir / "manifest.csv"
    assert manifest.exists()
    content = manifest.read_text()
    assert "quarantine" in content and "c0" in content


def test_apply_drops_malformed_rows_and_counts_normalized(tmp_path):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    proj.metadata.write_text("a|ok\nbadline\nb|\nc|fine\n", encoding="utf-8")
    plan = Plan()
    plan.normalize_file = ["columns"]

    stats = apply(proj, plan)

    assert stats["normalized"] == 2  # badline + b|
    rows_after, problems = metadata.read(proj.metadata)
    assert rows_after == [("a", "ok"), ("c", "fine")]
    assert problems == []
    log = (proj.dataset / "clean-log.csv").read_text()
    assert "drop-row" in log and "columns" in log


def test_apply_repairs_text(tmp_path):
    proj = make_project(tmp_path, [("a", "room 7")])
    plan = Plan()
    plan.repairs = {"a": ("room 7", "room seven")}
    plan.reasons = {"a": ["unspoken-text"]}

    stats = apply(proj, plan)

    assert stats["repaired"] == 1
    rows_after, _ = metadata.read(proj.metadata)
    assert rows_after == [("a", "room seven")]
