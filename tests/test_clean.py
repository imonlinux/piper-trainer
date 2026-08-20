"""Tests for clean: text repair, plan building, and apply."""
import csv
from pathlib import Path

import pytest

from piper_trainer import metadata
from piper_trainer.clean import (ABBREV, SYMBOLS, Plan, apply, build_plan,
                                 describe, repair_text, restore)
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


def test_build_plan_unresolved_names_only_unrepairable_ids(tmp_path):
    """A finding covering several ids where only some repair: ONE new
    finding carrying exactly the unrepairable ids (Task 6)."""
    proj = make_project(tmp_path, [("a", "room 7"), ("b", "year 1984"),
                                   ("c", "gate 42 open")])
    findings = [Finding("error", "unspoken-text", "digits in 3 clips",
                        ids=["a", "b", "c"])]
    plan = build_plan(proj, findings)
    assert set(plan.repairs) == {"a"}
    assert len(plan.unresolved) == 1
    assert plan.unresolved[0].ids == ["b", "c"]


def test_build_plan_collects_file_level_findings_into_normalize_file(tmp_path):
    proj = make_project(tmp_path, [("a", "x")])
    findings = [
        Finding("error", "crlf", "metadata.csv has CRLF line endings"),
        Finding("error", "columns", "2 row(s) lack a transcript"),
        Finding("error", "blank-row", "line 3 is blank"),
    ]
    plan = build_plan(proj, findings)
    assert sorted(plan.normalize_file) == ["blank-row", "columns", "crlf"]
    # file-level fixes never touch individual rows
    assert plan.touched == set()


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
    content = "".join(f"c{i}|text {i}\n" for i in range(10)) + "badline\nb|\n"
    proj.metadata.write_text(content, encoding="utf-8")
    plan = Plan()
    plan.normalize_file = ["columns"]

    stats = apply(proj, plan)

    assert stats["malformed_rows_dropped"] == 2  # badline + b|
    assert stats["line_endings_fixed"] is False  # file was already LF
    rows_after, problems = metadata.read(proj.metadata)
    assert rows_after == [(f"c{i}", f"text {i}") for i in range(10)]
    assert problems == []
    log = (proj.dataset / "clean-log.csv").read_text()
    assert "drop-row" in log and "columns" in log


def test_apply_preserves_malformed_rows_not_in_plan(tmp_path):
    """--only that excludes columns must not drop malformed rows."""
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    content = ("a|ok\nbadline,with,commas\nb|\nc|fine\n"
               "d|five\ne|six\n")
    proj.metadata.write_text(content, encoding="utf-8")
    plan = Plan()  # normalize_file empty: user narrowed with --only elsewhere
    plan.drop_rows = {"a"}
    plan.reasons = {"a": ["missing-wav"]}

    stats = apply(proj, plan)

    assert stats["malformed_rows_dropped"] == 0
    # malformed lines survive verbatim; kept rows fill the other positions
    assert proj.metadata.read_text() == \
        "c|fine\nbadline,with,commas\nb|\nd|five\ne|six\n"
    rows, problems = metadata.read(proj.metadata)
    assert rows == [("c", "fine"), ("d", "five"), ("e", "six")]
    assert len(problems) == 2


def test_apply_gates_blank_rows_too_and_fixes_endings(tmp_path):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    proj.metadata.write_bytes(
        b"a|ok\r\n\r\nb|\nc|fine\r\nd|four\r\ne|five\r\nf|six\r\n")
    plan = Plan()
    plan.normalize_file = ["crlf", "blank-row"]  # columns NOT gated in

    stats = apply(proj, plan)

    assert stats["malformed_rows_dropped"] == 1  # the blank line only
    assert stats["line_endings_fixed"] is True
    data = proj.metadata.read_bytes()
    assert b"\r" not in data
    assert data == b"a|ok\nc|fine\nb|\nd|four\ne|five\nf|six\n"  # 'b|' kept


def test_apply_preserves_blank_lines_not_in_plan(tmp_path):
    """A blank line the user did not ask to fix survives the rewrite too."""
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    proj.metadata.write_bytes(b"a|ok\r\n\r\nb|\nc|fine\r\n")
    plan = Plan()
    plan.normalize_file = ["crlf"]  # blank-row NOT gated

    stats = apply(proj, plan)

    assert stats["malformed_rows_dropped"] == 0
    assert stats["line_endings_fixed"] is True
    assert proj.metadata.read_bytes() == b"a|ok\n\nb|\nc|fine\n"


def test_apply_max_fraction_counts_gated_malformed(tmp_path):
    """6 good rows + 5 malformed, columns gated in: 0 + 5 over 6 + 5 > 0.34."""
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    content = "".join(f"c{i}|text {i}\n" for i in range(6)) + "bad\n" * 5
    proj.metadata.write_text(content, encoding="utf-8")
    plan = Plan()
    plan.normalize_file = ["columns"]
    with pytest.raises(RuntimeError, match="refusing to remove 5/11"):
        apply(proj, plan)
    stats = apply(proj, plan, force=True)
    assert stats["malformed_rows_dropped"] == 5
    assert stats["rows_remaining"] == 6


def test_apply_missing_metadata_raises(tmp_path):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        apply(proj, Plan())


# --------------------------------------------------------------------- describe

def test_describe_distinguishes_line_ending_repairs():
    plan = Plan()
    plan.normalize_file = ["crlf"]
    lines = describe(plan, 0, endings="crlf")
    assert any("converted CRLF line endings to LF" in ln for ln in lines)
    lines = describe(plan, 0, endings="mixed")
    assert any("converted CRLF line endings to LF" in ln for ln in lines)
    lines = describe(plan, 0, endings="none")
    assert any("added missing final newline" in ln for ln in lines)
    # no endings info supplied: neutral phrasing, no crash
    assert describe(plan, 0)


def test_apply_reports_which_ending_repair(tmp_path):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    proj.metadata.write_bytes(b"a|ok\r\n")
    plan = Plan()
    plan.normalize_file = ["crlf"]
    stats = apply(proj, plan)
    assert stats["line_endings_fixed"] is True
    assert stats["line_endings_repair"] == "converted CRLF line endings to LF"
    proj.metadata.write_bytes(b"a|ok")  # no final newline at all
    stats = apply(proj, plan)
    assert stats["line_endings_repair"] == "added missing final newline"


# -------------------------------------------------------------------- restore

def seed_quarantine(proj: Project, entries: list[tuple], files: list[str]):
    """entries: (timestamp, clip_id, action, reasons, text) rows for the
    manifest; files: stems to place in quarantine/."""
    qdir = proj.dataset / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    with (qdir / "manifest.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "clip_id", "action", "reasons", "text"])
        w.writerows(entries)
    for stem in files:
        (qdir / f"{stem}.wav").write_bytes(b"RIFF")


def test_restore_recovers_rows_from_manifest(tmp_path):
    proj = make_project(tmp_path, [("keep1", "stays")])
    seed_quarantine(proj, [
        ("20260101T000000Z", "gone1", "quarantine", "short-clips",
         "the first clip"),
        ("20260101T000000Z", "gone2", "quarantine", "cps-outliers",
         "the second clip"),
    ], files=["gone1", "gone2"])

    stats = restore(proj)

    assert stats["files_restored"] == 2
    assert stats["rows_restored"] == 2
    assert stats["conflicts"] == []
    rows, _ = metadata.read(proj.metadata)
    assert ("gone1", "the first clip") in rows
    assert ("gone2", "the second clip") in rows
    assert rows[0] == ("keep1", "stays")  # existing rows keep their position
    assert not (proj.dataset / "quarantine" / "gone1.wav").exists()
    assert (proj.wavs / "gone1.wav").exists()


def test_restore_files_only_leaves_metadata_alone(tmp_path):
    proj = make_project(tmp_path, [("keep1", "stays")])
    seed_quarantine(proj, [
        ("20260101T000000Z", "gone1", "quarantine", "short-clips", "text"),
    ], files=["gone1"])
    stats = restore(proj, files_only=True)
    assert stats["files_restored"] == 1
    assert stats["rows_restored"] == 0
    rows, _ = metadata.read(proj.metadata)
    assert rows == [("keep1", "stays")]


def test_restore_most_recent_manifest_entry_wins(tmp_path):
    proj = make_project(tmp_path, [])
    seed_quarantine(proj, [
        ("20260101T000000Z", "c1", "quarantine", "short-clips", "old text"),
        ("20260201T000000Z", "c1", "quarantine", "cps-outliers",
         "newer text"),
    ], files=["c1"])
    stats = restore(proj)
    assert stats["rows_restored"] == 1
    rows, _ = metadata.read(proj.metadata)
    assert rows == [("c1", "newer text")]


def test_restore_does_not_overwrite_existing_row(tmp_path):
    proj = make_project(tmp_path, [("c1", "hand-edited text")])
    seed_quarantine(proj, [
        ("20260101T000000Z", "c1", "quarantine", "short-clips",
         "manifest text"),
    ], files=["c1"])
    stats = restore(proj)
    # file comes back, row does not: the counts legitimately differ
    assert stats["files_restored"] == 1
    assert stats["rows_restored"] == 0
    assert stats["conflicts"] == ["c1"]
    rows, _ = metadata.read(proj.metadata)
    assert rows == [("c1", "hand-edited text")]


def test_restore_preserves_malformed_lines(tmp_path):
    proj = Project(root=tmp_path, name="t")
    proj.ensure()
    proj.metadata.write_bytes(b"keep|row\nbadline\n")
    seed_quarantine(proj, [
        ("20260101T000000Z", "gone1", "quarantine", "short-clips", "text"),
    ], files=["gone1"])
    restore(proj)
    data = proj.metadata.read_bytes()
    assert data == b"keep|row\ngone1|text\nbadline\n" or \
        data == b"keep|row\nbadline\ngone1|text\n"  # malformed survives


def test_restore_without_manifest_still_moves_files(tmp_path):
    proj = make_project(tmp_path, [("keep1", "stays")])
    qdir = proj.dataset / "quarantine"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "orphan.wav").write_bytes(b"RIFF")
    stats = restore(proj)
    assert stats["files_restored"] == 1
    assert stats["rows_restored"] == 0
    assert (proj.wavs / "orphan.wav").exists()


def test_apply_repairs_text(tmp_path):
    proj = make_project(tmp_path, [("a", "room 7")])
    plan = Plan()
    plan.repairs = {"a": ("room 7", "room seven")}
    plan.reasons = {"a": ["unspoken-text"]}

    stats = apply(proj, plan)

    assert stats["repaired"] == 1
    rows_after, _ = metadata.read(proj.metadata)
    assert rows_after == [("a", "room seven")]
