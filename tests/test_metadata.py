"""Tests for the canonical metadata.csv reader/writer."""
from pathlib import Path

import pytest

from piper_trainer import metadata


def round_trip(tmp_path: Path, rows: list[tuple[str, str]]) -> tuple:
    p = tmp_path / "sub" / "metadata.csv"
    metadata.write(p, rows)
    return metadata.read(p)


def test_round_trip_plain_ascii(tmp_path):
    rows = [("a", "hello world"), ("b", "second line of text")]
    got, problems = round_trip(tmp_path, rows)
    assert got == rows
    assert problems == []


def test_round_trip_text_with_delimiter(tmp_path):
    rows = [("a", "one | two")]
    got, problems = round_trip(tmp_path, rows)
    assert got == rows
    assert problems == []


def test_round_trip_text_with_backslash(tmp_path):
    rows = [("a", "back\\slash text")]
    got, problems = round_trip(tmp_path, rows)
    assert got == rows
    assert problems == []


def test_round_trip_text_with_escaped_delimiter(tmp_path):
    rows = [("a", "a\\|b")]
    got, problems = round_trip(tmp_path, rows)
    assert got == rows
    assert problems == []


def test_round_trip_unicode(tmp_path):
    rows = [("a", "héllo — wörld “quotes” ✓")]
    got, problems = round_trip(tmp_path, rows)
    assert got == rows
    assert problems == []


def test_round_trip_minimal_text(tmp_path):
    rows = [("a", "A")]
    got, problems = round_trip(tmp_path, rows)
    assert got == rows
    assert problems == []


def test_empty_text_is_a_problem_not_a_row(tmp_path):
    p = tmp_path / "metadata.csv"
    metadata.write(p, [("a", "ok"), ("b", "")])
    got, problems = metadata.read(p)
    assert got == [("a", "ok")]
    assert len(problems) == 1
    assert problems[0].code == "columns"
    assert problems[0].line_no == 2


def test_write_produces_lf_only(tmp_path):
    p = tmp_path / "metadata.csv"
    metadata.write(p, [("a", "hello"), ("b", "world")])
    data = p.read_bytes()
    assert b"\r\n" not in data
    assert b"\r" not in data
    assert data.endswith(b"\n")


def test_write_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "metadata.csv"
    metadata.write(p, [("a", "x")])
    assert p.exists()


def test_line_endings_lf(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"a|b\nc|d\n")
    assert metadata.line_endings(p) == "lf"


def test_line_endings_crlf(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"a|b\r\nc|d\r\n")
    assert metadata.line_endings(p) == "crlf"


def test_line_endings_mixed(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"a|b\nc|d\r\n")
    assert metadata.line_endings(p) == "mixed"


def test_line_endings_none(tmp_path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"a|b")
    assert metadata.line_endings(p) == "none"
    p.write_bytes(b"")
    assert metadata.line_endings(p) == "none"


def test_read_crlf_file_parses_rows(tmp_path):
    """Legacy CRLF files must still load."""
    p = tmp_path / "metadata.csv"
    p.write_bytes(b"one|first line\r\ntwo|second line\r\n")
    rows, problems = metadata.read(p)
    assert rows == [("one", "first line"), ("two", "second line")]
    assert problems == []


def test_read_crlf_with_escaped_delimiter(tmp_path):
    p = tmp_path / "metadata.csv"
    p.write_bytes(b"one|a \\| b\r\n")
    rows, problems = metadata.read(p)
    assert rows == [("one", "a | b")]
    assert problems == []


def test_read_legacy_unescaped_pipe_in_text(tmp_path):
    """Pre-contract files never escaped the delimiter; they must still load
    with the text intact — this is why read rejoins every field after the
    first instead of taking fields[1] alone."""
    p = tmp_path / "metadata.csv"
    p.write_bytes(b"a|one | two\nb|x | y | z\n")
    rows, problems = metadata.read(p)
    assert rows == [("a", "one | two"), ("b", "x | y | z")]
    assert problems == []


def test_blank_line_reported_and_excluded(tmp_path):
    p = tmp_path / "metadata.csv"
    p.write_bytes(b"a|b\n\nc|d\n")
    rows, problems = metadata.read(p)
    assert rows == [("a", "b"), ("c", "d")]
    assert len(problems) == 1
    assert problems[0].code == "blank-row"
    assert problems[0].line_no == 2


def test_line_without_delimiter_is_columns(tmp_path):
    p = tmp_path / "metadata.csv"
    p.write_bytes(b"onlyid\na|b\n")
    rows, problems = metadata.read(p)
    assert rows == [("a", "b")]
    assert problems[0].code == "columns"
    assert problems[0].line_no == 1


def test_empty_text_field_is_columns(tmp_path):
    p = tmp_path / "metadata.csv"
    p.write_bytes(b"a|\nb|ok\n")
    rows, problems = metadata.read(p)
    assert rows == [("b", "ok")]
    assert problems[0].code == "columns"
    assert problems[0].line_no == 1


def test_trailing_newline_is_not_a_blank_row(tmp_path):
    p = tmp_path / "metadata.csv"
    p.write_bytes(b"a|b\n")
    rows, problems = metadata.read(p)
    assert rows == [("a", "b")]
    assert problems == []
