"""Canonical reader/writer for dataset/metadata.csv.

The format is `clip_id|transcript`, one row per line, LF endings, with the
delimiter escaped by backslash when it appears in text. Three modules used to
implement this independently and disagreed about all three of those things.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

DELIMITER = "|"
ESCAPECHAR = "\\"
LINETERMINATOR = "\n"

_DIALECT = dict(delimiter=DELIMITER, quoting=csv.QUOTE_NONE,
                escapechar=ESCAPECHAR, lineterminator=LINETERMINATOR)


@dataclass(frozen=True)
class Problem:
    line_no: int          # 1-based
    code: str             # "blank-row" | "columns"
    detail: str


def read(path: Path) -> tuple[list[tuple[str, str]], list[Problem]]:
    """Parse metadata.csv.

    Returns (rows, problems). Rows are (clip_id, text) with escaping resolved.
    Malformed lines are reported in `problems`, never silently dropped.

    MUST open with newline="" so csv sees the raw line endings.
    """
    rows: list[tuple[str, str]] = []
    problems: list[Problem] = []
    with path.open(newline="") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                problems.append(Problem(n, "blank-row", f"line {n} is blank"))
                continue
            fields = next(csv.reader([line], **_DIALECT))
            if len(fields) < 2 or not fields[1].strip():
                problems.append(Problem(n, "columns",
                                        f"line {n} lacks a transcript"))
                continue
            rows.append((fields[0], DELIMITER.join(fields[1:])))
    return rows, problems


def write(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write metadata.csv with LF endings and correct escaping.

    MUST open with newline="" and pass lineterminator="\\n".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        csv.writer(fh, **_DIALECT).writerows(rows)


def line_endings(path: Path) -> str:
    """Return "lf", "crlf", "mixed", or "none".

    MUST read bytes, not text — text-mode reads translate newlines and make
    CRLF undetectable. This function exists because that bug shipped.
    """
    data = path.read_bytes()
    crlf = data.count(b"\r\n")
    lf = data.replace(b"\r\n", b"").count(b"\n")
    if crlf == 0 and lf == 0:
        return "none"
    if crlf and not lf:
        return "crlf"
    if lf and not crlf:
        return "lf"
    return "mixed"
