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
    raw: str = ""         # the line verbatim, sans terminator; for write-back


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
                problems.append(Problem(n, "blank-row", f"line {n} is blank",
                                        raw=line.rstrip("\r\n")))
                continue
            fields = next(csv.reader([line], **_DIALECT))
            if len(fields) < 2 or not fields[1].strip():
                problems.append(Problem(n, "columns",
                                        f"line {n} lacks a transcript",
                                        raw=line.rstrip("\r\n")))
                continue
            rows.append((fields[0], DELIMITER.join(fields[1:])))
    return rows, problems


def write(path: Path, rows: list[tuple[str, str]],
          raw_lines: dict[int, str] | None = None) -> None:
    """Write metadata.csv with LF endings and correct escaping.

    MUST open with newline="" and pass lineterminator="\\n".

    raw_lines maps a 1-based line number to a verbatim line (sans terminator)
    to preserve at that position — used by clean to keep malformed rows it
    was not asked to fix. Rows fill the remaining positions in order. A raw
    line keeps its bytes except the terminator: line endings are always LF.
    """
    raw_lines = raw_lines or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = list(rows)
    with path.open("w", newline="") as fh:
        wtr = csv.writer(fh, **_DIALECT)
        for n in range(1, len(rows) + len(raw_lines) + 1):
            if n in raw_lines:
                fh.write(raw_lines[n] + LINETERMINATOR)
            else:
                wtr.writerow(pending.pop(0))


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
