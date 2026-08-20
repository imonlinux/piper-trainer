# Work order: metadata format contract (findings §1.1, §1.2, §1.3, §3.1, §3.2)

Fixes the three critical findings by consolidating metadata I/O into one module,
then adds the test suite that keeps it fixed.

Do the tasks in order. Task 1 is the contract everything else depends on.

---

## Background: what is actually broken

Three modules each implement `metadata.csv` I/O ad hoc, and they disagree:

| | writes | reads |
|---|---|---|
| `transcribe.py` | `csv.writer`, delimiter `\|`, QUOTE_NONE, escapechar `\` — **but default `lineterminator='\r\n'`** | — |
| `clean.py` | same writer, same CRLF defect | bare `line.split("\|")`, no unescaping |
| `validate.py` | — | bare `line.split("\|")`, no unescaping |

Consequences:

1. Every `metadata.csv` is written with CRLF. (§1.1)
2. `validate` cannot detect this, because it reads with `Path.read_text()`, which
   applies universal-newline translation and converts `\r\n` → `\n` before the
   check `if "\r\n" in raw` ever runs. **The CRLF check is dead code.** (not in
   the assessment)
3. Writers escape the delimiter (`a | b` → `a \| b`); readers never unescape, so
   the backslash survives into the transcript and reaches espeak. (§1.3)
4. `clean --apply` rewrites the file with CRLF regardless of input. (not in the
   assessment)

---

## Task 1 — new module `src/piper_trainer/metadata.py`

Single source of truth for the `id|text` format. **No other module may parse or
write `metadata.csv` after this change.**

```python
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


def write(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write metadata.csv with LF endings and correct escaping.

    MUST open with newline="" and pass lineterminator="\\n".
    """


def line_endings(path: Path) -> str:
    """Return "lf", "crlf", "mixed", or "none".

    MUST read bytes, not text — Path.read_text() translates newlines and makes
    CRLF undetectable. This function exists because that bug shipped.
    """
```

### Required semantics

- `read` skips a blank line but records it as `Problem(n, "blank-row", ...)`.
- A line with no delimiter, or with an empty text field, is
  `Problem(n, "columns", ...)` and is **not** included in rows.
- A trailing newline at end of file is normal and is not a blank row.
- Round trip is exact for text containing `|`, `\`, and unicode:
  `read(write(rows)) == rows`.
- `write` creates parent directories if needed.

### Do not

- Do not use `Path.read_text()` or `Path.write_text()` anywhere in this module.
- Do not add a header row.
- Do not strip or normalize the transcript text beyond what csv does.

---

## Task 2 — adopt the helper

Replace every ad-hoc implementation. No behavioural change except correctness.

**`transcribe.py`**
- Replace the `csv.writer(...)` block for metadata with `metadata.write(...)`.
- Leave `audit.csv` alone — it is a normal CSV and is not affected.

**`clean.py`**
- Delete `read_rows` and `write_rows`; import and use `metadata.read` / `metadata.write`.
- `read_rows` currently returns only rows; call sites that need problems should
  take the second element.

**`validate.py`**
- Delete the inline parsing loop (the `for i, line in enumerate(raw.splitlines())`
  block) and the `raw = project.metadata.read_text()` line.
- Use `metadata.read()` for rows, and map returned `Problem`s onto the existing
  `Finding` codes `blank-row` and `columns`, preserving the current messages and
  line numbers.
- Use `metadata.line_endings()` for the `crlf` finding.
- Remove the now-unused `import csv` (finding §3.2).

---

## Task 3 — implement `normalize_file` (finding §1.2)

`clean.build_plan` collects `crlf` / `columns` / `blank-row` into
`plan.normalize_file`, `describe()` prints it, and `apply()` ignores it.

Because `apply()` now round-trips through `metadata.read`/`write`, blank rows and
CRLF are corrected incidentally. That is not good enough — the code should mean
what it says.

In `apply()`, when `plan.normalize_file` is non-empty:

- Rewrite `metadata.csv` through `metadata.write` (this fixes crlf and blank-row).
- For `columns`: drop the malformed rows that `metadata.read` reported as
  `Problem(code="columns")`, and record each in the clean log with
  `action="drop-row"`, `reasons="columns"`.
- Include a `normalized` count in the returned stats dict.

If you disagree and would rather remove the field entirely, say so rather than
leaving it half-wired.

---

## Task 4 — fix `_project()` (not in the assessment)

`cli.py::_project` constructs `Project` directly and never reads the name from
`project.json`:

```python
return Project(root=..., name=args.name or Path(args.project).resolve().name)
```

So `init /workspace/marvin-voice --name marvin` saves `name: "marvin"`, and every
later command silently uses `marvin-voice`. This breaks cache paths, config
filenames, and the default export stem.

Fix: prefer the saved name. Precedence is `--name` > `project.json` > directory
name. Keep it a one-line-ish change; do not restructure `Project`.

---

## Task 5 — tests

Add `tests/` with pytest. No network, no GPU, no audio processing — pure logic
only. Add `pytest` to a `[project.optional-dependencies] dev` group in
`pyproject.toml`.

### `tests/test_metadata.py` (the point of this whole work order)

- round trip: plain ASCII, text containing `|`, text containing `\`,
  text containing `\|`, unicode, empty-ish text
- `write` produces LF only — assert on **bytes**: `b"\r\n" not in path.read_bytes()`
- `line_endings` correctly returns `lf` / `crlf` / `mixed` for hand-written byte
  fixtures
- `read` on a CRLF file still parses rows correctly (legacy files must load)
- blank line → `Problem("blank-row")`, row excluded
- `id|` and `id` → `Problem("columns")`, row excluded
- trailing newline is not reported as a blank row

### `tests/test_clean.py`

- `repair_text`: each abbreviation in `ABBREV`, each symbol in `SYMBOLS`,
  single digits expand, **multi-digit numbers are left alone**, whitespace is
  collapsed, text needing no repair is returned unchanged
- `build_plan`: `only` / `exclude` filtering; quarantine implies drop_rows;
  a finding with no ids produces no actions
- `apply`: the `MAX_FRACTION` guard raises without `force` and proceeds with it;
  quarantined files move; metadata is rewritten without the dropped ids

### `tests/test_validate.py`

- `BAD_TEXT`: the fourteen cases already verified —
  sentence-final `I.` / `A.` do **not** flag; `Mr. Smith`, `Dr. Who`,
  `St. Andrews`, `Inc.`, `Ave.`, digits, `&`, `%` do flag
- `validate_dataset` on a fixture project: correct codes for missing-wav,
  orphan-wav, sample-rate, short-clips, batch-size
- `Finding.action` maps correctly via `ACTIONS`

### `tests/test_train.py`

- `build_command`: medium tier emits **no** `--model.*` architecture flags;
  low/high emit their tier's flags; `resume` produces `--ckpt_path` and
  suppresses warmstart; `warmstart` alone produces `--model.warmstart_ckpt`;
  `--trainer.default_root_dir` points at `runs-<tier>`
- `latest_checkpoint`: prefers `last.ckpt`; picks the newest across multiple
  `version_N` directories; returns `None` when the runs dir is absent

### `tests/test_export.py`

- `verify` rejects a stem/dataset mismatch and a missing `language.code`
- the config patch adds `dataset`, `audio.quality`, and `language` without
  touching `phoneme_id_map`, `num_symbols`, or `num_speakers`

Use `tmp_path` fixtures. Generate the few needed WAVs with the `wave` module
(a sine tone is fine) — do not commit binary fixtures.

---

## Acceptance

```bash
pytest -q                      # all pass
grep -rn 'split("|")' src/     # no hits
grep -rn "read_text\|write_text" src/piper_trainer/metadata.py   # no hits
grep -rn "^import csv" src/piper_trainer/validate.py             # no hits
```

Then, on a real project:

```bash
piper-trainer transcribe <proj>   # or hand-write a metadata.csv
python3 -c "print(open('<proj>/dataset/metadata.csv','rb').read()[:200])"
# must show \n, never \r\n
piper-trainer validate <proj>     # no crlf finding
```

---

## For review

When done, provide:

1. `git diff --stat`
2. the full contents of `src/piper_trainer/metadata.py`
3. `git diff src/piper_trainer/validate.py src/piper_trainer/clean.py`
4. `pytest -q` output (tail)
5. a note on any instruction you disagreed with and what you did instead

Do **not** paste whole unchanged files.
