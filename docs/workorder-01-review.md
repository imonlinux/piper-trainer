# Work order 01 — Review package

**Work order:** `docs/workorder-01-metadata-contract.md` (metadata format contract)
**Date:** 2026-08-20
**Status:** all five tasks complete; nothing committed — tree left for review.

---

## 1. `git diff --stat`

```
 pyproject.toml                  |  7 +++++++
 src/piper_trainer/clean.py      | 40 ++++++++++++++++++----------------------
 src/piper_trainer/cli.py        | 16 +++++++++++-----
 src/piper_trainer/transcribe.py |  6 ++----
 src/piper_trainer/validate.py   | 25 ++++++++++--------------
 5 files changed, 49 insertions(+), 45 deletions(-)
```

New (untracked) files, not in the diff stat:

| File | Purpose |
|---|---|
| `src/piper_trainer/metadata.py` | Task 1 — canonical metadata I/O |
| `tests/test_metadata.py` | 18 tests |
| `tests/test_clean.py` | 14 tests |
| `tests/test_validate.py` | 10 tests |
| `tests/test_train.py` | 10 tests |
| `tests/test_export.py` | 5 tests |
| `.gitignore` | added (see §5) |

---

## 2. Full contents of `src/piper_trainer/metadata.py`

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
```

Design notes:

- `read()` parses **per physical line** (`newline=""`), so `Problem.line_no`
  is exact, and a trailing newline produces no phantom blank row.
- With `QUOTE_NONE` + `escapechar`, the csv reader unescapes `\|` and `\\`,
  making `read(write(rows)) == rows` exact — including text containing `|`,
  `\`, and `\|`.
- Rows with more than one unescaped delimiter (legacy/hand-edited files) are
  tolerated: extra fields are re-joined with `|`, matching the old readers.
- CRLF files parse correctly (legacy datasets load); detection is separate
  via `line_endings()`.

---

## 3. `git diff src/piper_trainer/validate.py src/piper_trainer/clean.py`

```diff
diff --git a/src/piper_trainer/clean.py b/src/piper_trainer/clean.py
index 5fd4cf1..9615465 100644
--- a/src/piper_trainer/clean.py
+++ b/src/piper_trainer/clean.py
@@ -19,6 +19,7 @@ from dataclasses import dataclass, field
 from datetime import datetime, timezone
 from pathlib import Path
 
+from . import metadata
 from .config import Project
 from .validate import Finding
 
@@ -76,28 +77,12 @@ def repair_text(text: str) -> str:
     return re.sub(r"\s+", " ", out).strip()
 
 
-def read_rows(project: Project) -> list[tuple[str, str]]:
-    rows = []
-    for line in project.metadata.read_text().splitlines():
-        if not line.strip():
-            continue
-        parts = line.split("|")
-        if len(parts) >= 2:
-            rows.append((parts[0], "|".join(parts[1:])))
-    return rows
-
-
-def write_rows(project: Project, rows: list[tuple[str, str]]) -> None:
-    with project.metadata.open("w", newline="") as fh:
-        csv.writer(fh, delimiter="|", quoting=csv.QUOTE_NONE,
-                   escapechar="\\").writerows(rows)
-
-
 def build_plan(project: Project, findings: list[Finding],
                only: set[str] | None = None,
                exclude: set[str] | None = None) -> Plan:
     plan = Plan()
-    rows = dict(read_rows(project)) if project.metadata.exists() else {}
+    rows = dict(metadata.read(project.metadata)[0]) \
+        if project.metadata.exists() else {}
 
     for f in findings:
         code = f.code
@@ -164,7 +149,7 @@ def describe(plan: Plan, total_rows: int) -> list[str]:
 
 
 def apply(project: Project, plan: Plan, force: bool = False) -> dict:
-    rows = read_rows(project)
+    rows, problems = metadata.read(project.metadata)
     total = len(rows)
     if total and len(plan.drop_rows) / total > MAX_FRACTION and not force:
         raise RuntimeError(
@@ -173,6 +158,10 @@ def apply(project: Project, plan: Plan, force: bool = False) -> dict:
             f"validation threshold is wrong, not the data. Use --force to "
             f"override, or narrow with --only.")
 
+    col_problems = [p for p in problems if p.code == "columns"]
+    normalized = len(col_problems) + \
+        len([p for p in problems if p.code == "blank-row"])
+
     qdir = project.dataset / "quarantine"
     stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
     moved = []
@@ -191,9 +180,11 @@ def apply(project: Project, plan: Plan, force: bool = False) -> dict:
         if cid in plan.repairs:
             text = plan.repairs[cid][1]
         kept.append((cid, text))
-    write_rows(project, kept)
+    # Rewriting through the canonical writer incidentally removes CRLF line
+    # endings, blank rows, and the malformed rows reported as columns Problems.
+    metadata.write(project.metadata, kept)
 
-    if plan.quarantine or plan.drop_rows or plan.repairs:
+    if plan.normalize_file or plan.quarantine or plan.drop_rows or plan.repairs:
         manifest = qdir / "manifest.csv" if plan.quarantine else \
             project.dataset / "clean-log.csv"
         manifest.parent.mkdir(parents=True, exist_ok=True)
@@ -203,6 +194,10 @@ def apply(project: Project, plan: Plan, force: bool = False) -> dict:
             if new:
                 w.writerow(["timestamp", "clip_id", "action", "reasons", "text"])
             row_text = dict(rows)
+            # malformed rows have no usable clip id; identify by line number
+            for p in col_problems:
+                w.writerow([stamp, f"line {p.line_no}", "drop-row",
+                            "columns", p.detail])
             for cid in sorted(plan.quarantine):
                 w.writerow([stamp, cid, "quarantine",
                             ";".join(plan.reasons.get(cid, [])),
@@ -217,7 +212,8 @@ def apply(project: Project, plan: Plan, force: bool = False) -> dict:
                             f"{old} -> {new_t}"])
 
     return {"repaired": len(plan.repairs), "quarantined": len(moved),
-            "rows_removed": total - len(kept), "rows_remaining": len(kept)}
+            "rows_removed": total - len(kept) + len(col_problems),
+            "rows_remaining": len(kept), "normalized": normalized}
 
 
 def restore(project: Project, clip_ids: list[str] | None = None) -> int:
diff --git a/src/piper_trainer/validate.py b/src/piper_trainer/validate.py
index 844cd6d..ec01265 100644
--- a/src/piper_trainer/validate.py
+++ b/src/piper_trainer/validate.py
@@ -1,13 +1,13 @@
 """Pre-flight checks. Every rule here exists because it cost someone an hour."""
 from __future__ import annotations
 
-import csv
 import re
 import statistics
 import subprocess
 import wave
 from pathlib import Path
 
+from . import metadata
 from .config import Project, TIERS
 
 # What espeak-ng will mangle if left in a transcript.
@@ -84,23 +84,20 @@ def validate_dataset(project: Project, tier: str = "medium",
         return [Finding("error", "no-metadata",
                         f"{project.metadata} does not exist")]
 
-    raw = project.metadata.read_text()
-    if "\r\n" in raw:
+    endings = metadata.line_endings(project.metadata)
+    if endings == "crlf":
         f.append(Finding("error", "crlf",
                          "metadata.csv has CRLF line endings — strip with "
                          "sed -i 's/\\r$//'"))
+    elif endings == "mixed":
+        f.append(Finding("error", "crlf",
+                         "metadata.csv has mixed line endings (LF and CRLF) "
+                         "— normalize to LF (e.g. sed -i 's/\\r$//')"))
 
-    rows, bad_cols = [], []
-    for i, line in enumerate(raw.splitlines(), start=1):
-        if not line.strip():
-            f.append(Finding("error", "blank-row", f"line {i} is blank"))
-            continue
-        parts = line.split("|")
-        if len(parts) < 2 or not parts[1].strip():
-            bad_cols.append(i)
-            continue
-        rows.append((parts[0], "|".join(parts[1:])))
-
+    rows, problems = metadata.read(project.metadata)
+    for p in (p for p in problems if p.code == "blank-row"):
+        f.append(Finding("error", "blank-row", f"line {p.line_no} is blank"))
+    bad_cols = [p.line_no for p in problems if p.code == "columns"]
     if bad_cols:
         f.append(Finding("error", "columns",
                          f"{len(bad_cols)} row(s) lack a transcript "
```

---

## 4. `pytest -q` output (tail)

```
.............................................................          [100%]
61 passed in 0.12s
```

Breakdown: `test_metadata.py` 18, `test_clean.py` 14, `test_validate.py` 10,
`test_train.py` 10, `test_export.py` 5, plus 4 in-module doctest-adjacent
helpers counted within those files. Pure-logic only; no network, GPU, or
audio processing. WAV fixtures are generated with the `wave` module (sine
tones) inside `tmp_path`.

### Acceptance checks (work order §Acceptance)

| Check | Result |
|---|---|
| `pytest -q` all pass | 61 passed |
| `grep -rn 'split("|")' src/` | no hits |
| `grep -rn "read_text\|write_text" src/piper_trainer/metadata.py` | no hits |
| `grep -rn "^import csv" src/piper_trainer/validate.py` | no hits |

"On a real project" check (substitute for `transcribe`, which needs the
container's Whisper): hand-written `metadata.csv` with mixed LF/CRLF endings
and a `|` inside a transcript, plus three generated 22.05 kHz WAVs:

```
$ piper-trainer validate proj
✗ [crlf] (repair) metadata.csv has mixed line endings (LF and CRLF) — normalize...
rc=1
$ piper-trainer clean proj --apply
· normalize metadata.csv (crlf)
repaired: 0, quarantined: 0, rows_removed: 0, rows_remaining: 3, normalized: 0
$ piper-trainer validate proj
· [duration] 3 clips, 0.1 min total, median 2.0s
rc=0
$ od -c dataset/metadata.csv   # LF only, pipe escaped as \|
...   \   |  ...  \n ...
```

Also verified live: Task 4 name precedence
(`no --name -> smoke`, `--name other -> other`).

---

## 5. Disagreements / judgment calls

1. **`line_endings` docstring rephrased.** The work order's prescribed
   docstring literally contained the string `Path.read_text()` — which its
   own acceptance grep (`read_text\|write_text` → no hits) forbids. The two
   instructions contradict each other. I kept the intent ("text-mode reads
   translate newlines and make CRLF undetectable") and dropped the literal
   token so the acceptance check passes.

2. **Task 3 semantics chosen where the order was silent:**
   - `normalized` counts every line the rewrite removes (all `columns` +
     all `blank-row` problems), regardless of whether that code is in
     `plan.normalize_file` — the rewrite removes them either way, and the
     count should reflect what actually happened to the file.
   - Malformed rows are logged with `clip_id = "line {n}"`: the prescribed
     `Problem` shape (unchanged) carries no clip id, and a malformed row
     often has no usable one.
   - `rows_removed` now includes dropped malformed rows
     (`total - len(kept) + len(col_problems)`); the `MAX_FRACTION` guard
     still considers only plan-driven drops.

3. **Task 4 uses the tolerant `.meta()`** (corrupt `project.json` →
   directory-name fallback) rather than `Project.load`, which raises on
   corrupt JSON and would break *every* command for that project. The
   precedence `--name` > `project.json` > directory name is as specified.

4. **Extras not in the work order** (drop if unwanted):
   - `[tool.pytest.ini_options] pythonpath = ["src"]` — the package isn't
     installed in the dev environment; this lets tests import it without an
     editable install.
   - `.gitignore` (`__pycache__/`, `*.pyc`, `.pytest_cache/`) — added after
     `git add -A` briefly staged bytecode.

5. **`transcribe.py` keeps `import csv`** — still used for `audit.csv`,
   which the work order explicitly said to leave alone.

6. **Test interpretation:** "round trip ... empty-ish text" was read as
   minimal non-empty text ("A"); a truly empty text field is separately
   specified as a `columns` Problem and cannot round-trip by design. Both
   behaviors are tested.
