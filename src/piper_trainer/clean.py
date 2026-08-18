"""Act on validation findings.

Design rules, learned from datasets that were already too small:

1.  **Nothing is deleted.** Clips move to `dataset/quarantine/` with a manifest
    recording why. Disk is free; a discarded clip you cannot recover is not.
2.  **Text problems are repaired, not removed.** Dropping a clip because its
    transcript says "2026" throws away good audio over a text problem.
3.  **Dry run by default.** `--apply` is required to touch anything.
4.  **Refuse a mass cull.** If a run would quarantine more than a third of the
    dataset, the validation threshold was wrong, not the data.
"""
from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Project
from .validate import Finding

MAX_FRACTION = 0.34

# Text repairs. Deliberately conservative: expand what is unambiguous, and
# leave anything requiring judgement to the human (validate still flags it).
NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
# Keep in sync with validate.KNOWN_ABBREV — validate flags them, clean expands
# them. Anything not here is still flagged by the mid-sentence heuristic but
# left for a human, since the expansion would be a guess.
ABBREV = {
    r"\bMr\.": "Mister", r"\bMrs\.": "Missus", r"\bMs\.": "Miss",
    r"\bDr\.": "Doctor", r"\bSt\.": "Saint", r"\bProf\.": "Professor",
    r"\bSr\.": "Senior", r"\bJr\.": "Junior",
    r"\bInc\.": "Incorporated", r"\bLtd\.": "Limited",
    r"\bCo\.": "Company", r"\bAve\.": "Avenue", r"\bRd\.": "Road",
    r"\bNo\.": "number",
    r"\bvs\.": "versus", r"\betc\.": "etcetera",
    r"\be\.g\.": "for example", r"\bi\.e\.": "that is",
}
SYMBOLS = {"&": " and ", "%": " percent ", "@": " at ", "#": " number ",
           "$": " dollars "}


@dataclass
class Plan:
    quarantine: set[str] = field(default_factory=set)
    drop_rows: set[str] = field(default_factory=set)
    repairs: dict[str, tuple[str, str]] = field(default_factory=dict)  # id -> (old, new)
    normalize_file: list[str] = field(default_factory=list)
    reasons: dict[str, list[str]] = field(default_factory=dict)
    unresolved: list[Finding] = field(default_factory=list)

    def note(self, clip_id: str, code: str) -> None:
        self.reasons.setdefault(clip_id, []).append(code)

    @property
    def touched(self) -> set[str]:
        return self.quarantine | self.drop_rows | set(self.repairs)


def repair_text(text: str) -> str:
    out = text
    for pat, rep in ABBREV.items():
        out = re.sub(pat, rep, out)
    for sym, rep in SYMBOLS.items():
        out = out.replace(sym, rep)
    # single digits only; multi-digit numbers are ambiguous ("1984" is a year
    # or a quantity) and better handled by a human
    out = re.sub(r"(?<!\d)(\d)(?!\d)", lambda m: f" {NUMBER_WORDS[m.group(1)]} ", out)
    return re.sub(r"\s+", " ", out).strip()


def read_rows(project: Project) -> list[tuple[str, str]]:
    rows = []
    for line in project.metadata.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) >= 2:
            rows.append((parts[0], "|".join(parts[1:])))
    return rows


def write_rows(project: Project, rows: list[tuple[str, str]]) -> None:
    with project.metadata.open("w", newline="") as fh:
        csv.writer(fh, delimiter="|", quoting=csv.QUOTE_NONE,
                   escapechar="\\").writerows(rows)


def build_plan(project: Project, findings: list[Finding],
               only: set[str] | None = None,
               exclude: set[str] | None = None) -> Plan:
    plan = Plan()
    rows = dict(read_rows(project)) if project.metadata.exists() else {}

    for f in findings:
        code = f.code
        if only and code not in only:
            continue
        if exclude and code in exclude:
            continue

        action = f.action
        if action is None:
            continue

        if action == "repair":
            if code in ("crlf", "columns", "blank-row"):
                plan.normalize_file.append(code)
            elif code == "unspoken-text":
                for cid in f.ids:
                    old = rows.get(cid)
                    if old is None:
                        continue
                    new = repair_text(old)
                    if new != old:
                        plan.repairs[cid] = (old, new)
                        plan.note(cid, code)
                    else:
                        plan.unresolved.append(f)
        elif action == "drop-row":
            for cid in f.ids:
                plan.drop_rows.add(cid)
                plan.note(cid, code)
        elif action == "quarantine":
            for cid in f.ids:
                plan.quarantine.add(cid)
                plan.note(cid, code)

    # a quarantined clip's row goes too
    plan.drop_rows |= plan.quarantine
    return plan


def describe(plan: Plan, total_rows: int) -> list[str]:
    out = []
    if plan.normalize_file:
        out.append(f"· normalize metadata.csv ({', '.join(sorted(set(plan.normalize_file)))})")
    for cid, (old, new) in sorted(plan.repairs.items()):
        out.append(f"~ repair  {cid}\n    - {old}\n    + {new}")
    for cid in sorted(plan.quarantine):
        out.append(f"→ quarantine {cid}  [{', '.join(plan.reasons.get(cid, []))}]")
    for cid in sorted(plan.drop_rows - plan.quarantine):
        out.append(f"- drop row   {cid}  [{', '.join(plan.reasons.get(cid, []))}]")
    if plan.unresolved:
        out.append("")
        out.append("Needs a human (not auto-repairable):")
        for f in plan.unresolved:
            out.append(f"  {f}")
    if total_rows:
        pct = 100 * len(plan.drop_rows) / total_rows
        out.append("")
        out.append(f"summary: {len(plan.repairs)} repaired, "
                   f"{len(plan.quarantine)} quarantined, "
                   f"{len(plan.drop_rows)} rows removed "
                   f"({pct:.0f}% of {total_rows})")
    return out


def apply(project: Project, plan: Plan, force: bool = False) -> dict:
    rows = read_rows(project)
    total = len(rows)
    if total and len(plan.drop_rows) / total > MAX_FRACTION and not force:
        raise RuntimeError(
            f"refusing to remove {len(plan.drop_rows)}/{total} rows "
            f"({100*len(plan.drop_rows)/total:.0f}%). That usually means a "
            f"validation threshold is wrong, not the data. Use --force to "
            f"override, or narrow with --only.")

    qdir = project.dataset / "quarantine"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moved = []
    if plan.quarantine:
        qdir.mkdir(parents=True, exist_ok=True)
        for cid in sorted(plan.quarantine):
            src = project.wavs / f"{cid}.wav"
            if src.exists():
                shutil.move(str(src), str(qdir / src.name))
                moved.append(cid)

    kept: list[tuple[str, str]] = []
    for cid, text in rows:
        if cid in plan.drop_rows:
            continue
        if cid in plan.repairs:
            text = plan.repairs[cid][1]
        kept.append((cid, text))
    write_rows(project, kept)

    if plan.quarantine or plan.drop_rows or plan.repairs:
        manifest = qdir / "manifest.csv" if plan.quarantine else \
            project.dataset / "clean-log.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        new = not manifest.exists()
        with manifest.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["timestamp", "clip_id", "action", "reasons", "text"])
            row_text = dict(rows)
            for cid in sorted(plan.quarantine):
                w.writerow([stamp, cid, "quarantine",
                            ";".join(plan.reasons.get(cid, [])),
                            row_text.get(cid, "")])
            for cid in sorted(plan.drop_rows - plan.quarantine):
                w.writerow([stamp, cid, "drop-row",
                            ";".join(plan.reasons.get(cid, [])),
                            row_text.get(cid, "")])
            for cid, (old, new_t) in sorted(plan.repairs.items()):
                w.writerow([stamp, cid, "repair",
                            ";".join(plan.reasons.get(cid, [])),
                            f"{old} -> {new_t}"])

    return {"repaired": len(plan.repairs), "quarantined": len(moved),
            "rows_removed": total - len(kept), "rows_remaining": len(kept)}


def restore(project: Project, clip_ids: list[str] | None = None) -> int:
    """Move quarantined clips back. Metadata rows must be re-added by
    re-running transcribe, or by hand from quarantine/manifest.csv."""
    qdir = project.dataset / "quarantine"
    if not qdir.exists():
        return 0
    count = 0
    for wav in sorted(qdir.glob("*.wav")):
        if clip_ids and wav.stem not in clip_ids:
            continue
        shutil.move(str(wav), str(project.wavs / wav.name))
        count += 1
    return count
