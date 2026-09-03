// Clean view over a job's raw log tail. The runner tags machine-readable
// directives into the log (api/runner.py _emit): "##{nonce} TAG {json}".
// The pane parses those back out for a progress strip and a one-line
// RESULT. Train jobs emit only TARGET — Lightning's per-epoch output is
// plain stdout — so the current epoch also comes from "Epoch N:" lines.
// Lightning counts epochs from 0: while "Epoch 108:" runs, 108 of a
// 109-epoch ceiling are done.

import type { Job } from "./types";

const DIRECTIVE_RE = /^##[0-9a-f]+ (TARGET|PROGRESS|RESULT) (\{.*\})$/;
const EPOCH_RE = /\bEpoch (\d+):/;

export interface LogSummary {
  unit: string | null;
  current: number | null;
  total: number | null;
  result: Record<string, unknown> | null;
  error: string | null;
}

export function summarize(lines: string[], job: Job | null): LogSummary {
  // Seed from the job's server-side state first: it survives the client
  // tail cap, so a very long run whose TARGET directive has scrolled out
  // of the 5000-line tail still shows a bar.
  const s: LogSummary = {
    unit: job?.progress?.unit ?? null,
    current: job?.progress?.current ?? null,
    total: job?.progress?.total ?? null,
    result: job?.result ?? null,
    error: job?.error ?? null,
  };
  for (const line of lines) {
    const d = DIRECTIVE_RE.exec(line);
    if (d !== null) {
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(d[2]) as Record<string, unknown>;
      } catch {
        continue;
      }
      if (d[1] === "RESULT") {
        s.result = data;
        if (typeof data.error === "string") s.error = data.error;
      } else {
        // TARGET and PROGRESS share the current/total/unit shape.
        if (typeof data.total === "number") s.total = data.total;
        if (typeof data.unit === "string") s.unit = data.unit;
        if (typeof data.current === "number") s.current = data.current;
      }
    } else if (s.unit === "epoch") {
      const ep = EPOCH_RE.exec(line);
      if (ep) s.current = parseInt(ep[1], 10);
    }
  }
  return s;
}

// "epoch 108/109" — unit first, §1.3 style.
export function summaryText(s: LogSummary): string {
  if (s.total === null || s.total === 0) return "";
  return `${s.unit ?? ""} ${s.current ?? "-"}/${s.total}`.trim();
}

export function percentDone(s: LogSummary, state: string): number | null {
  if (state === "succeeded") return 100;
  if (s.total === null || s.current === null || s.total === 0) return null;
  return Math.max(0, Math.min(100, (s.current / s.total) * 100));
}

// One-line rendering of the RESULT payload; paths collapse to their
// basename, arrays to their items ("none" when empty), and objects with
// a name field to that name — enough to keep the strip to one row.
export function resultText(r: Record<string, unknown>): string {
  return Object.entries(r)
    .map(([k, v]) => `${k} ${shortValue(v)}`)
    .join(" · ");
}

function shortValue(v: unknown): string {
  if (typeof v === "string") {
    return v.includes("/") ? v.slice(v.lastIndexOf("/") + 1) : v;
  }
  if (Array.isArray(v)) {
    return v.length === 0 ? "none" : v.map(shortValue).join(",");
  }
  if (v !== null && typeof v === "object") {
    if ("name" in v) return shortValue((v as { name: unknown }).name);
    return JSON.stringify(v);
  }
  return String(v);
}
