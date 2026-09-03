import { useEffect, useRef, useState } from "react";
import { jobLogUrl } from "./api";
import type { Job, StreamMsg } from "./types";

// ------------------------------------------------------------- hash router
// Hash routes survive static hosting under /ui/app/ with no server
// rewrites, and they match the Bones routes exactly.

export function useHashRoute(): string {
  const [hash, setHash] = useState(() => location.hash || "#/projects");
  useEffect(() => {
    const on = () => setHash(location.hash || "#/projects");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

// ------------------------------------------------------------------ polling
// One interval, owned by one effect, cleared on unmount. The Bones bug
// this shape prevents: every render() leaking a timer (or a socket) that
// outlived its page (review finding 7). `paused` suspends the timer —
// e.g. while a websocket already carries live state — without tearing
// the effect down.

export function usePoll(
  fn: () => void | Promise<void>,
  ms: number,
  paused = false,
): void {
  const ref = useRef(fn);
  ref.current = fn;
  useEffect(() => {
    if (paused) return;
    const t = setInterval(() => {
      void ref.current();
    }, ms);
    return () => clearInterval(t);
  }, [ms, paused]);
}

// -------------------------------------------------------------- job stream
// Websocket to /api/jobs/{id}/stream plus the bounded log tail. The
// socket lives exactly as long as this effect: route away and the
// cleanup closes it (finding 7 again — a detached <pre> must never keep
// receiving lines).

const LOG_MAX_LINES = 5000;
const LOG_KEEP_LINES = 4000;

function trimTail(lines: string[]): string[] {
  // A chatty training run must not grow the log forever and hang the
  // tab; keep a tail, the full log stays at GET /api/jobs/{id}/log.
  return lines.length > LOG_MAX_LINES
    ? lines.slice(-LOG_KEEP_LINES)
    : lines;
}

export function useJobStream(jobId: string | null): {
  lines: string[];
  job: Job | null;
  logHref: string | null;
} {
  const [lines, setLines] = useState<string[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [logHref, setLogHref] = useState<string | null>(null);

  useEffect(() => {
    setLines([]);
    setJob(null);
    setLogHref(null);
    if (!jobId) return;

    setLogHref(jobLogUrl(jobId));
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${proto}://${location.host}/api/jobs/${jobId}/stream`,
    );
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as StreamMsg;
      if (msg.type === "log_reset") {
        // The tail replaces everything; a trailing newline would become
        // a phantom empty line in line-array form.
        const arr = msg.text === "" ? [] : msg.text.split("\n");
        if (arr[arr.length - 1] === "") arr.pop();
        setLines(trimTail(arr));
      } else if (msg.type === "log") {
        setLines((prev) => trimTail([...prev, msg.line]));
      } else if (msg.type === "state") {
        setJob(msg.job);
      }
    };
    return () => {
      ws.onclose = null;
      ws.close();
    };
  }, [jobId]);

  return { lines, job, logHref };
}

// Progress text, §1.3/§1.4 style: unit first (never an absolute epoch
// ceiling — review finding 12), dash when no counter has landed yet.
export function progressText(p: { unit?: string; current?: number; total: number } | null | undefined): string {
  if (!p || !p.total) return "";
  return `${p.unit ?? ""} ${p.current ?? "-"}/${p.total}`;
}
