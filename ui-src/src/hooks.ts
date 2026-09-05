import { useEffect, useRef, useState } from "react";
import { get, jobLogUrl } from "./api";
import type { Job, StagesData, StreamMsg } from "./types";

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
  connected: boolean;
} {
  const [lines, setLines] = useState<string[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [logHref, setLogHref] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    setLines([]);
    setJob(null);
    setLogHref(null);
    setConnected(false);
    if (!jobId) return;

    setLogHref(jobLogUrl(jobId));
    // A dropped socket must not strand the page: pages pause their poll
    // while the stream is live, so a silently dead socket would freeze
    // the job table forever. Reconnect with backoff; every connect gets
    // a fresh state + log tail from the server, so nothing is lost.
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let attempt = 0;
    let closed = false;

    function connect(): void {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(
        `${proto}://${location.host}/api/jobs/${jobId}/stream`,
      );
      ws.onopen = () => {
        attempt = 0;
        setConnected(true);
      };
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
      ws.onclose = () => {
        setConnected(false);
        if (closed) return;
        const delay = Math.min(1000 * 2 ** attempt, 15000);
        attempt += 1;
        timer = window.setTimeout(connect, delay);
      };
      ws.onerror = () => ws?.close(); // onclose follows and schedules retry
    }
    connect();
    return () => {
      closed = true;
      if (timer !== undefined) window.clearTimeout(timer);
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
    };
  }, [jobId]);

  return { lines, job, logHref, connected };
}

// Progress text, §1.3/§1.4 style: unit first (never an absolute epoch
// ceiling — review finding 12), dash when no counter has landed yet.
export function progressText(p: { unit?: string; current?: number; total: number } | null | undefined): string {
  if (!p || !p.total) return "";
  return `${p.unit ?? ""} ${p.current ?? "-"}/${p.total}`;
}

// ---------------------------------------------------------------- stages
// The shell's one poll (workorder-04 A3): GET /stages every 3 s, shared
// with pages through the shell's context so pages stop fetching
// overlapping data. A hidden tab suspends the fetch (not the timer) —
// it costs nothing while hidden and the first tick after returning
// catches up. An error keeps the last good data on screen and flips
// `stale`, so the status line can say "stale" instead of pretending.
export function useStages(name: string): {
  data: StagesData | null;
  loading: boolean;
  stale: boolean;
} {
  const [data, setData] = useState<StagesData | null>(null);
  const [loading, setLoading] = useState(true);
  const [stale, setStale] = useState(false);

  useEffect(() => {
    setData(null);
    setLoading(true);
    setStale(false);
  }, [name]);

  const tick = async (): Promise<void> => {
    if (document.hidden) return;
    try {
      const d = await get<StagesData>(
        `/projects/${encodeURIComponent(name)}/stages`,
      );
      setData(d);
      setLoading(false);
      setStale(false);
    } catch {
      setStale(true);
    }
  };

  // first fetch lands now, not on the first interval tick
  useEffect(() => {
    void tick();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);

  usePoll(tick, 3000);

  return { data, loading, stale };
}

// Duration between two ISO-8601 Z timestamps, human-shaped: `42s`,
// `12m 05s`, `3h 12m`. Used by the activity table and the overview.
export function durationText(
  started: string | null,
  finished: string | null,
): string {
  if (!started || !finished) return "";
  const ms = Date.parse(finished) - Date.parse(started);
  if (!(ms >= 0)) return "";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, "0")}m`;
}
