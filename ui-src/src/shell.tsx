import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { get, postEmpty } from "./api";
import { durationText, progressText, useStages } from "./hooks";
import type { Job, JobRef, StageName, StagesData } from "./types";

// The workorder-04 shell (A3): IdentityBar + Stepper + JobBar around
// every project route. One stages poll per project, shared with pages
// through context. Amber (--state-run) is reserved for work in flight;
// interaction stays monochrome.

export const STAGES: StageName[] = [
  "sources",
  "prepare",
  "transcribe",
  "audit",
  "train",
  "voices",
];

const STAGE_ROUTE: Record<StageName, (name: string) => string> = {
  sources: (n) => `#/sources/${n}`,
  prepare: (n) => `#/prepare/${n}`,
  transcribe: (n) => `#/transcribe/${n}`,
  audit: (n) => `#/audit/${n}`,
  train: (n) => `#/train/${n}`,
  voices: (n) => `#/voices/${n}`,
};

export const stageHref = (stage: StageName, name: string): string =>
  STAGE_ROUTE[stage](name);

// Job kinds in user vocabulary. Activity is the only place raw job
// vocabulary may surface; everywhere else speaks this.
const KIND_TEXT: Record<string, string> = {
  ingest: "adding sources",
  prepare: "preparing clips",
  transcribe: "writing transcripts",
  validate: "checking the dataset",
  clean: "cleaning the dataset",
  restore: "restoring clips",
  train: "training",
  export: "exporting voice",
  preview: "measuring",
  "fetch-checkpoint": "fetching base voice",
};

export const kindText = (kind: string): string => KIND_TEXT[kind] ?? kind;

// -------------------------------------------------------------- theme
// data-theme on <html>; dark is the default (A1). The toggle lives in
// the IdentityBar on shell routes and in the plain header elsewhere —
// reachable on every route (A4).

export function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState(() =>
    localStorage.getItem("ptt-theme") === "light" ? "light" : "dark",
  );
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ptt-theme", theme);
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

export function ThemeToggle() {
  const [theme, toggle] = useTheme();
  return (
    <button type="button" className="ghost" onClick={toggle}>
      {theme === "dark" ? "light mode" : "dark mode"}
    </button>
  );
}

// ------------------------------------------------------- stages context

const StagesContext = createContext<StagesData | null>(null);

/** Read the shell's stages poll from any page inside ProjectShell. */
export const useStagesData = (): StagesData | null => useContext(StagesContext);

// ------------------------------------------------------------- ProjectShell

export function ProjectShell(props: {
  name: string;
  stage: StageName | null;
  children: ReactNode;
}) {
  const { data, loading, stale } = useStages(props.name);
  const run =
    data && data.running.job
      ? { job: data.running.job, more: data.running.more }
      : null;
  return (
    <StagesContext.Provider value={data}>
      <IdentityBar
        name={props.name}
        data={data}
        loading={loading}
        stale={stale}
      />
      <Stepper name={props.name} />
      {run && <JobBar name={props.name} run={run} />}
      <main>
        <GateBanner name={props.name} stage={props.stage} data={data} />
        {props.children}
      </main>
    </StagesContext.Provider>
  );
}

// ---------------------------------------------------------- IdentityBar
// 46 px: back link · name · output-voice chip · status line · theme ·
// overflow. The status line is the idle JobBar: the one-line truth.

function IdentityBar(props: {
  name: string;
  data: StagesData | null;
  loading: boolean;
  stale: boolean;
}) {
  const { data } = props;
  const done = data
    ? STAGES.filter((s) => data.stages[s].status === "done").length
    : 0;
  let line: string;
  if (props.stale) line = "stale — API unreachable, retrying";
  else if (props.loading || !data) line = "loading…";
  else if (data.running.job) {
    line =
      `running: ${kindText(data.running.job.kind)}` +
      (progressText(data.running.job.progress)
        ? ` · ${progressText(data.running.job.progress)}`
        : "");
  } else line = `${done}/6 stages done · next: ${data.next.stage} — ${data.next.why}`;
  return (
    <header className="identity">
      <a href="#/projects">← projects</a>
      <strong>{props.name}</strong>
      {data && <span className="tag">{data.project.voice}</span>}
      <span className="status-line">{line}</span>
      <span className="spacer" />
      <ThemeToggle />
      <details className="overflow">
        <summary aria-label="more">⋯</summary>
        <div className="overflow-menu">
          <a href="#/doctor">doctor</a>
          <Version />
        </div>
      </details>
    </header>
  );
}

function Version() {
  const [text, setText] = useState("");
  useEffect(() => {
    get<{ version: string }>("/health")
      .then((h) => setText(`v${h.version}`))
      .catch(() => setText(""));
  }, []);
  return <span className="muted">{text}</span>;
}

// --------------------------------------------------------------- Stepper
// 68 px, six pills in pipeline order: a status dot and a 2 px top
// stripe in the status color — the stripe reads as a progress bar
// across the six. Every pill is always clickable, locked included; a
// locked destination explains itself with the gate banner.

export function Stepper({ name }: { name: string }) {
  const data = useContext(StagesContext);
  return (
    <nav className="stepper" aria-label="pipeline stages">
      {STAGES.map((s) => {
        const status = data?.stages[s]?.status ?? "locked";
        return (
          <a
            key={s}
            className={`step step-${status}`}
            href={STAGE_ROUTE[s](name)}
          >
            <span className="step-dot" aria-hidden="true" />
            {s}
          </a>
        );
      })}
    </nav>
  );
}

// ----------------------------------------------------------------- JobBar
// 42 px, exists only while ≥ 1 job runs; collapses into the IdentityBar
// status line when the last job finishes. ETA appears only after ≥ 2
// progress samples spanning ≥ 2 % of total; before that it reads
// "estimating…". Never an absolute wall-clock destination.

function JobBar(props: { name: string; run: { job: JobRef; more: number } }) {
  const { job, more } = props.run;
  const [cancelErr, setCancelErr] = useState("");
  const samples = useRef<{ t: number; cur: number; total: number }[]>([]);
  const prevId = useRef<string | null>(null);
  if (prevId.current !== job.id) {
    prevId.current = job.id;
    samples.current = [];
  }
  const p = job.progress;
  if (p && p.total) {
    const arr = samples.current;
    const last = arr[arr.length - 1];
    if (!last || last.cur !== (p.current ?? 0)) {
      arr.push({ t: Date.now(), cur: p.current ?? 0, total: p.total });
      if (arr.length > 30) arr.splice(0, arr.length - 30);
    }
  }
  const eta = etaText(samples.current);

  async function cancel(): Promise<void> {
    setCancelErr("");
    try {
      await postEmpty(`/jobs/${job.id}/cancel`);
      // the 3 s stages poll picks the state change up
    } catch (ex) {
      setCancelErr(String(ex));
    }
  }

  return (
    <div className="jobbar" role="status">
      <span className="jobbar-pulse" aria-hidden="true" />
      <span>{kindText(job.kind)}</span>
      {p && p.total ? (
        <>
          <span className="mono">{progressText(p)}</span>
          <span className="bar" aria-hidden="true">
            <span
              className="bar-fill"
              style={{ width: `${Math.min(100, ((p.current ?? 0) / p.total) * 100)}%` }}
            />
          </span>
          <span className="muted mono">
            {eta ?? "estimating…"}
          </span>
        </>
      ) : (
        <span className="muted">working…</span>
      )}
      {more > 0 && (
        <a href={`#/project/${props.name}/activity`}>+{more} more</a>
      )}
      <span className="spacer" />
      {cancelErr && <span className="error">{cancelErr}</span>}
      <button type="button" onClick={() => void cancel()}>
        cancel
      </button>
    </div>
  );
}

// ETA rule (A3): ≥ 2 samples spanning ≥ 2 % of total; rate from the
// outermost samples so backoff noise smooths out.
function etaText(
  samples: { t: number; cur: number; total: number }[],
): string | null {
  if (samples.length < 2) return null;
  const a = samples[0];
  const b = samples[samples.length - 1];
  const span = b.t - a.t;
  if (span < 1000) return null;
  if ((b.cur - a.cur) / b.total < 0.02) return null;
  const rate = (b.cur - a.cur) / (span / 1000);
  if (!(rate > 0)) return null;
  const rem = (b.total - b.cur) / rate;
  if (!isFinite(rem) || rem < 0) return null;
  if (rem < 90) return `~${Math.max(1, Math.round(rem))}s`;
  if (rem < 5400) return `~${Math.round(rem / 60)}m`;
  return `~${(rem / 3600).toFixed(1)}h`;
}

// ------------------------------------------------------------------ Gate
// Dismissible banner for a stage whose requirements are unmet (A3).
// One line, expandable to the full requirement list, dismissed for the
// session. The rest of the page renders normally underneath: locks
// teach, they don't gate-click.

function GateBanner(props: {
  name: string;
  stage: StageName | null;
  data: StagesData | null;
}) {
  const [open, setOpen] = useState(false);
  const key = `ptt-gate-dismissed:${props.name}:${props.stage ?? ""}`;
  const [dismissed, setDismissed] = useState(() => {
    try {
      return sessionStorage.getItem(key) !== null;
    } catch {
      return false;
    }
  });
  const { stage, data } = props;
  if (!stage || !data) return null;
  const info = data.stages[stage];
  if (!info || !info.blocked_by || dismissed) return null;
  return (
    <div className="gate" role="note">
      <div className="gate-line">
        <span>
          <strong>{stage}</strong> is not ready: {info.blocked_by}
        </span>
        <span className="row">
          <button type="button" className="ghost" onClick={() => setOpen((o) => !o)}>
            {open ? "hide" : "why?"}
          </button>
          <button
            type="button"
            className="ghost"
            onClick={() => {
              try {
                sessionStorage.setItem(key, "1");
              } catch {
                // private mode: dismissal just won't persist
              }
              setDismissed(true);
            }}
          >
            dismiss
          </button>
        </span>
      </div>
      {open && (
        <ul className="gate-reqs">
          {info.requirements.map((r) => (
            <li key={r.text} className={r.met ? "ok" : undefined}>
              {r.met ? "done —" : "missing —"} {r.text}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// --------------------------------------------------------- primitives
// Panel: titled bordered section, the only card primitive. Consequence:
// one sentence above a destructive or expensive action, naming what
// happens to data. Help: a per-panel `?` with a two-sentence explainer.

export function Panel(props: { title: string; help?: string; children: ReactNode }) {
  return (
    <section className="panel">
      <h2>
        {props.title}
        {props.help && <Help text={props.help} />}
      </h2>
      {props.children}
    </section>
  );
}

function Help({ text }: { text: string }) {
  const [open, setOpen] = useState(false);
  return (
    <span className="help">
      <button
        type="button"
        aria-label="what is this?"
        onClick={() => setOpen((o) => !o)}
      >
        ?
      </button>
      {open && <span className="help-text">{text}</span>}
    </span>
  );
}

export function Consequence(props: { children: ReactNode }) {
  return <p className="consequence">{props.children}</p>;
}

// Recent-activity row for the overview: kind in user vocabulary, state,
// when, duration. Raw ids stay on the Activity page.
export function RecentJob(props: { job: Job }) {
  const j = props.job;
  return (
    <li>
      <span>{kindText(j.kind)}</span>
      <span className={`state-${j.state}`}>{j.state}</span>
      <span className="muted mono">
        {durationText(j.started_at, j.finished_at)}
      </span>
    </li>
  );
}
