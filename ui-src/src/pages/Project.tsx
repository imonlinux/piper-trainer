import { useEffect, useState } from "react";
import { ApiError, del, get, patch } from "../api";
import { usePoll } from "../hooks";
import {
  Panel,
  RecentJob,
  Stepper,
  stageHref,
  useStagesData,
} from "../shell";
import type { Job, ProjectDetail } from "../types";

// Overview (workorder-04 A5): the next card, the stage mini-map, the
// Output / Started-from settings split, the last five jobs, and the
// danger zone. Everything else moved out — sources live on their own
// page, transcription on its own page, the job table on Activity. The
// definition panel is the project.json editor: fix entries here when a
// manual correction beats re-creating the project.

// Values are JSON: plain text becomes a string, everything else must
// parse as JSON. null removes the key.
function defValueToText(v: unknown): string {
  return typeof v === "string" ? v : JSON.stringify(v) ?? "";
}

function DefinitionPanel(props: {
  name: string;
  definition: Record<string, unknown>;
  busy: boolean;
  onUpdated: (d: Record<string, unknown>) => void;
}) {
  const [editKey, setEditKey] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [addKey, setAddKey] = useState("");
  const [addVal, setAddVal] = useState("");
  const [err, setErr] = useState("");
  const [saved, setSaved] = useState("");

  async function apply(key: string, raw: string): Promise<void> {
    setErr("");
    setSaved("");
    let value: unknown;
    try {
      value = JSON.parse(raw);
    } catch {
      value = raw;
    }
    try {
      const r = await patch<{ definition: Record<string, unknown> }>(
        `/projects/${encodeURIComponent(props.name)}/definition`,
        { updates: { [key]: value } },
      );
      props.onUpdated(r.definition);
      setEditKey(null);
      setAddKey("");
      setAddVal("");
      setSaved(value === null ? `${key} removed` : `${key} saved`);
    } catch (ex) {
      setErr(String(ex));
    }
  }

  return (
    <Panel
      title="definition"
      help="project.json — the file every stage reads. Values are JSON: plain text becomes a string; numbers, true/false and objects must be valid JSON. A value of null removes the entry. The API refuses edits while a job is running."
    >
      <p className="consequence">
        edits land on the next job — a running job is never touched.
        changing espeak_voice re-phonemizes the whole dataset against the
        new voice on the next prepare or train.
      </p>
      <table>
        <tbody>
          {Object.entries(props.definition).map(([k, v]) => (
            <tr key={k}>
              <td className="mono">{k}</td>
              {editKey === k ? (
                <td>
                  <input
                    className="def-edit"
                    value={draft}
                    autoFocus
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") void apply(k, draft);
                      if (e.key === "Escape") setEditKey(null);
                    }}
                  />
                </td>
              ) : (
                <td className="mono defval">{defValueToText(v)}</td>
              )}
              <td>
                {k === "name" ? (
                  <span className="muted">fixed</span>
                ) : editKey === k ? (
                  <span className="row">
                    <button
                      onClick={() => void apply(k, draft)}
                      disabled={props.busy}
                    >
                      save
                    </button>
                    <button className="ghost" onClick={() => setEditKey(null)}>
                      cancel
                    </button>
                  </span>
                ) : (
                  <button
                    className="ghost"
                    disabled={props.busy}
                    onClick={() => {
                      setEditKey(k);
                      setDraft(defValueToText(v));
                      setSaved("");
                    }}
                  >
                    edit
                  </button>
                )}
              </td>
            </tr>
          ))}
          <tr>
            <td>
              <input
                placeholder="key"
                value={addKey}
                disabled={props.busy}
                onChange={(e) => setAddKey(e.target.value)}
              />
            </td>
            <td>
              <input
                placeholder="value"
                value={addVal}
                disabled={props.busy}
                onChange={(e) => setAddVal(e.target.value)}
              />
            </td>
            <td>
              <button
                disabled={props.busy || !addKey.trim()}
                onClick={() => void apply(addKey.trim(), addVal)}
              >
                add
              </button>
            </td>
          </tr>
        </tbody>
      </table>
      {props.busy && (
        <p className="muted">a job is running — edits are refused until it finishes</p>
      )}
      {err && <p className="error">{err}</p>}
      {saved && <p className="ok">{saved}</p>}
    </Panel>
  );
}

export function ProjectPage({ name }: { name: string }) {
  const [p, setP] = useState<ProjectDetail | null>(null);
  const [loadError, setLoadError] = useState<Error | null>(null);
  const [actionError, setActionError] = useState("");
  // stages come from the shell's one poll — no second fetch here
  const stages = useStagesData();

  useEffect(() => {
    setP(null);
    setLoadError(null);
    get<ProjectDetail>(`/projects/${encodeURIComponent(name)}`)
      .then(setP)
      .catch((e: Error) => setLoadError(e));
  }, [name]);

  // Only the recent-activity list stays live here; the shell owns the
  // stage poll. A 404 surfaces (and pauses this poll) instead of spin.
  usePoll(async () => {
    try {
      const jobs = await get<Job[]>(`/projects/${encodeURIComponent(name)}/jobs`);
      setP((prev) => (prev ? { ...prev, jobs } : prev));
    } catch (ex) {
      if (ex instanceof ApiError && ex.status === 404) setLoadError(ex);
    }
  }, 2000, loadError !== null);

  // Definition edits must not leave the settings panel above showing the
  // old espeak_voice/tier/catalog_path until the next full reload.
  function applyDefinition(d: Record<string, unknown>): void {
    setP((prev) =>
      prev
        ? {
            ...prev,
            definition: d,
            config: {
              espeak_voice: (d.espeak_voice as string | null) ?? null,
              tier: (d.tier as string | null) ?? null,
              catalog_path: (d.catalog_path as string | null) ?? null,
              target_epochs: (d.target_epochs as number | null) ?? null,
              transcripts_provided: (d.transcripts_provided as boolean | null) ?? null,
            },
          }
        : prev,
    );
  }

  async function doDelete(): Promise<void> {
    if (!confirm(`Move project "${name}" to .trash? Nothing is destroyed.`))
      return;
    setActionError("");
    try {
      await del(`/projects/${name}`);
      location.hash = "#/projects";
    } catch (ex) {
      setActionError(String(ex));
    }
  }

  const cfg = p?.config ?? {};
  const recent = (p?.jobs ?? []).slice(0, 5);

  return (
    <>
      {loadError instanceof ApiError && loadError.status === 404 ? (
        <>
          <h1>{name}</h1>
          <p className="error">no such project: {name}</p>
        </>
      ) : loadError ? (
        <p className="error">{String(loadError)}</p>
      ) : (
        <>
          <h1>{p?.name ?? name}</h1>
          {p && (
            <p className="muted">
              {p.clips} clips · {p.minutes ?? "?"} min ·{" "}
              {p.dataset.rows} dataset rows
              {p.dataset.malformed_lines > 0
                ? `, ${p.dataset.malformed_lines} malformed`
                : ""}
            </p>
          )}

          <Panel
            title="next"
            help="The one thing to do now, derived from where the project actually is. The stepper below is the whole pipeline; every stage is always clickable."
          >
            {stages ? (
              <div className="next-card">
                <p className="next-why">{stages.next.why}</p>
                <a className="button-primary" href={stageHref(stages.next.stage, name)}>
                  go to {stages.next.stage}
                </a>
              </div>
            ) : (
              <p className="muted">loading…</p>
            )}
          </Panel>
          <Stepper name={name} />

          <Panel
            title="settings"
            help="Output describes what you get; Started-from describes where the voice's accent comes from. The base voice never changes after export."
          >
            <div className="settings-cols">
              <div>
                <h3>output</h3>
                <p>
                  voice name: {stages?.project.voice ?? "—"}
                  <br />
                  exported voices: {p?.voices.length ?? 0}
                  <br />
                  location: out/
                </p>
              </div>
              <div>
                <h3>started from</h3>
                <p>
                  {cfg.catalog_path
                    ? `base voice: ${cfg.catalog_path}`
                    : "no base voice selected — training runs from scratch"}
                  {cfg.tier ? (
                    <>
                      <br />
                      tier: {cfg.tier}
                    </>
                  ) : null}
                </p>
              </div>
            </div>
          </Panel>

          {p?.definition && (
            <DefinitionPanel
              name={name}
              definition={p.definition}
              busy={!!stages?.running.job}
              onUpdated={applyDefinition}
            />
          )}

          <Panel
            title="recent activity"
            help="The five most recent jobs. The full history — cancelled runs, failed fetches, audition logs — lives on the activity page."
          >
            {recent.length === 0 ? (
              <p className="muted">no jobs yet</p>
            ) : (
              <ul className="recent">
                {recent.map((j) => (
                  <RecentJob key={j.id} job={j} />
                ))}
              </ul>
            )}
            <p>
              <a href={`#/project/${name}/activity`}>all activity</a>
            </p>
          </Panel>
        </>
      )}

      <Panel title="danger">
        <button onClick={() => void doDelete()}>delete (moves to .trash)</button>
        {actionError && <span className="error"> {actionError}</span>}
      </Panel>
    </>
  );
}
