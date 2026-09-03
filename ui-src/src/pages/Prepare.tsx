import { useEffect, useRef, useState } from "react";
import { ApiError, del, fileUrl, get, post, postEmpty } from "../api";
import type { Job, Peaks, PreviewRow, Region, SourceInfo } from "../types";
import { Wave } from "../components/Wave";

type Ref<T> = { current: T };

// The segment tuner (§6.2) — the screen that justifies the UI. VAD dials,
// waveform of one source with regions overlaid, each preview run joining
// a sweep the user can compare and promote to the full prepare.

const SLIDERS = [
  { key: "energy_threshold", label: "energy threshold", min: 20, max: 90, step: 1, dflt: 55 },
  { key: "min_dur", label: "min duration (s)", min: 0.5, max: 5, step: 0.1, dflt: 1.5 },
  { key: "max_dur", label: "max duration (s)", min: 2, max: 20, step: 0.5, dflt: 10 },
  { key: "max_silence", label: "max silence (s)", min: 0.1, max: 1.5, step: 0.05, dflt: 0.4 },
  { key: "pad", label: "pad (s)", min: 0, max: 0.5, step: 0.01, dflt: 0.15 },
] as const;

type TunerParams = {
  source: string;
  channel: string;
  denoise: boolean;
} & Record<string, string | number | boolean>;

export function PreparePage({ name }: { name: string }) {
  const [sources, setSources] = useState<SourceInfo[] | null>(null);
  const [source, setSource] = useState("");
  const [channel, setChannel] = useState("downmix");
  const [denoiseFirst, setDenoiseFirst] = useState(true);
  const [dials, setDials] = useState<Record<string, number>>(
    Object.fromEntries(SLIDERS.map((s) => [s.key, s.dflt])),
  );
  const [peaks, setPeaks] = useState<Peaks | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [segSweep, setSegSweep] = useState<PreviewRow[]>([]);
  const [dnSweep, setDnSweep] = useState<PreviewRow[]>([]);
  const [failedPreviews, setFailedPreviews] = useState<{ id: string; error: string }[]>([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [gone, setGone] = useState(false);
  const sweepFp = useRef("");

  // ------------------------------------------------------------- sources
  useEffect(() => {
    let alive = true;
    get<SourceInfo[]>(`/projects/${name}/sources`)
      .then((s) => {
        if (!alive) return;
        setSources(s);
        if (s.length > 0) setSource(s[0].name);
      })
      .catch((e: Error) => {
        // A deleted project must say so, not masquerade as "no sources"
        // (same shape as the project page's 404 guard).
        if (!alive) return;
        if (e instanceof ApiError && e.status === 404) setGone(true);
        else setSources([]);
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);

  // --------------------------------------------------------------- peaks
  useEffect(() => {
    if (!source) return;
    let alive = true;
    setPeaks(null);
    get<Peaks>(
      `/projects/${name}/sources/${encodeURIComponent(source)}/peaks?channel=${channel}&buckets=2000`,
    )
      .then((pk) => {
        if (alive) setPeaks(pk);
      })
      .catch(() => {
        if (alive) setPeaks(null);
      });
    return () => {
      alive = false;
    };
  }, [name, source, channel]);

  // ------------------------------------------- sweep + failed job polling
  // Previews are immutable once listed, so refresh the arrays only when
  // the set of ids changes: re-rendering identical rows would kill the
  // denoise A/B <audio> nodes mid-playback (the Bones fingerprint).
  useSweepPoll(
    name,
    sweepFp,
    setSegSweep,
    setDnSweep,
    setFailedPreviews,
    gone,
    () => setGone(true),
  );

  // Newest segment preview is selected by default, once the sweep lands.
  useEffect(() => {
    if (selectedId === null && segSweep.length > 0) {
      setSelectedId(segSweep[0].id);
    }
  }, [segSweep, selectedId]);

  const selected =
    [...segSweep, ...dnSweep].find((r) => r.id === selectedId) ?? null;
  const regions: Region[] =
    selected?.stage === "segment" ? (selected.result.clips ?? []) : [];

  function tunerParams(): TunerParams {
    return { source, channel, denoise: denoiseFirst, ...dials };
  }

  async function preview(stage: "segment" | "denoise"): Promise<void> {
    setError("");
    setMessage("");
    const params =
      stage === "segment"
        ? tunerParams()
        : { source, channel, seconds: 25 };
    try {
      await post(`/projects/${name}/preview`, { stage, params });
      setMessage(`${stage} preview queued…`);
    } catch (ex) {
      setError(String(ex));
    }
  }

  async function promote(row: PreviewRow): Promise<void> {
    setError("");
    if (!confirm(`Run the full prepare with these parameters?\n${JSON.stringify(row.params)}`))
      return;
    try {
      await postEmpty(`/projects/${name}/previews/${row.id}/promote`);
      setMessage("full prepare queued from promoted parameters");
    } catch (ex) {
      setError(String(ex));
    }
  }

  async function prune(): Promise<void> {
    if (!confirm("Delete all previews? They are freely discardable.")) return;
    await del(`/projects/${name}/previews`).catch(() => undefined);
    setSelectedId(null);
    sweepFp.current = ""; // force the next poll to redraw
  }

  if (gone) {
    return (
      <>
        <h1>{name}</h1>
        <p className="error">no such project: {name}</p>
        <p>
          <a href="#/projects">back to projects</a>
        </p>
      </>
    );
  }
  if (sources === null) return <p className="muted">loading…</p>;
  const hasSources = sources.length > 0;

  return (
    <>
      <h1>Prepare tuner — {name}</h1>
      <p className="muted">
        adjust the VAD dials, preview against one source, promote the winner{" "}
        <a href={`#/project/${name}`}>back to project</a>
      </p>
      {error && <p className="error">{error}</p>}
      {message && <p className="muted">{message}</p>}

      <h2>Source</h2>
      <div className="row">
        <select
          value={source}
          onChange={(e) => setSource(e.target.value)}
        >
          {sources.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name}
            </option>
          ))}
        </select>
        {["downmix", "left", "right"].map((c) => (
          <label className="inline" key={c}>
            <input
              type="radio"
              name="chan"
              value={c}
              checked={channel === c}
              onChange={() => setChannel(c)}
            />
            {c}
          </label>
        ))}
      </div>
      {hasSources ? (
        <Wave data={peaks} regions={regions} />
      ) : (
        <p className="muted">
          no sources — upload audio on the project page first
        </p>
      )}

      <h2>VAD parameters</h2>
      <div className="row" style={{ flexWrap: "wrap" }}>
        {SLIDERS.map((s) => (
          <label className="inline" key={s.key}>
            {s.label}
            <input
              type="range"
              min={s.min}
              max={s.max}
              step={s.step}
              value={dials[s.key]}
              style={{ width: "14em" }}
              onChange={(e) =>
                setDials((d) => ({ ...d, [s.key]: parseFloat(e.target.value) }))
              }
            />
            <span className="muted">{dials[s.key]}</span>
          </label>
        ))}
        <label
          className="inline"
          title="the full pipeline denoises before segmenting; previews should judge the same audio"
        >
          <input
            type="checkbox"
            checked={denoiseFirst}
            onChange={(e) => setDenoiseFirst(e.target.checked)}
          />
          denoise first
        </label>
      </div>
      <div className="row">
        <button disabled={!hasSources} onClick={() => void preview("segment")}>
          preview segment
        </button>
        <button disabled={!hasSources} onClick={() => void preview("denoise")}>
          preview denoise A/B
        </button>
      </div>

      {failedPreviews.map((f) => (
        <p className="error" key={f.id}>
          preview job {f.id} failed: {f.error || "unknown error"}
        </p>
      ))}

      <h2>Clips</h2>
      {selected === null ? (
        <p className="muted">
          click a preview id below to inspect its clips here
        </p>
      ) : (
        <SelectedClips row={selected} project={name} />
      )}

      <h2>Segment sweep</h2>
      {segSweep.length === 0 ? (
        <p className="muted">no segment previews yet</p>
      ) : (
        <SweepTable rows={segSweep} kind="segment" onSelect={setSelectedId} onPromote={(r) => void promote(r)} />
      )}

      <h2>Denoise A/B</h2>
      {dnSweep.length === 0 ? (
        <p className="muted">no denoise previews yet</p>
      ) : (
        <>
          <SweepTable rows={dnSweep} kind="denoise" onSelect={setSelectedId} onPromote={null} />
          <div className="grid">
            {dnSweep.slice(0, 3).map((row) => (
              <div className="cell" key={row.id}>
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    setSelectedId(row.id);
                  }}
                >
                  {row.id}
                </a>
                {(row.result.audio ?? []).map((f) => (
                  <audio
                    key={f}
                    controls
                    src={fileUrl(name, row.dir, f)}
                    style={{ width: "100%" }}
                  />
                ))}
              </div>
            ))}
          </div>
        </>
      )}

      <p>
        <button onClick={() => void prune()}>prune all previews</button>
      </p>
    </>
  );
}

// One poller for the whole page: sweep halves and failed preview jobs
// (finding 11 — a preview that dies writes no preview.json, so the sweep
// alone would sit on "queued…" forever). A 404 from either endpoint means
// the project was deleted under the open tab; onGone flips the page and
// `gone` tears this interval down, so the 404s stop instead of repeating
// every 2 seconds behind the error message.
function useSweepPoll(
  name: string,
  sweepFp: Ref<string>,
  setSeg: (rows: PreviewRow[]) => void,
  setDn: (rows: PreviewRow[]) => void,
  setFailed: (rows: { id: string; error: string }[]) => void,
  gone: boolean,
  onGone: () => void,
): void {
  const failFp = useRef("");
  useEffect(() => {
    if (gone) return; // poll stays dead once the page has flipped
    async function tick(): Promise<void> {
      const [previews, jobs] = await Promise.all([
        get<PreviewRow[]>(`/projects/${name}/previews`),
        get<Job[]>(`/projects/${name}/jobs`),
      ]).catch((e: Error) => {
        if (e instanceof ApiError && e.status === 404) onGone();
        return [[], []] as [PreviewRow[], Job[]];
      });
      const seg = previews.filter((r) => r.stage === "segment");
      const dn = previews.filter((r) => r.stage === "denoise");
      const fp = seg.map((r) => r.id).join() + "|" + dn.map((r) => r.id).join();
      if (fp !== sweepFp.current) {
        sweepFp.current = fp;
        setSeg(seg);
        setDn(dn);
      }
      const fails = jobs
        .filter((j) => j.kind === "preview" && j.state === "failed")
        .map((j) => ({ id: j.id, error: j.error ?? "" }));
      const ffp = fails.map((f) => f.id).join();
      if (ffp !== failFp.current) {
        failFp.current = ffp;
        setFailed(fails);
      }
    }
    const t = setInterval(() => void tick(), 2000);
    void tick();
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name, gone]);
}

function SelectedClips({ row, project }: { row: PreviewRow; project: string }) {
  const audio = row.result.audio ?? [];
  const head =
    row.stage === "denoise"
      ? `denoise A/B from ${row.id} (original vs denoised, ${row.result.seconds ?? "?"}s)`
      : `clips from ${row.id} (${row.result.clip_count ?? "?"} total, first ${audio.length} playable)`;
  // A segment preview that found nothing: say which dials to move.
  const empty =
    row.stage === "segment" && row.result.clip_count === 0 ? (
      <p className="notice">
        0 clips: the splitter rejected every region. Lower the energy
        threshold, lower the min duration for short utterances, or pick
        left/right when only one channel has audio (downmix averages in the
        silent side). Then preview segment again.
      </p>
    ) : null;
  return (
    <>
      <h3>{head}</h3>
      {empty ?? (
        <div className="grid">
          {audio.map((f) => (
            <div className="cell" key={f}>
              {f}
              <audio
                controls
                src={fileUrl(project, row.dir, f)}
                style={{ width: "100%" }}
              />
            </div>
          ))}
        </div>
      )}
    </>
  );
}

function SweepTable(props: {
  rows: PreviewRow[];
  kind: "segment" | "denoise";
  onSelect: (id: string) => void;
  onPromote: ((row: PreviewRow) => void) | null;
}) {
  const isSeg = props.kind === "segment";
  return (
    <table>
      <thead>
        <tr>
          <th>id</th>
          <th>params</th>
          <th>{isSeg ? "clips" : "seconds"}</th>
          <th>histogram</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {props.rows.map((row) => {
          const p = (row.params ?? {}) as Record<string, unknown>;
          const sum =
            `${String(p.source ?? "?")} · ${String(p.channel ?? "downmix")}` +
            (isSeg
              ? ` · energy ${String(p.energy_threshold)} · dn ${String(p.denoise !== false)}`
              : ` · ${String(p.seconds)}s`);
          const hist = (row.result.histogram ?? [])
            .map((h) => `${h.from}-${h.to}s: ${h.count}`)
            .join(", ");
          return (
            <tr key={row.id}>
              <td>
                <a
                  href="#"
                  onClick={(e) => {
                    e.preventDefault();
                    props.onSelect(row.id);
                  }}
                >
                  {row.id}
                </a>
              </td>
              <td>{sum}</td>
              <td className="num">
                {String(
                  (isSeg ? row.result.clip_count : row.result.seconds) ?? "",
                )}
              </td>
              <td>{isSeg ? hist : ""}</td>
              <td>
                {props.onPromote && isSeg ? (
                  <button onClick={() => props.onPromote?.(row)}>promote</button>
                ) : null}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
