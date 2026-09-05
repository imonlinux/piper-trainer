import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, get, patch, post, postWav } from "../api";
import type { Checkpoint, ProjectDetail, VoiceInfo } from "../types";

// The Voices screen (§6.5): export a training checkpoint under a
// governed name, tune inference with the three sliders that change how
// a voice actually sounds, hear it instantly, download the pair.
// Audition A/B/C over checkpoints builds on `say` and ships next.

const fmtMB = (b: number) => `${(b / 1e6).toFixed(0)} MB`;

export function VoicesPage({ name }: { name: string }) {
  const [voices, setVoices] = useState<VoiceInfo[] | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [gone, setGone] = useState(false);
  const [loadErr, setLoadErr] = useState("");

  // export form
  const [ckpts, setCkpts] = useState<Checkpoint[]>([]);
  const [ckptSel, setCkptSel] = useState("");
  const [voiceName, setVoiceName] = useState("");
  const [exportMsg, setExportMsg] = useState("");
  const [exportErr, setExportErr] = useState("");
  // The box is prefilled with the same default the server computes
  // ({name}-{tier}); typing overrides it, clearing it hands the choice
  // back to the server. Editable default, not a lock.
  const nameTouched = useRef(false);
  const [tier, setTier] = useState<string | null>(null);

  const reload = useCallback(async () => {
    try {
      const list = await get<VoiceInfo[]>(`/projects/${name}/voices`);
      setVoices(list);
      setLoadErr("");
      setSel((cur) =>
        cur && list.some((v) => v.stem === cur)
          ? cur
          : (list[0]?.stem ?? null),
      );
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setGone(true);
        return;
      }
      setLoadErr(e instanceof Error ? e.message : String(e));
    }
  }, [name]);

  useEffect(() => {
    void reload();
  }, [reload]);

  useEffect(() => {
    get<Checkpoint[]>(`/projects/${name}/checkpoints`)
      .then(setCkpts)
      .catch(() => setCkpts([]));
  }, [name]);

  useEffect(() => {
    get<ProjectDetail>(`/projects/${name}`)
      .then((d) => {
        const t = d.config.tier ?? "medium";
        setTier(t);
        if (!nameTouched.current) setVoiceName(`${name}-${t}`);
      })
      .catch(() => setTier("medium"));
  }, [name]);

  async function doExport() {
    setExportErr("");
    setExportMsg("");
    const params: Record<string, unknown> = {};
    if (ckptSel) params.checkpoint = ckptSel; // project-relative run path
    if (voiceName.trim()) params.voice_name = voiceName.trim();
    try {
      const job = await post<{ id: string }>(`/projects/${name}/jobs`, {
        kind: "export",
        params,
      });
      setExportMsg(
        `export job ${job.id} started — watch it on the project page, then refresh here`,
      );
    } catch (err) {
      setExportErr(err instanceof Error ? err.message : String(err));
    }
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
  if (voices === null) return <p className="muted">loading…</p>;

  const selected = voices.find((v) => v.stem === sel) ?? null;

  return (
    <>
      <h1>Voices — {name}</h1>
      <p className="muted">
        exported .onnx voices in out/ ·{" "}
        <a href={`#/project/${name}`}>back to project</a>
      </p>
      {loadErr && <p className="error">{loadErr}</p>}

      {voices.length === 0 ? (
        <p className="muted">
          no exported voices yet — export a checkpoint below after a training
          run
        </p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>voice</th>
              <th>quality</th>
              <th>epoch</th>
              <th>size</th>
              <th>exported</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {voices.map((v) => (
              <tr
                key={v.stem}
                onClick={() => setSel(v.stem)}
                style={{ cursor: "pointer" }}
                className={v.stem === sel ? "selected" : undefined}
              >
                <td>{v.stem}</td>
                <td>{v.quality ?? "-"}</td>
                <td>{v.checkpoint_epoch ?? "-"}</td>
                <td>{fmtMB(v.size_bytes)}</td>
                <td>{v.mtime.replace("T", " ").slice(0, 16)}</td>
                <td>
                  {v.problems.length > 0 && (
                    <span className="chip chip-error" title={v.problems.join("; ")}>
                      config problems
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2>Export a checkpoint</h2>
      <form onSubmit={(e) => { e.preventDefault(); void doExport(); }}>
        <div className="row">
          <select value={ckptSel} onChange={(e) => setCkptSel(e.target.value)}>
            <option value="">latest checkpoint (auto)</option>
            {ckpts
              .filter((c) => c.source === "run")
              .map((c) => (
                <option key={c.path} value={c.path}>
                  {c.tier} · {c.name}
                  {c.epoch != null ? ` · epoch ${c.epoch}` : ""}
                </option>
              ))}
          </select>
          <input
            value={voiceName}
            onChange={(e) => {
              nameTouched.current = true;
              setVoiceName(e.target.value);
            }}
            placeholder={`voice name (default: ${name}-${tier ?? "…"})`}
            style={{ width: "24em" }}
          />
          <button type="submit">export</button>
        </div>
      </form>
      <p className="muted">
        the .onnx stem, the config's dataset field, and this name must agree —
        export enforces it, because a mismatched voice either never appears or
        throws VoiceNotFoundError where it is used
      </p>
      {exportMsg && <p className="muted">{exportMsg}</p>}
      {exportErr && <p className="error">{exportErr}</p>}

      {selected && (
        <VoiceTuner
          key={selected.stem}
          name={name}
          voice={selected}
          onTouched={() => void reload()}
        />
      )}
    </>
  );
}

// Slider + say panel for one voice. Local slider state; `save` persists
// into the .onnx.json (PATCH), `say` sends the sliders explicitly so the
// preview always matches what is on screen, saved or not.
function VoiceTuner(props: {
  name: string;
  voice: VoiceInfo;
  onTouched: () => void;
}) {
  const { name, voice, onTouched } = props;
  const [lengthScale, setLengthScale] = useState(
    String(voice.inference.length_scale ?? 1.0),
  );
  const [noiseScale, setNoiseScale] = useState(
    String(voice.inference.noise_scale ?? 0.667),
  );
  const [noiseW, setNoiseW] = useState(String(voice.inference.noise_w ?? 0.8));
  const [text, setText] = useState(
    "All it took was nine thousand credits and a very patient machine.",
  );
  const [saying, setSaying] = useState(false);
  const [sayErr, setSayErr] = useState("");
  const [saveMsg, setSaveMsg] = useState("");
  const [saveErr, setSaveErr] = useState("");
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const lastUrl = useRef<string | null>(null);

  useEffect(
    () => () => {
      if (lastUrl.current !== null) URL.revokeObjectURL(lastUrl.current);
    },
    [],
  );

  function currentParams() {
    return {
      length_scale: parseFloat(lengthScale),
      noise_scale: parseFloat(noiseScale),
      noise_w: parseFloat(noiseW),
    };
  }

  async function doSay() {
    setSayErr("");
    setSaying(true);
    try {
      const blob = await postWav(`/projects/${name}/voices/${voice.stem}/say`, {
        text,
        ...currentParams(),
      });
      if (lastUrl.current !== null) URL.revokeObjectURL(lastUrl.current);
      const url = URL.createObjectURL(blob);
      lastUrl.current = url;
      setAudioUrl(url);
    } catch (err) {
      setSayErr(err instanceof Error ? err.message : String(err));
    } finally {
      setSaying(false);
    }
  }

  async function doSave() {
    setSaveErr("");
    setSaveMsg("");
    try {
      await patch(`/projects/${name}/voices/${voice.stem}`, currentParams());
      setSaveMsg("saved into the .onnx.json — this is what say uses by default");
      onTouched();
    } catch (err) {
      setSaveErr(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <>
      <h2>Tune — {voice.stem}</h2>
      {voice.problems.length > 0 && (
        <p className="error">export.verify problems: {voice.problems.join("; ")}</p>
      )}
      <div className="row">
        <label className="inline">
          speed (length_scale) {lengthScale}
          <input
            type="range"
            min="0.1"
            max="3"
            step="0.05"
            value={lengthScale}
            onChange={(e) => setLengthScale(e.target.value)}
          />
        </label>
      </div>
      <div className="row">
        <label className="inline">
          pitch variation (noise_scale) {noiseScale}
          <input
            type="range"
            min="0"
            max="2"
            step="0.05"
            value={noiseScale}
            onChange={(e) => setNoiseScale(e.target.value)}
          />
        </label>
      </div>
      <div className="row">
        <label className="inline">
          phoneme width (noise_w) {noiseW}
          <input
            type="range"
            min="0"
            max="2"
            step="0.05"
            value={noiseW}
            onChange={(e) => setNoiseW(e.target.value)}
          />
        </label>
      </div>
      <div className="row">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          style={{ width: "34em" }}
          placeholder="text to say"
        />
        <button onClick={() => void doSay()} disabled={saying}>
          {saying ? "synthesizing…" : "say"}
        </button>
        <button onClick={() => void doSave()}>save tuning</button>
      </div>
      {sayErr && <p className="error">{sayErr}</p>}
      {saveMsg && <p className="muted">{saveMsg}</p>}
      {saveErr && <p className="error">{saveErr}</p>}
      {audioUrl && <audio controls src={audioUrl} />}
      <p className="muted">
        downloads:{" "}
        <a href={`/api/projects/${name}/voices/${voice.stem}/download?file=onnx`}>
          {voice.stem}.onnx
        </a>{" "}
        ·{" "}
        <a
          href={`/api/projects/${name}/voices/${voice.stem}/download?file=json`}
        >
          {voice.stem}.onnx.json
        </a>{" "}
        (both files are required to use the voice)
      </p>
    </>
  );
}
