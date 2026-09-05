import { useEffect, useState } from "react";
import { get, post } from "../api";
import { usePoll } from "../hooks";
import { Consequence, Panel, kindText } from "../shell";
import type { Doctor, Job } from "../types";

// Transcribe (workorder-04 A5): batch transcription with the normalize
// toggle and the retranscribe consequence, plus the informational
// device gate from doctor. The gate teaches; it doesn't lock — batch
// transcription works regardless.

export function TranscribePage({ name }: { name: string }) {
  const [devices, setDevices] = useState<string[] | null>(null);
  const [normalize, setNormalize] = useState(true);
  const [running, setRunning] = useState(false);
  const [actionError, setActionError] = useState("");
  const [lastMsg, setLastMsg] = useState("");
  // last summary comes from the newest transcribe job's result
  const [last, setLast] = useState<Job | null>(null);

  useEffect(() => {
    get<Doctor>("/doctor")
      .then((d) => setDevices(d.transcribe_devices ?? []))
      .catch(() => setDevices(null));
  }, []);

  // Track the newest transcribe job: live state while it runs, the
  // stats summary once it succeeds.
  usePoll(async () => {
    try {
      const jobs = await get<Job[]>(
        `/projects/${encodeURIComponent(name)}/jobs`,
      );
      const t = jobs.find((j) => j.kind === "transcribe") ?? null;
      setLast(t);
      setRunning(
        t !== null && (t.state === "running" || t.state === "queued"),
      );
    } catch {
      // transient; the next tick retries
    }
  }, 2000);

  async function runTranscribe(): Promise<void> {
    setActionError("");
    setLastMsg("");
    try {
      await post<Job>(`/projects/${name}/jobs`, {
        kind: "transcribe",
        params: normalize ? {} : { normalize: false },
      });
      setLastMsg("transcription started — progress is on the status line up top");
    } catch (ex) {
      setActionError(String(ex));
    }
  }

  // Clip count for the consequence line: from the last job's stats if
  // we have one, else the generic wording.
  const stats = (last?.result?.stats ?? undefined) as
    | {
        clips?: number;
        transcribed?: number;
        skipped?: number;
        total_seconds?: number;
        normalized?: { engine?: string | null };
      }
    | undefined;
  const clips = stats?.clips;

  return (
    <>
      <h1>transcribe</h1>
      <p className="muted">
        writes a transcript for every clip in the dataset using a large
        speech model — batch work, no latency requirement
      </p>

      <Panel
        title="record new audio"
        help="Record straight into the browser and ingest the take as a source. This needs a microphone the server can see; doctor lists what was found."
      >
        {devices === null ? (
          <p className="muted">checking doctor…</p>
        ) : devices.length === 0 ? (
          <p>
            no transcription devices found —{" "}
            <a href="#/doctor">doctor shows what's installed</a>
          </p>
        ) : (
          <p className="muted">
            devices found: {devices.join(", ")} — record-and-ingest arrives
            with the guided chains; until then record externally and add the
            file on the sources page
          </p>
        )}
      </Panel>

      <Panel
        title="batch transcribe"
        help="Whisper writes a transcript per clip. Fresh clips are always transcribed; the toggle below only matters for clips that already have text."
      >
        <label className="inline">
          <input
            type="checkbox"
            checked={normalize}
            onChange={(e) => setNormalize(e.target.checked)}
          />{" "}
          normalize text (Mr. → Mister, digits → words)
        </label>
        <Consequence>
          {clips
            ? `retranscribe overwrites the transcript column for all ${clips.toLocaleString()} clips — hand corrections are lost`
            : "retranscribe overwrites the transcript column for every clip — hand corrections are lost"}
        </Consequence>
        <p className="row">
          <button disabled={running} onClick={() => void runTranscribe()}>
            {running ? "transcribing…" : "run transcribe"}
          </button>
          {lastMsg && <span className="muted">{lastMsg}</span>}
          {actionError && <span className="error">{actionError}</span>}
        </p>
      </Panel>

      {last && last.state !== "running" && last.state !== "queued" && (
        <Panel title="last transcription" help="Numbers from the most recent transcription job's result.">
          {last.state === "succeeded" && stats ? (
            <p>
              {stats.clips ?? 0} clips · {stats.transcribed ?? 0} transcribed
              {stats.skipped ? `, ${stats.skipped} skipped (kept existing text)` : ""}
              {stats.total_seconds
                ? ` · ${Math.round(stats.total_seconds / 60)} min of audio`
                : ""}
              {stats.normalized?.engine === null && (
                <>
                  {" "}
                  <span className="error">
                    numbers left as digits — inflect is not installed on the
                    server (pip install 'piper-trainer[runtime]')
                  </span>
                </>
              )}
            </p>
          ) : (
            <p>
              <span className={`state-${last.state}`}>{last.state}</span>
              {last.error ? ` — ${last.error}` : ""}
            </p>
          )}
          <p className="muted mono">{kindText(last.kind)} · {last.id}</p>
        </Panel>
      )}
    </>
  );
}
