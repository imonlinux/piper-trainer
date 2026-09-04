// Response shapes of the /api surface (src/piper_trainer/api/app.py).
// Kept hand-written and narrow: only fields the UI renders.

export interface Progress {
  total: number;
  unit?: string;
  current?: number;
}

export type JobState =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "interrupted"
  | string;

export interface Job {
  id: string;
  kind: string;
  stage: string | null;
  project: string;
  params: Record<string, unknown>;
  state: JobState;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  exit_code: number | null;
  pid: number | null;
  progress: Progress | null;
  result: Record<string, unknown> | null;
  artifacts: string[];
  error: string | null;
}

export interface ProjectSummary {
  name: string;
  path: string;
  clips: number;
  minutes: number | null;
  tiers_trained: string[];
  last_job: { kind: string; state: string } | null;
  prepare_params?: Record<string, unknown> | null;
}

export interface ProjectConfig {
  espeak_voice?: string | null;
  tier?: string | null;
  catalog_path?: string | null;
  target_epochs?: number | null;
  transcripts_provided?: boolean | null;
}

export interface ProjectDetail extends ProjectSummary {
  config: ProjectConfig;
  directories: Record<string, number>;
  dataset: { rows: number; malformed_lines: number; line_endings: string | null };
  voices: string[];
  checkpoints: unknown[];
  jobs: Job[];
}

export interface SourceInfo {
  name: string;
  codec?: string | null;
  sample_rate?: number | string | null;
  channels?: number | string | null;
  duration?: number | string | null;
}

export interface Peaks {
  duration: number;
  peaks: number[];
}

// One row of GET /api/projects/{name}/dataset (Audit screen §6.3).
// Text comes from metadata.csv (source of truth); scores join from
// audit.csv; `missing`/`quarantined` come from the filesystem + manifest.
export interface DatasetRow {
  id: string;
  text: string;
  duration: number | null;
  cps: number | null;
  lang_prob: number | null;
  missing: boolean;
  quarantined: boolean;
}

export interface QuarantineEntry {
  timestamp: string;
  clip_id: string;
  action: string;
  reasons: string;
  text: string;
}

export interface Dataset {
  rows: DatasetRow[];
  quarantine: QuarantineEntry[];
}

// One validation finding (result of a validate/clean job).
// `ids` names affected clip stems; findings without ids are dataset-level.
export interface Finding {
  level: "error" | "warn" | "info" | string;
  code: string;
  message: string;
  ids: string[];
  action: string | null;
}

export interface Catalog {
  source: "live" | "snapshot" | string;
  generated: string | null;
  // language -> locale -> voice -> [qualities]
  languages: Record<string, Record<string, Record<string, string[]>>>;
}

export interface DoctorCheck {
  status: "ok" | "error" | "info" | string;
  message: string;
}

export interface Doctor {
  ok: boolean;
  checks: DoctorCheck[];
  transcribe_devices: string[];
}

export interface Region {
  start: number;
  end: number;
}

export interface HistBin {
  from: number;
  to: number;
  count: number;
}

// One row of GET /api/projects/{name}/previews: the sweep entries.
// `result` differs by stage; both variants share the audio file list.
// `dir` is the preview's directory relative to the project root and is
// what playable file URLs go through (/files/{dir}/{name}).
export interface PreviewRow {
  id: string;
  dir: string;
  stage: "segment" | "denoise" | string;
  project: string;
  params: Record<string, unknown>;
  result: {
    clip_count?: number;
    audio?: string[];
    histogram?: HistBin[];
    seconds?: number;
    clips?: Region[];
    level?: { peak_dbfs: number; rms_dbfs: number; speech_dbfs: number };
    per_source?: {
      source: string;
      clips: number;
      seconds: number;
      level: { peak_dbfs: number; rms_dbfs: number; speech_dbfs: number } | null;
      error: string | null;
    }[];
    zeros?: string[];
  };
  created_at?: string;
}

// Server-pushed websocket frames on /api/jobs/{id}/stream.
export type StreamMsg =
  | { type: "log_reset"; text: string }
  | { type: "log"; line: string }
  | { type: "state"; job: Job };

// One entry of GET /api/projects/{name}/checkpoints. `source` splits the
// two warmstart families: catalog = fetched base voice (absolute `dir`
// plus the fetched file mapping), run = a checkpoint of this project's
// own runs-<tier> (project-relative `path`).
export interface Checkpoint {
  source: "catalog" | "run" | string;
  catalog_path?: string;
  dir?: string;
  files?: Record<string, string>;
  fetched_at?: string;
  tier?: string;
  path?: string;
  name?: string;
  epoch?: number | null;
  mtime?: string;
}

// GET /api/projects/{name}/train/preview (§6.4): steps math is always
// honest; the wall-clock projection only appears when a previous train
// run provides a real seconds-per-epoch basis.
export interface TrainPreview {
  clips: number;
  epochs: number;
  batch_size: number;
  sample_rate: number | null;
  steps_per_epoch: number | null;
  total_steps: number | null;
  seconds_per_epoch: number | null;
  projected_seconds: number | null;
  basis: string | null;
}

// GET /api/projects/{name}/voices (§6.5): one exported voice in out/.
// `problems` mirrors export.verify — empty means the name agreement
// (.onnx stem == config dataset field) and config shape are intact.
export interface VoiceInfo {
  stem: string;
  size_bytes: number;
  mtime: string;
  checkpoint_epoch: number | null;
  quality: string | null;
  language: string | null;
  espeak_voice: string | null;
  inference: Record<string, number>;
  problems: string[];
}
