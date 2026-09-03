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
  };
  created_at?: string;
}

// Server-pushed websocket frames on /api/jobs/{id}/stream.
export type StreamMsg =
  | { type: "log_reset"; text: string }
  | { type: "log"; line: string }
  | { type: "state"; job: Job };
