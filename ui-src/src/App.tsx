import { useEffect, useState } from "react";
import { get } from "./api";
import { useHashRoute } from "./hooks";
import { ActivityPage } from "./pages/Activity";
import { AuditPage } from "./pages/Audit";
import { DoctorPage } from "./pages/Doctor";
import { NewProjectPage } from "./pages/NewProject";
import { PreparePage } from "./pages/Prepare";
import { ProjectPage } from "./pages/Project";
import { ProjectsPage } from "./pages/Projects";
import { SourcesPage } from "./pages/Sources";
import { TranscribePage } from "./pages/Transcribe";
import { TrainPage } from "./pages/Train";
import { VoicesPage } from "./pages/Voices";
import { ProjectShell, ThemeToggle } from "./shell";
import type { StageName } from "./types";

type Route =
  | { page: "projects" }
  | { page: "new" }
  | { page: "doctor" }
  | { page: "project"; name: string }
  | { page: "activity"; name: string }
  | { page: "sources"; name: string }
  | { page: "transcribe"; name: string }
  | { page: "prepare"; name: string }
  | { page: "audit"; name: string }
  | { page: "train"; name: string }
  | { page: "voices"; name: string };

// Same hash routes the pre-React UI used: bookmarks and muscle memory
// keep working across the cutover (workorder-04 A4). New pages follow
// the same shape. The /activity suffix must be tested before the bare
// project route or it would parse as a project name.
function parseRoute(hash: string): Route {
  const h = hash || "#/projects";
  if (h === "#/new") return { page: "new" };
  if (h === "#/doctor") return { page: "doctor" };
  if (h.startsWith("#/project/")) {
    const rest = h.slice("#/project/".length);
    if (rest.endsWith("/activity")) {
      return {
        page: "activity",
        name: decodeURIComponent(rest.slice(0, -"/activity".length)),
      };
    }
    return { page: "project", name: decodeURIComponent(rest) };
  }
  if (h.startsWith("#/sources/"))
    return { page: "sources", name: decodeURIComponent(h.slice("#/sources/".length)) };
  if (h.startsWith("#/transcribe/"))
    return { page: "transcribe", name: decodeURIComponent(h.slice("#/transcribe/".length)) };
  if (h.startsWith("#/prepare/"))
    return { page: "prepare", name: decodeURIComponent(h.slice("#/prepare/".length)) };
  if (h.startsWith("#/audit/"))
    return { page: "audit", name: decodeURIComponent(h.slice("#/audit/".length)) };
  if (h.startsWith("#/train/"))
    return { page: "train", name: decodeURIComponent(h.slice("#/train/".length)) };
  if (h.startsWith("#/voices/"))
    return { page: "voices", name: decodeURIComponent(h.slice("#/voices/".length)) };
  return { page: "projects" };
}

// Routes the shell wraps, with the stage whose gate the shell shows.
// The overview and Activity have no single stage (stage: null).
const SHELL_ROUTES: Record<
  Exclude<Route, { page: "projects" | "new" | "doctor" }>["page"],
  StageName | null
> = {
  project: null,
  activity: null,
  sources: "sources",
  transcribe: "transcribe",
  prepare: "prepare",
  audit: "audit",
  train: "train",
  voices: "voices",
};

function Health() {
  const [text, setText] = useState("");
  useEffect(() => {
    get<{ version: string }>("/health")
      .then((h) => setText(`v${h.version}`))
      .catch(() => setText("API unreachable"));
  }, []);
  return <span className="muted">{text}</span>;
}

export default function App() {
  const hash = useHashRoute();
  const route = parseRoute(hash);

  // The old global header stays on the three non-shell routes; on
  // project routes the IdentityBar carries the brand and the shell IS
  // the health indicator (a dead API reads "stale" in the status line).
  const shellStage =
    route.page in SHELL_ROUTES
      ? SHELL_ROUTES[route.page as keyof typeof SHELL_ROUTES]
      : null;

  let page;
  switch (route.page) {
    case "new":
      page = <NewProjectPage />;
      break;
    case "doctor":
      page = <DoctorPage />;
      break;
    case "project":
      page = (
        <ProjectShell key={route.name} name={route.name} stage={null}>
          <ProjectPage name={route.name} />
        </ProjectShell>
      );
      break;
    case "activity":
      page = (
        <ProjectShell key={route.name} name={route.name} stage={null}>
          <ActivityPage name={route.name} />
        </ProjectShell>
      );
      break;
    case "sources":
      page = (
        <ProjectShell key={route.name} name={route.name} stage="sources">
          <SourcesPage name={route.name} />
        </ProjectShell>
      );
      break;
    case "transcribe":
      page = (
        <ProjectShell key={route.name} name={route.name} stage="transcribe">
          <TranscribePage name={route.name} />
        </ProjectShell>
      );
      break;
    case "prepare":
      page = (
        <ProjectShell key={route.name} name={route.name} stage="prepare">
          <PreparePage name={route.name} />
        </ProjectShell>
      );
      break;
    case "audit":
      page = (
        <ProjectShell key={route.name} name={route.name} stage="audit">
          <AuditPage name={route.name} />
        </ProjectShell>
      );
      break;
    case "train":
      page = (
        <ProjectShell key={route.name} name={route.name} stage="train">
          <TrainPage name={route.name} />
        </ProjectShell>
      );
      break;
    case "voices":
      page = (
        <ProjectShell key={route.name} name={route.name} stage="voices">
          <VoicesPage name={route.name} />
        </ProjectShell>
      );
      break;
    default:
      page = <ProjectsPage />;
  }

  if (shellStage !== null || route.page in SHELL_ROUTES) {
    return page; // shell routes render their own chrome
  }
  return (
    <>
      <header>
        <strong>piper-trainer</strong>
        <nav>
          <a href="#/projects">Projects</a>
          <a href="#/new">New project</a>
          <a href="#/doctor">Doctor</a>
        </nav>
        <span className="spacer" />
        <ThemeToggle />
        <Health />
      </header>
      <main>{page}</main>
    </>
  );
}
