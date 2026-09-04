import { useEffect, useState } from "react";
import { get } from "./api";
import { useHashRoute } from "./hooks";
import { DoctorPage } from "./pages/Doctor";
import { NewProjectPage } from "./pages/NewProject";
import { PreparePage } from "./pages/Prepare";
import { ProjectPage } from "./pages/Project";
import { ProjectsPage } from "./pages/Projects";

type Route =
  | { page: "projects" }
  | { page: "new" }
  | { page: "doctor" }
  | { page: "project"; name: string }
  | { page: "prepare"; name: string };

// Same hash routes the pre-React UI used: bookmarks and muscle memory
// keep working across the cutover.
function parseRoute(hash: string): Route {
  const h = hash || "#/projects";
  if (h === "#/new") return { page: "new" };
  if (h === "#/doctor") return { page: "doctor" };
  if (h.startsWith("#/project/"))
    return { page: "project", name: decodeURIComponent(h.slice(10)) };
  if (h.startsWith("#/prepare/"))
    return { page: "prepare", name: decodeURIComponent(h.slice(10)) };
  return { page: "projects" };
}

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
  let page;
  switch (route.page) {
    case "new":
      page = <NewProjectPage />;
      break;
    case "doctor":
      page = <DoctorPage />;
      break;
    case "project":
      page = <ProjectPage key={route.name} name={route.name} />;
      break;
    case "prepare":
      page = <PreparePage key={route.name} name={route.name} />;
      break;
    default:
      page = <ProjectsPage />;
  }
  return (
    <>
      <header>
        <strong>piper-trainer</strong>
        <span className="tag">react</span>
        <nav>
          <a href="#/projects">Projects</a>
          <a href="#/new">New project</a>
          <a href="#/doctor">Doctor</a>
        </nav>
        <Health />
      </header>
      <main>{page}</main>
    </>
  );
}
