import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./app.css";

// StrictMode double-invokes effects in dev on purpose: a socket or
// timer that cannot survive that is a cleanup bug, which is exactly
// the class of bug this UI has had before (Bones finding 7).
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
