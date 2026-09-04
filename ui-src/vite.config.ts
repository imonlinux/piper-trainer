import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The built app is served by FastAPI's StaticFiles mount at /ui (see
// src/piper_trainer/api/app.py): UI_DIR/index.html answers at /ui/. base
// must match or every asset URL 404s. The Docker build overrides outDir on
// the command line; this repo-relative one is for local `npm run build` so
// the bundle lands where the running server (and the ptt test container)
// can serve it without any copy step.
export default defineConfig({
  plugins: [react()],
  base: "/ui/",
  build: { outDir: "../src/piper_trainer/ui", emptyOutDir: true },
  server: {
    port: 5173,
    // `vite dev` against a ./run.sh serve backend. ws: true carries the
    // /api/jobs/{id}/stream websocket through the proxy.
    proxy: {
      "/api": { target: "http://localhost:8000", ws: true },
    },
  },
});
