import { useEffect, useRef } from "react";
import type { Peaks, Region } from "../types";

// Server-side peaks (decision §8.1) drawn on a canvas, with the preview's
// detected speech regions overlaid. Same rendering as Bones' drawWave.

export function Wave(props: { data: Peaks | null; regions: Region[] }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const W = cv.width;
    const H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#f8f8f8";
    ctx.fillRect(0, 0, W, H);
    const data = props.data;
    if (!data) return;
    const dur = data.duration || 1;
    const n = data.peaks.length;
    const bw = W / n;
    ctx.fillStyle = "#888";
    for (let i = 0; i < n; i++) {
      const h = Math.max(1, data.peaks[i] * (H - 10));
      ctx.fillRect(i * bw, (H - h) / 2, Math.max(1, bw), h);
    }
    for (const r of props.regions) {
      const x0 = (r.start / dur) * W;
      const x1 = (r.end / dur) * W;
      ctx.fillStyle = "rgba(35, 134, 54, 0.22)";
      ctx.fillRect(x0, 0, x1 - x0, H);
      ctx.strokeStyle = "#22863a";
      ctx.strokeRect(x0 + 0.5, 0.5, x1 - x0 - 1, H - 1);
    }
  }, [props.data, props.regions]);

  return <canvas ref={ref} width={1000} height={160} style={{ width: "100%" }} />;
}
