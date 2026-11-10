import { useState } from "react";
import { Figure, Legend, Segmented, Stat, color, useWidth } from "../kit";

// Data printed by `python3 scripts/figures/10b-profiling.py --viz`, which parses
// the hand-built traces in content/labs/10b-profiling/fixture_traces.py with the
// lab solution. Times are microseconds from the first GPU event. Nothing here is
// a GPU measurement. Each span is [start, end, stream].
const DATA = {"launch_bound": {"window": 193, "busy": 0.3627, "class": "launch_bound", "spans": [[0, 4, 7], [10, 13, 7], [20, 24, 7], [30, 33, 7], [40, 44, 7], [50, 53, 7], [60, 64, 7], [70, 73, 7], [80, 84, 7], [90, 93, 7], [100, 104, 7], [110, 113, 7], [120, 124, 7], [130, 133, 7], [140, 144, 7], [150, 153, 7], [160, 164, 7], [170, 173, 7], [180, 184, 7], [190, 193, 7]], "gaps": [[4, 10, "idle"], [13, 20, "idle"], [24, 30, "idle"], [33, 40, "idle"], [44, 50, "idle"], [53, 60, "idle"], [64, 70, "idle"], [73, 80, "idle"], [84, 90, "idle"], [93, 100, "idle"], [104, 110, "idle"], [113, 120, "idle"], [124, 130, "idle"], [133, 140, "idle"], [144, 150, "idle"], [153, 160, "idle"], [164, 170, "idle"], [173, 180, "idle"], [184, 190, "idle"]]}, "sync_bound": {"window": 930, "busy": 0.871, "class": "sync_bound", "spans": [[0, 50, 7], [50, 100, 7], [100, 150, 7], [150, 200, 7], [200, 250, 7], [250, 300, 7], [300, 350, 7], [350, 400, 7], [400, 410, 7], [530, 580, 7], [580, 630, 7], [630, 680, 7], [680, 730, 7], [730, 780, 7], [780, 830, 7], [830, 880, 7], [880, 930, 7]], "gaps": [[410, 530, "host_sync"]]}, "multistream": {"window": 260, "busy": 0.8077, "class": "healthy", "spans": [[0, 100, 7], [50, 150, 8], [200, 260, 9]], "gaps": [[150, 200, "idle"]]}} as const;

type Name = keyof typeof DATA;

const OPTIONS: { value: Name; label: string }[] = [
  { value: "launch_bound", label: "Launch-bound" },
  { value: "sync_bound", label: "Sync-bound" },
  { value: "multistream", label: "Two streams" },
];

export default function TraceTimeline() {
  const [name, setName] = useState<Name>("launch_bound");
  const [ref, width] = useWidth();
  const d = DATA[name];
  const left = 12;
  const right = 12;
  const innerW = Math.max(width - left - right, 40);
  const xOf = (t: number) => left + (t / d.window) * innerW;
  const streams = Array.from(new Set(d.spans.map((s) => s[2]))).sort((a, b) => a - b);
  const laneH = 26;
  const top = 10;
  const height = top + streams.length * laneH + 26;
  const idle = d.gaps.reduce((sum, g) => sum + (g[1] - g[0]), 0);

  return (
    <Figure
      title="Where the GPU waits"
      controls={<Segmented label="Trace" value={name} options={OPTIONS} onChange={setName} />}
      readout={
        <>
          <Stat label="GPU busy" value={`${(d.busy * 100).toFixed(1)}%`} tone={color.ok} />
          <Stat label="Window" value={`${d.window.toFixed(0)} µs`} tone={color.b} />
          <Stat label="Idle gaps" value={`${d.gaps.length} / ${idle.toFixed(0)} µs`} tone={color.accent} />
          <Stat label="Verdict" value={d.class.replace("_", "-")} tone={color.c} />
        </>
      }
      caption="Each block is a GPU event on one stream. Rose spans are idle gaps. The sync-bound gap begins where a cudaStreamSynchronize ends. The two-stream trace shows 21/26 busy, because overlapping events count once."
    >
      <div ref={ref}>
        <svg width={width} height={height} role="img" aria-label="Timeline of GPU events and idle gaps for the selected trace">
          {d.gaps.map((g, i) => (
            <rect
              key={`g${i}`}
              x={xOf(g[0])}
              y={top}
              width={Math.max(1, xOf(g[1]) - xOf(g[0]))}
              height={streams.length * laneH}
              style={{ fill: g[2] === "host_sync" ? color.bad : color.line, opacity: g[2] === "host_sync" ? 0.45 : 0.6 }}
            />
          ))}
          {d.spans.map((s, i) => (
            <rect
              key={i}
              x={xOf(s[0])}
              y={top + streams.indexOf(s[2]) * laneH + 4}
              width={Math.max(1, xOf(s[1]) - xOf(s[0]) - 0.5)}
              height={laneH - 8}
              style={{ fill: color.ok }}
            />
          ))}
          <text x={left} y={height - 8} style={{ fill: color.muted, fontSize: 11 }}>0 µs</text>
          <text x={left + innerW} y={height - 8} textAnchor="end" style={{ fill: color.muted, fontSize: 11 }}>
            {d.window.toFixed(0)} µs
          </text>
        </svg>
        <Legend
          items={[
            { label: "GPU event", tone: color.ok },
            { label: "Idle gap", tone: color.line },
            { label: "Gap after a host sync", tone: color.bad },
          ]}
        />
      </div>
    </Figure>
  );
}
