import React, { useState, useRef } from "react";
import { fmt } from "../lib/format";
import type { SignalsPayload } from "../api";

interface SignalPanelProps {
  signals: SignalsPayload | null;
  loading: boolean;
  error: string;
}

const CLASS_INFO: Record<string, { short: string; label: string; css: string }> = {
  compute_bound: { short: "CB", label: "Compute-Bound", css: "badge badge--accent" },
  memory_bound: { short: "MB", label: "Memory-Bound", css: "badge badge--warning" },
  balanced: { short: "BA", label: "Balanced", css: "badge badge--success" },
  unknown: { short: "?", label: "Unknown", css: "badge badge--neutral" },
};

function WorkloadClassBadge({ wclass }: { wclass: string }) {
  const info = CLASS_INFO[wclass] || CLASS_INFO.unknown;
  return <span className={info.css} title={`${info.label} (${wclass})`}>{info.label}</span>;
}

function ConfidenceBar({ value, label }: { value: number; label?: string }) {
  const pct = Math.round(value * 100);
  const color = pct >= 60 ? "var(--success)" : pct >= 30 ? "var(--warning)" : "var(--danger)";
  return (
    <div className="confidence-bar" title={label || `${pct}%`}>
      <div className="confidence-bar__fill" style={{ width: `${pct}%`, background: color }} />
      <span className="confidence-bar__label">{pct}%</span>
    </div>
  );
}

function Collapsible({ title, defaultOpen = false, badge, children }: {
  title: string; defaultOpen?: boolean; badge?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="collapsible">
      <button className="collapsible__toggle" onClick={() => setOpen(!open)}>
        <span className="collapsible__arrow">{open ? "▼" : "▶"}</span>
        <span>{title}</span>
        {badge ? <span className="badge badge--neutral" style={{ marginLeft: 6, fontSize: 10 }}>{badge}</span> : null}
      </button>
      {open ? <div className="collapsible__body">{children}</div> : null}
    </div>
  );
}

function ScalingChart({
  observations,
  posteriorCurve,
  unit,
  color,
  lengthscale,
  sigma2f,
}: {
  observations?: { x: number[]; y: number[] };
  posteriorCurve?: { x: number[]; mean: number[]; upper_2sigma: number[]; lower_2sigma: number[] } | null;
  unit: string;
  color: string;
  lengthscale?: number;
  sigma2f?: number;
}) {
  void sigma2f;
  const W = 320;
  const H = 140;
  const PAD = { left: 44, right: 14, top: 14, bottom: 20 };
  const plotW = W - PAD.left - PAD.right;
  const plotH = H - PAD.top - PAD.bottom;

  const obsX = observations?.x || [];
  const obsY = observations?.y || [];
  if (obsX.length === 0 && !posteriorCurve) return null;

  const allX = [...obsX];
  const allY = [...obsY];
  if (posteriorCurve) {
    allX.push(...posteriorCurve.x);
    allY.push(...posteriorCurve.upper_2sigma);
    allY.push(...posteriorCurve.lower_2sigma);
  }
  if (allX.length === 0) return null;

  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const yMin = Math.min(...allY);
  const yMax = Math.max(...allY);
  const xPad = Math.max((xMax - xMin) * 0.05, 1);
  const yPad = Math.max((yMax - yMin) * 0.1, 0.1);
  const x0 = xMin - xPad;
  const x1 = xMax + xPad;
  const y0 = Math.max(0, yMin - yPad);
  const y1 = yMax + yPad;

  const toX = (x: number) => PAD.left + ((x - x0) / (x1 - x0)) * plotW;
  const toY = (y: number) => PAD.top + plotH - ((y - y0) / (y1 - y0)) * plotH;

  let bandPath = "";
  let meanPath = "";
  if (posteriorCurve && posteriorCurve.x.length > 1) {
    const pts = posteriorCurve;
    bandPath = `M ${toX(pts.x[0])} ${toY(pts.upper_2sigma[0])}`;
    for (let i = 1; i < pts.x.length; i++) {
      bandPath += ` L ${toX(pts.x[i])} ${toY(pts.upper_2sigma[i])}`;
    }
    for (let i = pts.x.length - 1; i >= 0; i--) {
      bandPath += ` L ${toX(pts.x[i])} ${toY(pts.lower_2sigma[i])}`;
    }
    bandPath += " Z";
    meanPath = `M ${toX(pts.x[0])} ${toY(pts.mean[0])}`;
    for (let i = 1; i < pts.x.length; i++) {
      meanPath += ` L ${toX(pts.x[i])} ${toY(pts.mean[i])}`;
    }
  }

  return (
    <svg width={W} height={H} className="scaling-chart">
      <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + plotH} stroke="#ccc" strokeWidth={1} />
      <line x1={PAD.left} y1={PAD.top + plotH} x2={PAD.left + plotW} y2={PAD.top + plotH} stroke="#ccc" strokeWidth={1} />
      {bandPath ? <path d={bandPath} fill={color} opacity={0.1} /> : null}
      {meanPath ? <path d={meanPath} fill="none" stroke={color} strokeWidth={2} opacity={0.8} /> : null}
      {obsX.map((px, i) => (
        <circle key={`obs-${i}`} cx={toX(px)} cy={toY(obsY[i])} r={2.5} fill={color} opacity={0.5}>
          <title>x={px}, y={obsY[i].toFixed(unit === "MiB" ? 0 : 3)} {unit}</title>
        </circle>
      ))}
      <text x={PAD.left - 4} y={toY(yMax) + 3} textAnchor="end" fontSize={9} fill="#666">{yMax.toFixed(unit === "MiB" ? 0 : 2)}</text>
      <text x={PAD.left - 4} y={toY(yMin) + 3} textAnchor="end" fontSize={9} fill="#666">{yMin.toFixed(unit === "MiB" ? 0 : 2)}</text>
      <text x={PAD.left + plotW / 2} y={H - 2} textAnchor="middle" fontSize={9} fill="#999">input size →</text>
      <text x={2} y={PAD.top + plotH / 2} textAnchor="start" fontSize={8} fill="#999" transform={`rotate(-90 8 ${PAD.top + plotH / 2})`}>{unit}</text>
      <rect x={PAD.left + 4} y={PAD.top + 1} width={8} height={6} fill={color} opacity={0.1} stroke={color} strokeWidth={0.5} />
      <text x={PAD.left + 15} y={PAD.top + 7} fontSize={7} fill="#999">±2σ</text>
      <line x1={PAD.left + 32} y1={PAD.top + 4} x2={PAD.left + 44} y2={PAD.top + 4} stroke={color} strokeWidth={2} opacity={0.8} />
      <text x={PAD.left + 47} y={PAD.top + 7} fontSize={7} fill="#999">GP mean</text>
      <circle cx={PAD.left + 72} cy={PAD.top + 4} r={2.5} fill={color} opacity={0.5} />
      <text x={PAD.left + 78} y={PAD.top + 7} fontSize={7} fill="#999">obs</text>
      {lengthscale != null ? (
        <text x={W - PAD.right} y={PAD.top + 7} textAnchor="end" fontSize={7} fill="#bbb">l={lengthscale}</text>
      ) : null}
    </svg>
  );
}

function GpuBaselineDetail({ gpuId, bl }: { gpuId: string; bl: any }) {
  return (
    <div className="config-baseline-gpu">
      {gpuId !== "*" ? (
        <span className="config-baseline-gpu__label badge badge--neutral">GPU {gpuId}</span>
      ) : null}
      <div className="config-baseline-card__metrics">
        <div className="config-baseline-metric">
          <span className="config-baseline-metric__label">VRAM</span>
          <span className="config-baseline-metric__value">
            {(bl.vram_mib?.n ?? 0) > 0 ? `${bl.vram_mib.mean.toFixed(0)} ± ${bl.vram_mib.std.toFixed(0)} MiB` : "—"}
          </span>
          <span className="small-muted">n={bl.vram_mib?.n ?? 0}, conf={((bl.vram_mib?.confidence ?? 0) * 100).toFixed(0)}%</span>
        </div>
        <div className="config-baseline-metric">
          <span className="config-baseline-metric__label">Latency</span>
          <span className="config-baseline-metric__value">
            {(bl.latency_sec?.n ?? 0) > 0 ? `${bl.latency_sec.mean.toFixed(2)} ± ${bl.latency_sec.std.toFixed(2)} s` : "—"}
          </span>
          <span className="small-muted">n={bl.latency_sec?.n ?? 0}, conf={((bl.latency_sec?.confidence ?? 0) * 100).toFixed(0)}%</span>
        </div>
      </div>
      {((bl.vram_mib?.observations?.x?.length ?? 0) > 0 || (bl.latency_sec?.observations?.x?.length ?? 0) > 0) ? (
        <div className="config-baseline-card__charts">
          {(bl.vram_mib?.observations?.x?.length ?? 0) > 0 ? (
            <div className="scaling-chart-wrap" style={{ gridColumn: 1 }}>
              <span className="scaling-chart-label">VRAM vs input size (GP)</span>
              <ScalingChart
                observations={bl.vram_mib.observations}
                posteriorCurve={bl.vram_mib.posterior_curve}
                unit="MiB" color="var(--accent)"
                lengthscale={bl.vram_mib.lengthscale}
                sigma2f={bl.vram_mib.sigma2_f}
              />
            </div>
          ) : null}
          {(bl.latency_sec?.observations?.x?.length ?? 0) > 0 ? (
            <div className="scaling-chart-wrap" style={{ gridColumn: 2 }}>
              <span className="scaling-chart-label">Latency vs input size (GP)</span>
              <ScalingChart
                observations={bl.latency_sec.observations}
                posteriorCurve={bl.latency_sec.posterior_curve}
                unit="sec" color="var(--info)"
                lengthscale={bl.latency_sec.lengthscale}
                sigma2f={bl.latency_sec.sigma2_f}
              />
            </div>
          ) : null}
        </div>
      ) : (
        <p className="small-muted">No GP data yet.</p>
      )}
    </div>
  );
}

const FIXED_COMP_COLORS: Record<string, string> = {
  rfdiffusion: "#4e79a7",
  proteinmpnn: "#f28e2b",
  esm: "#e15759",
  mmseqs2: "#76b7b2",
  colabfold: "#4b8f8c",
  protenix: "#59a14f",
  vina_gpu: "#edc948",
  "vina-gpu": "#edc948",
  diffdock: "#b07aa1",
  boltzgen: "#ff9da7",
};
const FALLBACK_COMP_COLORS: Record<string, string> = {};
const PALETTE = [
  "#4e79a7",
  "#f28e2b",
  "#e15759",
  "#76b7b2",
  "#59a14f",
  "#edc948",
  "#b07aa1",
  "#ff9da7",
  "#9c755f",
  "#bab0ac",
  "#6b6ecf",
  "#b5cf6b",
  "#e7ba52",
  "#ad494a",
  "#7b4173",
  "#3182bd",
  "#31a354",
  "#756bb1",
  "#636363",
  "#d6616b",
];
function compColor(comp: string): string {
  const key = String(comp || "").trim().toLowerCase();
  if (FIXED_COMP_COLORS[key]) {
    return FIXED_COMP_COLORS[key];
  }
  if (!FALLBACK_COMP_COLORS[key]) {
    FALLBACK_COMP_COLORS[key] = PALETTE[Object.keys(FALLBACK_COMP_COLORS).length % PALETTE.length];
  }
  return FALLBACK_COMP_COLORS[key];
}

type TimelineData = NonNullable<NonNullable<SignalsPayload["signals"]["campaign_scheduler"]>["gpu_timelines"]>;

const ZOOM_PRESETS = [
  { label: "All", sec: 0 },
  { label: "10m", sec: 600 },
  { label: "5m", sec: 300 },
  { label: "2m", sec: 120 },
  { label: "1m", sec: 60 },
  { label: "30s", sec: 30 },
  { label: "15s", sec: 15 },
  { label: "5s", sec: 5 },
];

function GpuTimelineGantt({ timelines, schedulerData }: { timelines: TimelineData; schedulerData?: any }) {
  const [zoomSec, setZoomSec] = useState(0);
  const svgRef = useRef<SVGSVGElement>(null);
  const legendRef = useRef<HTMLDivElement>(null);
  const gpuIds = Object.keys(timelines).sort();
  const allEntries = gpuIds.flatMap(gid => timelines[gid].entries);
  if (allEntries.length === 0) {
    return (
      <div style={{ marginTop: 12 }}>
        <h4 style={{ fontSize: 13, margin: "0 0 6px" }}>Predictive GPU Timeline</h4>
        <p className="small-muted">No active tasks — timeline is empty.</p>
      </div>
    );
  }

  const now = Date.now() / 1000;

  const effectiveEnd = (e: typeof allEntries[0]) => {
    if (e.completed) return e.start_time + e.elapsed_sec;
    if (e.is_predicted) return e.predicted_end_time;
    return e.start_time + e.elapsed_sec;
  };

  const minStart = Math.min(...allEntries.map(e => e.start_time));
  const maxEnd = Math.max(...allEntries.map(e => effectiveEnd(e)));
  const PREDICT_HORIZON_SEC = 1000;
  const horizonCap = now + PREDICT_HORIZON_SEC;
  const tStart = zoomSec > 0 ? Math.max(now - zoomSec, Math.min(now, minStart)) : Math.min(now, minStart);
  const tEnd = Math.min(horizonCap, Math.max(now + 10, maxEnd + 5));
  const duration = tEnd - tStart;

  const W = 900;
  const LABEL_W = 60;
  const PAD = { top: 28, bottom: 16, right: 12 };
  const plotW = W - LABEL_W - PAD.right;

  const pxPerSec = plotW / duration;
  const MIN_BAR_PX = 8;

  const SUB_ROW_H = 16;
  const SUB_GAP = 3;
  const gpuSubRowCounts: Record<string, number> = {};
  const entrySubRows: Record<string, number> = {};
  const uid = (e: { task_id: string; start_time: number }) =>
    `${e.task_id}@${e.start_time}`;

  for (const gpuId of gpuIds) {
    const initsByTask: Record<string, Array<typeof allEntries[0]>> = {};
    for (const e of timelines[gpuId].entries) {
      if (e.is_init) {
        const realId = e.task_id.replace(/__init_phase$/, "");
        (initsByTask[realId] ??= []).push(e);
      }
    }
    const findInit = (taskId: string, inferStart: number) => {
      const inits = initsByTask[taskId];
      if (!inits || inits.length === 0) return undefined;
      if (inits.length === 1) {
        const gap = inferStart - inits[0].predicted_end_time;
        return gap >= 0 && gap < 60 ? inits[0] : undefined;
      }
      let best: typeof allEntries[0] | undefined;
      let bestGap = Infinity;
      for (const ie of inits) {
        const gap = inferStart - ie.predicted_end_time;
        if (gap >= 0 && gap < bestGap) {
          best = ie;
          bestGap = gap;
        }
      }
      return best;
    };
    const rawNonInit = timelines[gpuId].entries.filter(e => !e.is_init);
    const actualForSubRow = rawNonInit.filter(e => !e.is_predicted);
    const predByComp: Record<string, typeof rawNonInit> = {};
    for (const e of rawNonInit.filter(e => e.is_predicted)) {
      (predByComp[e.component] ??= []).push(e);
    }
    const collapsedForSubRow: typeof rawNonInit = [];
    for (const [comp, group] of Object.entries(predByComp)) {
      const earliest = Math.min(...group.map(e => e.start_time));
      const latestEnd = Math.max(...group.map(e => effectiveEnd(e)));
      const s = { ...group[0] };
      s.task_id = `__pred_summary_${comp}_${gpuId}`;
      s.start_time = earliest;
      s.predicted_end_time = latestEnd;
      collapsedForSubRow.push(s as any);
    }
    const inferTaskIds = new Set(rawNonInit.map(e => e.task_id));
    const standaloneInits = timelines[gpuId].entries.filter(e => {
      if (!e.is_init) return false;
      const realId = e.task_id.replace(/__init_phase$/, "");
      return !inferTaskIds.has(realId);
    });
    const nonInit = [...actualForSubRow, ...collapsedForSubRow, ...standaloneInits]
      .sort((a, b) => a.start_time - b.start_time);
    const subRowEnds: number[] = [];
    const subRowComp: (string | undefined)[] = [];

    for (const e of nonInit) {
      const initEntry = e.is_init ? undefined : findInit(e.task_id, e.start_time);
      const effectiveStart = initEntry ? Math.min(initEntry.start_time, e.start_time) : e.start_time;
      const fullEnd = Math.max(effectiveEnd(e), e.predicted_end_time);
      const taskDuration = fullEnd - effectiveStart;
      const effectiveDuration = Math.max(taskDuration, MIN_BAR_PX / pxPerSec);
      const eEnd = effectiveStart + effectiveDuration;
      let assigned = -1;
      for (let r = 0; r < subRowEnds.length; r++) {
        if (subRowEnds[r] <= effectiveStart && subRowComp[r] === e.component) {
          assigned = r;
          subRowEnds[r] = eEnd;
          break;
        }
      }
      if (assigned < 0) {
        for (let r = 0; r < subRowEnds.length; r++) {
          if (subRowEnds[r] <= effectiveStart) {
            assigned = r;
            subRowEnds[r] = eEnd;
            subRowComp[r] = e.component;
            break;
          }
        }
      }
      if (assigned < 0) {
        assigned = subRowEnds.length;
        subRowEnds.push(eEnd);
        subRowComp.push(e.component);
      }
      entrySubRows[uid(e)] = assigned;
      if (initEntry) entrySubRows[uid(initEntry)] = assigned;
    }
    gpuSubRowCounts[gpuId] = Math.max(1, subRowEnds.length);
  }

  const gpuRowHeight = (gpuId: string) => gpuSubRowCounts[gpuId] * (SUB_ROW_H + SUB_GAP) + 10;
  const gpuRowY: Record<string, number> = {};
  let currentY = PAD.top;
  for (const gpuId of gpuIds) {
    gpuRowY[gpuId] = currentY;
    currentY += gpuRowHeight(gpuId);
  }
  const H = currentY + PAD.bottom;

  const toX = (t: number) => LABEL_W + ((t - tStart) / duration) * plotW;
  const nowX = toX(now);

  const targetTicks = 10;
  const rawInterval = duration / targetTicks;
  const niceIntervals = [5, 10, 15, 30, 60, 120, 300, 600];
  const tickInterval = niceIntervals.find(n => n >= rawInterval) || Math.ceil(rawInterval / 60) * 60;
  const ticks: number[] = [];
  let tick = Math.ceil(tStart / tickInterval) * tickInterval;
  while (tick <= tEnd) { ticks.push(tick); tick += tickInterval; }

  const comps = [...new Set(allEntries.map(e => e.component))].sort();
  const hasBackfill = allEntries.some(e => e.is_backfill);
  const hasDone = allEntries.some(e => e.completed);
  const hasPredicted = allEntries.some(e => e.is_predicted);
  const hasStale = allEntries.some(e => e.prediction_stale);
  const hasInit = allEntries.some(e => e.is_init);
  const hasKilled = allEntries.some(e => e.is_killed);

  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, margin: "0 0 6px" }}>
        <h4 style={{ fontSize: 13, margin: 0 }}>Predictive GPU Timeline</h4>
        <div style={{ display: "flex", gap: 2 }}>
          {ZOOM_PRESETS.map(z => (
            <button
              key={z.sec}
              onClick={() => setZoomSec(z.sec)}
              style={{
                fontSize: 10, padding: "1px 5px", cursor: "pointer",
                border: "1px solid #ccc", borderRadius: 3,
                background: zoomSec === z.sec ? "#4a90d9" : "#f5f5f5",
                color: zoomSec === z.sec ? "#fff" : "#555",
              }}
            >{z.label}</button>
          ))}
        </div>
        <button
          onClick={() => {
            const payload = {
              exported_at: new Date().toISOString(),
              epoch: Date.now() / 1000,
              campaign_scheduler: schedulerData || null,
            };
            const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `gpu_timeline_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
            a.click();
            URL.revokeObjectURL(url);
          }}
          style={{
            fontSize: 10, padding: "1px 6px", cursor: "pointer",
            border: "1px solid #4a90d9", borderRadius: 3,
            background: "#fff", color: "#4a90d9", marginLeft: 4,
          }}
        >Export JSON</button>
        <button
          onClick={() => {
            if (!svgRef.current) return;
            const EXPORT_PX_PER_SEC = 10;
            const EXPORT_LABEL_W = 70;
            const EXPORT_PAD_R = 16;

            const fullMinStart = Math.min(...allEntries.map(e => e.start_time));
            const fullMaxEnd = Math.max(...allEntries.map(e => {
              if (e.completed) return e.start_time + e.elapsed_sec;
              if (e.is_predicted) return e.predicted_end_time;
              return e.start_time + e.elapsed_sec;
            }));
            const fullDuration = Math.max(1, fullMaxEnd - fullMinStart);
            const exportPlotW = Math.ceil(fullDuration * EXPORT_PX_PER_SEC);
            const exportW = EXPORT_LABEL_W + exportPlotW + EXPORT_PAD_R;
            const exportTStart = fullMinStart;
            const exportTEnd = fullMaxEnd;

            const toExportX = (t: number) =>
              EXPORT_LABEL_W + ((t - exportTStart) / fullDuration) * exportPlotW;

            const exportSubRows: Record<string, number> = {};
            const exportGpuSubRowCounts: Record<string, number> = {};
            for (const gid of gpuIds) {
              const initByTaskE: Record<string, Array<typeof allEntries[0]>> = {};
              for (const e of timelines[gid].entries) {
                if (e.is_init) {
                  const rid = e.task_id.replace(/__init_phase$/, "");
                  (initByTaskE[rid] ??= []).push(e);
                }
              }
              const findInitE = (tid: string, iStart: number) => {
                const arr = initByTaskE[tid];
                if (!arr || !arr.length) return undefined;
                if (arr.length === 1) {
                  const g = iStart - arr[0].predicted_end_time;
                  return g >= 0 && g < 60 ? arr[0] : undefined;
                }
                let b: typeof allEntries[0] | undefined; let bg = Infinity;
                for (const ie of arr) { const g = iStart - ie.predicted_end_time; if (g >= 0 && g < bg) { b = ie; bg = g; } }
                return b;
              };
              const rawNW = timelines[gid].entries.filter(e => !e.is_init);
              const actualFSR = rawNW.filter(e => !e.is_predicted);
              const pbc: Record<string, typeof rawNW> = {};
              for (const e of rawNW.filter(e => e.is_predicted)) (pbc[e.component] ??= []).push(e);
              const collapsed: typeof rawNW = [];
              for (const [comp, grp] of Object.entries(pbc)) {
                const s = { ...grp[0] };
                s.task_id = `__pred_summary_${comp}_${gid}`;
                s.start_time = Math.min(...grp.map(e => e.start_time));
                s.predicted_end_time = Math.max(...grp.map(e => effectiveEnd(e)));
                collapsed.push(s as any);
              }
              const inferIdsE = new Set(rawNW.map(e => e.task_id));
              const standaloneInitsE = timelines[gid].entries.filter(e => {
                if (!e.is_init) return false;
                const rid = e.task_id.replace(/__init_phase$/, "");
                return !inferIdsE.has(rid);
              });
              const nw = [...actualFSR, ...collapsed, ...standaloneInitsE]
                .sort((a, b) => a.start_time - b.start_time);
              const sre: number[] = [];
              const sreComp: (string | undefined)[] = [];
              for (const e of nw) {
                const wu = e.is_init ? undefined : findInitE(e.task_id, e.start_time);
                const es = wu ? Math.min(wu.start_time, e.start_time) : e.start_time;
                const fe = Math.max(effectiveEnd(e), e.predicted_end_time);
                const eDur = Math.max(fe - es, MIN_BAR_PX / EXPORT_PX_PER_SEC);
                const eEnd = es + eDur;
                let assigned = -1;
                for (let r = 0; r < sre.length; r++) {
                  if (sre[r] <= es && sreComp[r] === e.component) {
                    assigned = r; sre[r] = eEnd; break;
                  }
                }
                if (assigned < 0) {
                  for (let r = 0; r < sre.length; r++) {
                    if (sre[r] <= es) {
                      assigned = r; sre[r] = eEnd; sreComp[r] = e.component; break;
                    }
                  }
                }
                if (assigned < 0) {
                  assigned = sre.length; sre.push(eEnd); sreComp.push(e.component);
                }
                exportSubRows[uid(e)] = assigned;
                if (wu) exportSubRows[uid(wu)] = assigned;
              }
              exportGpuSubRowCounts[gid] = Math.max(1, sre.length);
            }
            const eGpuRowH = (gid: string) => exportGpuSubRowCounts[gid] * (SUB_ROW_H + SUB_GAP) + 10;
            const eGpuRowY: Record<string, number> = {};
            let eCurY = PAD.top;
            for (const gid of gpuIds) { eGpuRowY[gid] = eCurY; eCurY += eGpuRowH(gid); }
            const exportH = eCurY + PAD.bottom;

            const eDur = exportTEnd - exportTStart;
            const eTargetTicks = Math.max(5, Math.floor(exportPlotW / 80));
            const eRawInt = eDur / eTargetTicks;
            const eNice = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
            const eTickInt = eNice.find(n => n >= eRawInt) || Math.ceil(eRawInt / 60) * 60;
            const eTicks: number[] = [];
            let eTick = Math.ceil(exportTStart / eTickInt) * eTickInt;
            while (eTick <= exportTEnd) { eTicks.push(eTick); eTick += eTickInt; }

            const legendItems = [...new Set(allEntries.map(e => e.component))].sort();
            const barH = SUB_ROW_H;
            let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${exportW}" height="${exportH}" style="background:#fafcfd">`;
            svg += `<defs>`;
            svg += `<pattern id="pat-backfill" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#fff" stroke-width="2" opacity="0.45"/></pattern>`;
            svg += `<pattern id="pat-stale" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="#ff9800" stroke-width="1.5" opacity="0.6"/></pattern>`;
            svg += `<pattern id="pat-init" patternUnits="userSpaceOnUse" width="6" height="6"><line x1="0" y1="3" x2="6" y2="3" stroke="#fff" stroke-width="1.5" opacity="0.5"/></pattern>`;
            svg += `<pattern id="pat-killed" patternUnits="userSpaceOnUse" width="8" height="8"><line x1="0" y1="0" x2="8" y2="8" stroke="#e15759" stroke-width="1.5" opacity="0.7"/><line x1="8" y1="0" x2="0" y2="8" stroke="#e15759" stroke-width="1.5" opacity="0.7"/></pattern>`;
            svg += `</defs>`;

            for (const t of eTicks) {
              const x = toExportX(t);
              const label = `${((t - exportTStart) | 0)}s`;
              svg += `<line x1="${x}" y1="${PAD.top - 4}" x2="${x}" y2="${exportH - PAD.bottom}" stroke="#e0e4e8" stroke-width="0.5"/>`;
              svg += `<text x="${x}" y="${PAD.top - 8}" text-anchor="middle" font-size="9" fill="#999">${label}</text>`;
            }

            for (const gid of gpuIds) {
              const rowY = eGpuRowY[gid];
              const rowH = eGpuRowH(gid);
              svg += `<rect x="0" y="${rowY}" width="${exportW}" height="${rowH}" fill="${gpuIds.indexOf(gid) % 2 === 0 ? '#f4f7fa' : '#fafcfd'}"/>`;
              svg += `<text x="4" y="${rowY + 14}" font-size="10" fill="#888" font-weight="600">GPU ${gid}</text>`;

              const initByTaskR: Record<string, Array<typeof allEntries[0]>> = {};
              for (const e of timelines[gid].entries) {
                if (e.is_init) {
                  const rid = e.task_id.replace(/__init_phase$/, "");
                  (initByTaskR[rid] ??= []).push(e);
                }
              }
              const findInitRE = (tid: string, iStart: number) => {
                const arr = initByTaskR[tid];
                if (!arr || !arr.length) return undefined;
                if (arr.length === 1) {
                  const g = iStart - arr[0].predicted_end_time;
                  return g >= 0 && g < 60 ? arr[0] : undefined;
                }
                let b: typeof allEntries[0] | undefined; let bg = Infinity;
                for (const ie of arr) { const g = iStart - ie.predicted_end_time; if (g >= 0 && g < bg) { b = ie; bg = g; } }
                return b;
              };

              const rawNW = timelines[gid].entries.filter(e => !e.is_init);
              const actualEntries = rawNW.filter(e => !e.is_predicted);
              const pbc: Record<string, typeof rawNW> = {};
              for (const e of rawNW.filter(e => e.is_predicted)) (pbc[e.component] ??= []).push(e);
              const collapsedEntries: any[] = [];
              for (const [comp, grp] of Object.entries(pbc)) {
                const s = { ...grp[0], _predicted_count: grp.length };
                s.task_id = `__pred_summary_${comp}_${gid}`;
                s.start_time = Math.min(...grp.map(e => e.start_time));
                s.predicted_end_time = Math.max(...grp.map(e => effectiveEnd(e)));
                collapsedEntries.push(s);
              }

              for (const e of [...actualEntries, ...collapsedEntries]) {
                const subRow = exportSubRows[uid(e)] ?? 0;
                const barY = rowY + 4 + subRow * (SUB_ROW_H + SUB_GAP);
                const color = compColor(e.component);
                const isPredicted = e.is_predicted;
                const isDone = e.completed;
                const isKilled = e.is_killed;
                const isStale = e.prediction_stale;
                const isBackfill = e.is_backfill;
                const initEntry = findInitRE(e.task_id, e.start_time);

                const eEnd = effectiveEnd(e);
                const rawW = (eEnd - e.start_time) * EXPORT_PX_PER_SEC;
                const bW = Math.max(MIN_BAR_PX, rawW);
                const bX = toExportX(e.start_time);

                const strokeDash = isPredicted ? "4 3" : "none";
                const strokeColor = isKilled ? "#e15759" : isDone ? "#999" : color;
                const strokeW = isKilled ? 2 : isDone ? 1 : isPredicted ? 1.5 : 0.5;
                const barOpacity = isKilled ? 0.35 : 1;

                if (initEntry) {
                  const wuX = toExportX(initEntry.start_time);
                  const wuW = Math.max(MIN_BAR_PX, (initEntry.predicted_end_time - initEntry.start_time) * EXPORT_PX_PER_SEC);
                  svg += `<rect x="${wuX}" y="${barY}" width="${wuW}" height="${barH}" rx="3" fill="${color}" opacity="0.45"/>`;
                  svg += `<rect x="${wuX}" y="${barY}" width="${wuW}" height="${barH}" rx="3" fill="url(#pat-init)"/>`;
                  svg += `<line x1="${bX}" y1="${barY}" x2="${bX}" y2="${barY + barH}" stroke="#333" stroke-width="1.5"/>`;
                }

                if (!isDone && !isPredicted && e.predicted_end_time > e.start_time + e.elapsed_sec) {
                  const ghostStart = bX + bW;
                  const ghostEnd = toExportX(e.predicted_end_time);
                  const ghostW = Math.max(0, ghostEnd - ghostStart);
                  if (ghostW > 2) {
                    svg += `<rect x="${ghostStart}" y="${barY}" width="${ghostW}" height="${barH}" rx="3" fill="${color}" opacity="0.2"/>`;
                  }
                }

                svg += `<rect x="${bX}" y="${barY}" width="${bW}" height="${barH}" rx="3" fill="${color}" opacity="${barOpacity}" stroke="${strokeColor}" stroke-width="${strokeW}" stroke-dasharray="${strokeDash}"/>`;

                if (isBackfill) svg += `<rect x="${bX}" y="${barY}" width="${bW}" height="${barH}" rx="3" fill="url(#pat-backfill)"/>`;
                if (isStale && !isDone) svg += `<rect x="${bX}" y="${barY}" width="${bW}" height="${barH}" rx="3" fill="url(#pat-stale)"/>`;
                if (isKilled) svg += `<rect x="${bX}" y="${barY}" width="${bW}" height="${barH}" rx="3" fill="url(#pat-killed)"/>`;
                if (isDone && !isKilled && bW > 6) svg += `<text x="${bX + bW - 2}" y="${barY + barH / 2 + 1}" text-anchor="end" dominant-baseline="middle" font-size="9" fill="#fff" font-weight="700">✓</text>`;
                if (isKilled && bW > 6) svg += `<text x="${bX + bW - 2}" y="${barY + barH / 2 + 1}" text-anchor="end" dominant-baseline="middle" font-size="9" fill="#e15759" font-weight="700">✗</text>`;
                if (isPredicted && (e as any)._predicted_count > 1 && bW > 16) {
                  svg += `<text x="${bX + 4}" y="${barY + barH / 2 + 1}" dominant-baseline="middle" font-size="8" fill="#fff" font-weight="700">×${(e as any)._predicted_count}</text>`;
                }

                if (bW > 40) {
                  svg += `<text x="${bX + (isPredicted && (e as any)._predicted_count > 1 ? 20 : 4)}" y="${barY + barH / 2 + 1}" dominant-baseline="middle" font-size="8" fill="#fff" opacity="0.9">${e.component}</text>`;
                }
              }
            }
            svg += `</svg>`;

            const svgBlob = new Blob([svg], { type: "image/svg+xml;charset=utf-8" });
            const svgUrl = URL.createObjectURL(svgBlob);
            const img = new Image();
            img.onload = () => {
              const legendH = 30;
              const totalH = exportH + legendH;
              const canvas = document.createElement("canvas");
              const scale = 2;
              canvas.width = exportW * scale;
              canvas.height = totalH * scale;
              const ctx = canvas.getContext("2d")!;
              ctx.scale(scale, scale);
              ctx.fillStyle = "#fafcfd";
              ctx.fillRect(0, 0, exportW, totalH);
              ctx.drawImage(img, 0, 0, exportW, exportH);
              ctx.fillStyle = "#f0f2f4";
              ctx.fillRect(0, exportH, exportW, legendH);
              let lx = 12;
              ctx.font = "11px sans-serif";
              for (const comp of legendItems) {
                const c = compColor(comp);
                ctx.fillStyle = c;
                ctx.fillRect(lx, exportH + 9, 12, 12);
                ctx.fillStyle = "#555";
                ctx.fillText(comp, lx + 16, exportH + 19);
                lx += ctx.measureText(comp).width + 32;
              }
              canvas.toBlob((blob) => {
                if (!blob) return;
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `gpu_timeline_${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
                a.click();
                URL.revokeObjectURL(url);
              }, "image/png");
              URL.revokeObjectURL(svgUrl);
            };
            img.src = svgUrl;
          }}
          style={{
            fontSize: 10, padding: "1px 6px", cursor: "pointer",
            border: "1px solid #4a90d9", borderRadius: 3,
            background: "#fff", color: "#4a90d9",
          }}
        >Export PNG</button>
      </div>
      <svg ref={svgRef} width={W} height={H} style={{ background: "#fafcfd", border: "1px solid #e7edef", borderRadius: "6px 6px 0 0", display: "block" }}>
        <defs>
          {/* Backfill: white diagonal stripes */}
          <pattern id="pat-backfill" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="#fff" strokeWidth={2} opacity={0.45} />
          </pattern>
          {/* Uncertain/stale: orange diagonal */}
          <pattern id="pat-stale" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="6" stroke="#ff9800" strokeWidth={1.5} opacity={0.6} />
          </pattern>
          {/* Init: horizontal stripes (white) — visually distinct from backfill's diagonal */}
          <pattern id="pat-init" patternUnits="userSpaceOnUse" width="4" height="4">
            <line x1="0" y1="2" x2="4" y2="2" stroke="#fff" strokeWidth={2} opacity={0.55} />
          </pattern>
          {/* Killed: red X pattern */}
          <pattern id="pat-killed" patternUnits="userSpaceOnUse" width="8" height="8">
            <line x1="0" y1="0" x2="8" y2="8" stroke="#e15759" strokeWidth={1.5} opacity={0.7} />
            <line x1="8" y1="0" x2="0" y2="8" stroke="#e15759" strokeWidth={1.5} opacity={0.7} />
          </pattern>
        </defs>

        {/* Grid */}
        {ticks.map(t => (
          <line key={t} x1={toX(t)} y1={PAD.top - 4} x2={toX(t)} y2={H - PAD.bottom} stroke="#eee" strokeWidth={1} />
        ))}
        {/* Now line */}
        <line x1={nowX} y1={PAD.top - 4} x2={nowX} y2={H - PAD.bottom}
          stroke="var(--danger)" strokeWidth={1.5} strokeDasharray="4 2" />
        <text x={nowX} y={PAD.top - 8} textAnchor="middle" fontSize={9} fill="var(--danger)">now</text>

        {/* Tick labels */}
        {ticks.map(t => {
          const rel = t - now;
          const label = Math.abs(rel) < 0.5 ? "0" : rel > 0 ? `+${rel.toFixed(0)}s` : `${rel.toFixed(0)}s`;
          return <text key={`l-${t}`} x={toX(t)} y={H - PAD.bottom + 14} textAnchor="middle" fontSize={9} fill="#999">{label}</text>;
        })}

        {/* GPU rows */}
        {gpuIds.map((gpuId, rowIdx) => {
          const y = gpuRowY[gpuId];
          const rowH = gpuRowHeight(gpuId);
          const tl = timelines[gpuId];
          const vramPct = tl.total_vram_mb > 0 ? tl.current_reserved_mb / tl.total_vram_mb : 0;
          return (
            <g key={gpuId}>
              <rect x={0} y={y} width={W} height={rowH} fill={rowIdx % 2 === 0 ? "#fff" : "#f8f9fa"} />
              <line x1={LABEL_W} y1={y + rowH} x2={W} y2={y + rowH} stroke="#eee" strokeWidth={0.5} />
              <text x={4} y={y + rowH / 2 + 1} dominantBaseline="middle" fontSize={11} fill="#333" fontWeight={600}>
                GPU {gpuId}
              </text>
              {/* VRAM mini bar */}
              <rect x={LABEL_W - 42} y={y + rowH / 2 + 8} width={36} height={4} rx={2} fill="#eee" />
              <rect x={LABEL_W - 42} y={y + rowH / 2 + 8} width={36 * Math.min(1, vramPct)} height={4} rx={2}
                fill={vramPct > 0.8 ? "var(--danger)" : "var(--accent)"} />

              {/* Task bars — initEntry entries rendered as left-extension of their inference bar */}
              {(() => {
                const initMapArr: Record<string, Array<typeof tl.entries[0]>> = {};
                const rawInference: typeof tl.entries = [];
                for (const e of tl.entries) {
                  if (e.is_init) {
                    const realId = e.task_id.replace(/__init_phase$/, "");
                    (initMapArr[realId] ??= []).push(e);
                  } else {
                    rawInference.push(e);
                  }
                }
                const findInitR = (taskId: string, inferStart: number) => {
                  const inits = initMapArr[taskId];
                  if (!inits || inits.length === 0) return undefined;
                  if (inits.length === 1) {
                    const gap = inferStart - inits[0].predicted_end_time;
                    return gap >= 0 && gap < 60 ? inits[0] : undefined;
                  }
                  let best: typeof tl.entries[0] | undefined;
                  let bestGap = Infinity;
                  for (const ie of inits) {
                    const gap = inferStart - ie.predicted_end_time;
                    if (gap >= 0 && gap < bestGap) { best = ie; bestGap = gap; }
                  }
                  return best;
                };
                const actualEntries = rawInference.filter(e => !e.is_predicted);
                const predictedByComp: Record<string, typeof rawInference> = {};
                for (const e of rawInference.filter(e => e.is_predicted)) {
                  (predictedByComp[e.component] ??= []).push(e);
                }
                const collapsedPredicted: typeof rawInference = [];
                for (const [comp, group] of Object.entries(predictedByComp)) {
                  const earliest = Math.min(...group.map(e => e.start_time));
                  const latestEnd = Math.max(...group.map(e => effectiveEnd(e)));
                  const summary = { ...group[0] };
                  summary.task_id = `__pred_summary_${comp}_${gpuId}`;
                  summary.start_time = earliest;
                  summary.predicted_end_time = latestEnd;
                  (summary as any)._predicted_count = group.length;
                  collapsedPredicted.push(summary as any);
                }
                const inferenceEntries = [...actualEntries, ...collapsedPredicted];
                const renderedInitIds = new Set<string>();

                const inferenceElements = inferenceEntries.map(e => {
                  const subRow = entrySubRows[uid(e)] ?? 0;
                  const initEntry = findInitR(e.task_id, e.start_time);
                  if (initEntry) renderedInitIds.add(e.task_id);

                  const initX1 = initEntry ? Math.max(toX(initEntry.start_time), LABEL_W) : 0;
                  const inferX1 = Math.max(toX(e.start_time), LABEL_W);
                  const x2 = toX(effectiveEnd(e));
                  const rawW = x2 - inferX1;
                  const barW = Math.max(rawW, MIN_BAR_PX);
                  const initW = initEntry ? Math.max(MIN_BAR_PX, inferX1 - initX1) : 0;
                  const barY = y + 4 + subRow * (SUB_ROW_H + SUB_GAP);
                  const barH = SUB_ROW_H;
                  const color = compColor(e.component);
                  const isDone = e.completed;
                  const isKilled = e.is_killed;
                  const isPredicted = e.is_predicted;
                  const isBackfill = e.is_backfill;
                  const isStale = e.prediction_stale;

                  const strokeDash = isPredicted ? "4 3" : "none";
                  const strokeColor = isKilled ? "#e15759" : isDone ? "#999" : isPredicted ? color : color;
                  const strokeW = isKilled ? 2 : isDone ? 1 : isPredicted ? 1.5 : 0.5;
                  const fillOpacity = isKilled ? 0.35 : 1;

                  return (
                    <g key={uid(e)}>
                      {/* Init: left extension with horizontal stripe pattern */}
                      {initEntry && initW > 0 ? (
                        <>
                          <rect x={initX1} y={barY} width={initW} height={barH}
                            rx={3} fill={color} opacity={0.45} />
                          <rect x={initX1} y={barY} width={initW} height={barH}
                            rx={3} fill="url(#pat-init)" />
                          {/* Separator line between initEntry and inference */}
                          <line x1={inferX1} y1={barY} x2={inferX1} y2={barY + barH}
                            stroke="#333" strokeWidth={1.5} />
                          <title>
                            {`${e.component} init: ${(initEntry.predicted_end_time - initEntry.start_time).toFixed(1)}s`}
                          </title>
                        </>
                      ) : null}
                      {/* Predicted latency ghost — shows expected remaining time for running tasks */}
                      {!isDone && !isPredicted && e.predicted_end_time > e.start_time + e.elapsed_sec ? (() => {
                        const ghostStart = inferX1 + barW;
                        const ghostEnd = toX(e.predicted_end_time);
                        const ghostW = Math.max(0, ghostEnd - ghostStart);
                        return ghostW > 2 ? (
                          <rect x={ghostStart} y={barY} width={ghostW} height={barH} rx={3}
                            fill={color} opacity={0.2} />
                        ) : null;
                      })() : null}
                      {/* Inference bar */}
                      <rect x={inferX1} y={barY} width={barW} height={barH} rx={3}
                        fill={color} opacity={fillOpacity}
                        stroke={strokeColor} strokeWidth={strokeW}
                        strokeDasharray={strokeDash} />
                      {/* Backfill overlay */}
                      {isBackfill ? (
                        <rect x={inferX1} y={barY} width={barW} height={barH} rx={3}
                          fill="url(#pat-backfill)" />
                      ) : null}
                      {/* Stale/uncertain overlay */}
                      {isStale && !isDone ? (
                        <rect x={inferX1} y={barY} width={barW} height={barH} rx={3}
                          fill="url(#pat-stale)" />
                      ) : null}
                      {/* Killed overlay */}
                      {isKilled ? (
                        <rect x={inferX1} y={barY} width={barW} height={barH} rx={3}
                          fill="url(#pat-killed)" />
                      ) : null}
                      {/* Done marker */}
                      {isDone && !isKilled && barW > 6 ? (
                        <text x={inferX1 + barW - 2} y={barY + barH / 2 + 1} textAnchor="end"
                          dominantBaseline="middle" fontSize={9} fill="#fff" fontWeight={700}>✓</text>
                      ) : null}
                      {/* Killed marker */}
                      {isKilled && barW > 6 ? (
                        <text x={inferX1 + barW - 2} y={barY + barH / 2 + 1} textAnchor="end"
                          dominantBaseline="middle" fontSize={9} fill="#e15759" fontWeight={700}>✗</text>
                      ) : null}
                      {/* Predicted count badge */}
                      {isPredicted && (e as any)._predicted_count > 1 && barW > 16 ? (
                        <text x={inferX1 + 4} y={barY + barH / 2 + 1} dominantBaseline="middle"
                          fontSize={8} fill="#fff" fontWeight={700}>×{(e as any)._predicted_count}</text>
                      ) : null}
                      {/* Tooltip */}
                      <title>
{e.component} ({isKilled ? "killed" : isPredicted ? "predicted" : isBackfill ? "backfill" : "primary"})
{isPredicted ? "📍 Predicted placement" : ""}
{initEntry ? `Init: ${(initEntry.predicted_end_time - initEntry.start_time).toFixed(1)}s + ` : ""}
{isKilled ? `Killed at ${e.elapsed_sec.toFixed(1)}s (worker guard-killed)` : isDone ? `Completed in ${e.elapsed_sec.toFixed(1)}s` : isPredicted ? `Predicted: ${(e.predicted_end_time - e.start_time).toFixed(1)}s` : `Elapsed: ${e.elapsed_sec.toFixed(1)}s / Remaining: ${e.remaining_sec.toFixed(1)}s`}
VRAM: {e.predicted_vram_mb.toFixed(0)} MB
{isStale ? "⚠ uncertain" : ""}
                      </title>
                    </g>
                  );
                });

                const orphanInits: typeof tl.entries = [];
                for (const [realId, inits] of Object.entries(initMapArr)) {
                  if (!renderedInitIds.has(realId)) {
                    orphanInits.push(...inits);
                  }
                }

                return [
                  ...inferenceElements,
                  ...orphanInits.map(ie => {
                    const subRow = entrySubRows[uid(ie)] ?? (gpuSubRowCounts[gpuId] - 1);
                    const x1 = Math.max(toX(ie.start_time), LABEL_W);
                    const completedAt = (ie as any).completed_at as number | null | undefined;
                    const killedAt = (ie as any).killed_at as number | null | undefined;
                    const terminalAt = completedAt ?? killedAt;
                    const isInflight = (terminalAt == null) && ie.start_time < now;
                    const effectiveEndSec = terminalAt != null
                      ? Math.max(ie.predicted_end_time, terminalAt)
                      : (isInflight ? Math.max(ie.predicted_end_time, now) : ie.predicted_end_time);
                    const x2 = toX(effectiveEndSec);
                    const barW = Math.max(x2 - x1, MIN_BAR_PX);
                    const barY = y + 4 + subRow * (SUB_ROW_H + SUB_GAP);
                    const barH = SUB_ROW_H;
                    const color = compColor(ie.component);
                    const actualDur = effectiveEndSec - ie.start_time;
                    return (
                      <g key={`orphan-init-${ie.task_id}`}>
                        <rect x={x1} y={barY} width={barW} height={barH}
                          rx={3} fill={color} opacity={0.35} />
                        <rect x={x1} y={barY} width={barW} height={barH}
                          rx={3} fill="url(#pat-init)" />
                        {ie.prediction_stale ? (
                          <rect x={x1} y={barY} width={barW} height={barH}
                            rx={3} fill="url(#pat-stale)" />
                        ) : null}
                        {killedAt != null ? (
                          <rect x={x1} y={barY} width={barW} height={barH}
                            rx={3} fill="url(#pat-killed)" />
                        ) : null}
                        <title>
{`${ie.component} pre-init${isInflight ? " (in-flight)" : killedAt != null ? " (killed)" : completedAt != null ? " (completed)" : " (projected)"}
Duration: ${actualDur.toFixed(1)}s${terminalAt != null && terminalAt > ie.predicted_end_time ? ` (predicted ${(ie.predicted_end_time - ie.start_time).toFixed(1)}s)` : ""}
VRAM: ${ie.predicted_vram_mb.toFixed(0)} MB
${ie.prediction_stale ? "⚠ uncertain" : ""}`}
                        </title>
                      </g>
                    );
                  }),
                ];
              })()}
            </g>
          );
        })}
      </svg>
      {/* Legend — HTML flexbox for natural wrapping */}
      <div ref={legendRef} style={{
        display: "flex", flexWrap: "wrap", gap: "6px 14px", padding: "6px 8px",
        background: "#fafcfd", border: "1px solid #e7edef", borderTop: "none",
        borderRadius: "0 0 6px 6px", fontSize: 11, color: "#555",
      }}>
        {comps.map(comp => (
          <span key={comp} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, background: compColor(comp), display: "inline-block" }} />
            {comp}
          </span>
        ))}
        {hasInit ? (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <svg width={12} height={12}><rect width={12} height={12} rx={2} fill="#888" /><rect width={12} height={12} rx={2} fill="url(#pat-init)" /></svg>
            init
          </span>
        ) : null}
        {hasBackfill ? (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <svg width={12} height={12}><rect width={12} height={12} rx={2} fill="#888" /><rect width={12} height={12} rx={2} fill="url(#pat-backfill)" /></svg>
            backfill
          </span>
        ) : null}
        {hasDone ? (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, background: "#888", border: "1.5px dotted #666", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: 8, color: "#fff", fontWeight: 700 }}>✓</span>
            done
          </span>
        ) : null}
        {hasPredicted ? (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, background: "#888", border: "1.5px dashed #555", display: "inline-block" }} />
            predicted
          </span>
        ) : null}
        {hasStale ? (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <svg width={12} height={12}><rect width={12} height={12} rx={2} fill="url(#pat-stale)" stroke="#ff9800" strokeWidth={1} /></svg>
            uncertain
          </span>
        ) : null}
        {hasKilled ? (
          <span style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 12, height: 12, borderRadius: 2, background: "rgba(225,87,89,0.3)", border: "2px solid #e15759", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: 8, color: "#e15759", fontWeight: 700 }}>✗</span>
            killed
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function SignalPanel({ signals, loading, error }: SignalPanelProps) {
  const data = signals?.signals;

  return (
    <section className="panel">
      <div className="section-header">
        <div>
          <h2>Signal Service</h2>
          <p className="small-muted">Real-time resource profiling, interference modeling, and latency tracking.</p>
        </div>
        {loading && !data ? <span className="small-muted">Loading...</span> : null}
      </div>

      {error ? <div className="error-banner">{error}</div> : null}
      {!data ? null : (
        <>
          {/* Campaign Scheduler / Planner Scenario */}
          {data.campaign_scheduler ? (
            <div className="signal-section">
              <h3>Planner — Scheduling Scenario</h3>
              <p className="small-muted">
                Campaign priority queue and GPU timeline projections.
                Primary campaign: <strong>{data.campaign_scheduler.primary_campaign || "none"}</strong>
                {" "}· {data.campaign_scheduler.campaign_count} campaigns
              </p>

              {/* Campaign Queue */}
              {Object.keys(data.campaign_scheduler.campaigns).length > 0 ? (
                <div className="table-wrap table-wrap--bounded">
                  <table className="compact-table">
                    <thead>
                      <tr>
                        <th>campaign</th>
                        <th>role</th>
                        <th>pending</th>
                        <th>active</th>
                        <th>completed</th>
                        <th title="Plan WSJF remaining_est — predicted residual runtime (Σ μ_lat × waves across not-yet-complete DAG stages).  FIFO fallback when blank.">eta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(data.campaign_scheduler.campaigns).map(([cid, cq]) => {
                        const failed = (cq.failed_tasks ?? 0) > 0;
                        const live = cq.pending_tasks + cq.active_tasks;
                        const done = cq.is_dag_complete === true
                          ? !failed
                          : (cq.is_dag_complete === undefined
                              && live <= 0
                              && !failed
                              && (cq.succeeded_tasks ?? cq.completed_tasks) > 0);
                        const isPrimary = cid === data.campaign_scheduler!.primary_campaign;
                        let roleBadge: JSX.Element;
                        if (failed) {
                          roleBadge = <span className="badge badge--danger">failed</span>;
                        } else if (done) {
                          roleBadge = <span className="badge badge--success">done</span>;
                        } else if (isPrimary) {
                          roleBadge = <span className="badge badge--accent">primary</span>;
                        } else {
                          roleBadge = <span className="badge badge--neutral">backfill</span>;
                        }
                        const remaining = cq.remaining_est_sec;
                        let etaCell: JSX.Element;
                        if (remaining == null || !isFinite(remaining) || remaining <= 0) {
                          etaCell = (
                            <span className="small-muted" title="no GP observation cluster-wide → FIFO fallback">—</span>
                          );
                        } else {
                          const total = Math.round(remaining);
                          const h = Math.floor(total / 3600);
                          const m = Math.floor((total % 3600) / 60);
                          const s = total % 60;
                          const label = h > 0
                            ? `${h}h ${m}m`
                            : m > 0
                              ? `${m}m ${s}s`
                              : `${s}s`;
                          etaCell = (
                            <span title={`remaining_est = ${remaining.toFixed(1)}s (Plan)`}>{label}</span>
                          );
                        }
                        return (
                          <tr key={cid}>
                            <td title={cid}>{cid.length > 12 ? `${cid.slice(0, 12)}…` : cid}</td>
                            <td>{roleBadge}</td>
                            <td>{cq.pending_tasks}</td>
                            <td>{cq.active_tasks}</td>
                            <td>{cq.completed_tasks}</td>
                            <td>{etaCell}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : null}

              {/* GPU Timelines — Gantt-style predictive view */}
              {Object.keys(data.campaign_scheduler.gpu_timelines).length > 0 ? (
                <GpuTimelineGantt timelines={data.campaign_scheduler.gpu_timelines} schedulerData={data.campaign_scheduler} />
              ) : null}
            </div>
          ) : null}

          {/* Scheduling Scenario — GPU Resource Views + Component Envelopes */}
          {data.scheduling_scenario ? (
            <div className="signal-section">
              <h3>Scheduling Scenario</h3>
              <p className="small-muted">
                System-wide resource snapshot consulted before each placement decision.
                {data.scheduling_scenario.n_stale_envelopes > 0
                  ? ` ⚠ ${data.scheduling_scenario.n_stale_envelopes} stale envelope(s)`
                  : null}
              </p>

              {/* GPU Resource Views */}
              {Object.keys(data.scheduling_scenario.gpu_views).length > 0 ? (
                <div style={{ marginBottom: 12 }}>
                  <h4 style={{ fontSize: 13, margin: "0 0 6px" }}>GPU Resource Views</h4>
                  <div className="signal-cards">
                    {Object.entries(data.scheduling_scenario.gpu_views).map(([gpuId, gv]) => {
                      const usedPct = gv.total_vram_mib > 0 ? (gv.reserved_vram_mib / gv.total_vram_mib) * 100 : 0;
                      return (
                        <div key={gpuId} className="signal-card" style={{ minWidth: 200 }}>
                          <h4 style={{ fontSize: 12 }}>GPU {gpuId}</h4>
                          <div className="small-muted">
                            {gv.reserved_vram_mib.toFixed(0)} / {gv.total_vram_mib.toFixed(0)} MiB
                            {" "}· {gv.available_vram_mib.toFixed(0)} free
                            {gv.utilization_pct > 0 ? ` · ${gv.utilization_pct.toFixed(0)}% util` : null}
                          </div>
                          <div style={{ background: "#eee", height: 8, borderRadius: 4, overflow: "hidden", margin: "4px 0" }}>
                            <div style={{
                              width: `${Math.min(100, usedPct)}%`,
                              height: "100%",
                              background: usedPct > 80 ? "var(--danger)" : usedPct > 50 ? "var(--warning)" : "var(--success)",
                              borderRadius: 4,
                            }} />
                          </div>
                          {gv.inflight_components.length > 0 ? (
                            <div style={{ fontSize: 11 }}>
                              {gv.inflight_components.map((c, i) => (
                                <span key={i} className="badge badge--neutral" style={{ marginRight: 3, fontSize: 10 }}>{c}</span>
                              ))}
                            </div>
                          ) : <span className="small-muted" style={{ fontSize: 11 }}>idle</span>}
                          {gv.projected_clear_sec > 0 ? (
                            <div className="small-muted" style={{ fontSize: 10, marginTop: 2 }}>
                              clears in ~{gv.projected_clear_sec.toFixed(0)}s
                            </div>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}

              {/* Co-location Map */}
              {Object.keys(data.scheduling_scenario.colocation_map).length > 0 ? (
                <div style={{ marginBottom: 12 }}>
                  <h4 style={{ fontSize: 13, margin: "0 0 6px" }}>Co-location Map</h4>
                  <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontSize: 12 }}>
                    {Object.entries(data.scheduling_scenario.colocation_map).map(([gpuId, comps]) => (
                      <div key={gpuId}>
                        <strong>GPU {gpuId}:</strong>{" "}
                        {comps.join(" + ")}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {/* Component Envelopes */}
              {Object.keys(data.scheduling_scenario.component_envelopes).length > 0 ? (
                <Collapsible title="Component Envelopes" badge={`${Object.keys(data.scheduling_scenario.component_envelopes).length}`}>
                  <div className="table-wrap table-wrap--bounded">
                    <table className="compact-table">
                      <thead>
                        <tr>
                          <th>component</th>
                          <th>VRAM p50</th>
                          <th>latency p50</th>
                          <th>class</th>
                          <th>confidence</th>
                          <th>n</th>
                          <th>status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(data.scheduling_scenario.component_envelopes).map(([comp, env]) => (
                          <tr key={comp}>
                            <td>{comp}</td>
                            <td>{env.vram_p50_mib > 0 ? `${env.vram_p50_mib.toFixed(0)} MiB` : "—"}</td>
                            <td>{env.latency_p50_sec > 0 ? `${env.latency_p50_sec.toFixed(2)}s` : "—"}</td>
                            <td><WorkloadClassBadge wclass={env.workload_class} /></td>
                            <td>
                              <span className={`badge badge--${env.confidence_label === "high" ? "success" : env.confidence_label === "medium" ? "warning" : "danger"}`}>
                                {env.confidence_label}
                              </span>
                            </td>
                            <td>{env.n_observations}</td>
                            <td>
                              {env.drift_stale
                                ? <span className="badge badge--warning">stale ({env.drift_events_count})</span>
                                : <span className="small-muted">ok</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Collapsible>
              ) : null}
            </div>
          ) : null}

          {/* Initialization Profiles */}
          {data.init_profiles && Object.keys(data.init_profiles).length > 0 ? (
            <div className="signal-section">
              <h3>Init Latency</h3>
              <p className="small-muted">Interference-aware cold-start initialization time per component × GPU.</p>
              <div className="table-wrap">
                <table className="compact-table">
                  <thead><tr><th>component</th><th>GPU</th><th>μ (sec)</th><th>σ (sec)</th><th>n</th><th>confidence</th></tr></thead>
                  <tbody>
                    {Object.entries(data.init_profiles).flatMap(([comp, gpus]) =>
                      Object.entries(gpus).map(([gpuId, w]) => (
                        <tr key={`${comp}-${gpuId}`}>
                          <td>{comp}</td>
                          <td>{gpuId === "*" ? "pooled" : gpuId}</td>
                          <td>{w.mu_sec.toFixed(1)}s</td>
                          <td>{w.sigma_sec.toFixed(1)}s</td>
                          <td>{w.n}</td>
                          <td><ConfidenceBar value={w.confidence} /></td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}

          {/* Resource Profiles */}
          <div className="signal-section">
            <h3>Resource Profiles</h3>
            <p className="small-muted">Per-component learned characteristics with uncertainty. Confidence narrows as observations accumulate.</p>
            {Object.keys(data.resource_profiles || {}).length === 0 ? (
              <p className="small-muted">No resource profiles yet — observations are needed to build profiles.</p>
            ) : (
              <div className="table-wrap table-wrap--bounded">
                <table className="compact-table">
                  <thead>
                    <tr>
                      <th>component</th>
                      <th>workload class</th>
                      <th>gpu_util</th>
                      <th>arith_intensity</th>
                      <th>configs</th>
                      <th>vram_conf</th>
                      <th>latency_conf</th>
                      <th>samples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.resource_profiles).map(([comp, p]) => (
                      <tr key={comp}>
                        <td>{comp}</td>
                        <td><WorkloadClassBadge wclass={p.workload_class} /></td>
                        <td>{p.gpu_util_ema != null ? `${p.gpu_util_ema.toFixed(0)}%` : "-"}</td>
                        <td>{p.arithmetic_intensity != null ? `${fmt(p.arithmetic_intensity, 0)} us/MiB` : "-"}</td>
                        <td>{Object.keys(p.config_baselines).length}</td>
                        <td><ConfidenceBar value={p.vram_confidence} /></td>
                        <td><ConfidenceBar value={p.latency_confidence} /></td>
                        <td>{p.sample_count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Config Baselines detail — collapsible per-GPU */}
          {Object.entries(data.resource_profiles || {}).map(([comp, p]) => {
            const configs = Object.entries(p.config_baselines || {});
            if (configs.length === 0) return null;
            return (
              <div key={comp} className="signal-section signal-section--nested">
                <h3>{comp} — Config Baselines</h3>
                {configs.map(([fp, cp]) => {
                  const gpuEntries = Object.entries(cp.gpu_baselines || {});
                  const pooled = cp.gpu_baselines?.["*"];
                  const perGpuEntries = gpuEntries.filter(([id]) => id !== "*");
                  return (
                    <div key={fp} className="config-baseline-card">
                      <div className="config-baseline-card__header">
                        <span className="config-baseline-card__fp" title={fp}>{fp.length > 20 ? `${fp.slice(0, 20)}…` : fp || "(default)"}</span>
                        <span className="small-muted">{pooled?.total_observations ?? 0} obs · {pooled?.campaigns_observed ?? 0} campaigns</span>
                      </div>
                      {/* Always show pooled (All GPUs) */}
                      {pooled ? (
                        <div className="config-baseline-gpu">
                          <span className="config-baseline-gpu__label badge badge--neutral">All GPUs (pooled)</span>
                          <GpuBaselineDetail gpuId="*" bl={pooled} />
                        </div>
                      ) : null}
                      {/* Per-GPU details — collapsible */}
                      {perGpuEntries.length > 0 ? (
                        <Collapsible title="Per-GPU Baselines" badge={`${perGpuEntries.length} GPUs`}>
                          {perGpuEntries.map(([gpuId, bl]) => (
                            <GpuBaselineDetail key={gpuId} gpuId={gpuId} bl={bl} />
                          ))}
                        </Collapsible>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            );
          })}

          {/* Interference Matrix */}
          <div className="signal-section">
            <h3>Interference Matrix</h3>
            <p className="small-muted">Pairwise slowdown and VRAM overhead between co-located components.</p>
            {Object.keys(data.interference?.pairwise || {}).length === 0 ? (
              <p className="small-muted">No pairwise records yet.</p>
            ) : (
              <div className="table-wrap table-wrap--bounded">
                <table className="compact-table">
                  <thead>
                    <tr>
                      <th>pair</th>
                      <th>slowdown</th>
                      <th>vram_overhead</th>
                      <th>gpu_util_share</th>
                      <th>confidence</th>
                      <th>samples</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.interference.pairwise).map(([pair, rec]) => (
                      <tr key={pair}>
                        <td>{pair}</td>
                        <td>{rec.slowdown_median != null ? `+${(rec.slowdown_median * 100).toFixed(0)}%` : "-"}</td>
                        <td>{rec.vram_overhead_median != null ? `+${(rec.vram_overhead_median * 100).toFixed(0)}%` : "-"}</td>
                        <td>{rec.gpu_util_share_median != null ? `${(rec.gpu_util_share_median * 100).toFixed(0)}%` : "-"}</td>
                        <td><ConfidenceBar value={rec.confidence} /></td>
                        <td>{rec.n_slowdown}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Self-Interference (Plan fix 2-D storage) */}
          <div className="signal-section">
            <h3>Self-Interference</h3>
            <p className="small-muted">
              Intra-worker (same-component) slowdown by ``(input_size, N_concurrency)``.
              Populated by <code>_self_slowdown_obs</code> — the diagonal of the interference
              matrix lives here after  storage migration.
            </p>
            {Object.keys(data.interference?.self_slowdown || {}).length === 0 ? (
              <p className="small-muted">No self-interference observations yet.</p>
            ) : (
              <div className="table-wrap table-wrap--bounded">
                <table className="compact-table">
                  <thead>
                    <tr>
                      <th>component</th>
                      <th>median_delta</th>
                      <th>mean_delta</th>
                      <th>n_valid</th>
                      <th>n_obs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.interference!.self_slowdown || {}).map(([comp, rec]) => (
                      <tr key={comp}>
                        <td>{comp}</td>
                        <td>{`+${(rec.median_delta * 100).toFixed(0)}%`}</td>
                        <td>{`+${(rec.mean_delta * 100).toFixed(0)}%`}</td>
                        <td>{rec.n_valid}</td>
                        <td>{rec.n_obs}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Solo Baselines */}
          <div className="signal-section">
            <h3>Solo Baselines</h3>
            <div className="signal-cards">
              <div className="signal-card">
                <h4>Latency</h4>
                {Object.keys(data.interference?.solo_baselines || {}).length === 0 ? (
                  <p className="small-muted">No baselines.</p>
                ) : (
                  <div className="table-wrap">
                    <table className="compact-table">
                      <thead><tr><th>component</th><th>median</th><th>n</th></tr></thead>
                      <tbody>
                        {Object.entries(data.interference.solo_baselines).map(([comp, b]) => (
                          <tr key={comp}><td>{comp}</td><td>{b.median_sec?.toFixed(2)}s</td><td>{b.n}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
              <div className="signal-card">
                <h4>VRAM</h4>
                {Object.keys(data.interference?.solo_vram || {}).length === 0 ? (
                  <p className="small-muted">No baselines.</p>
                ) : (
                  <div className="table-wrap">
                    <table className="compact-table">
                      <thead><tr><th>component</th><th>median</th><th>n</th></tr></thead>
                      <tbody>
                        {Object.entries(data.interference.solo_vram).map(([comp, b]) => (
                          <tr key={comp}><td>{comp}</td><td>{b.median_mib?.toFixed(0)} MiB</td><td>{b.n}</td></tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Active Latency Trackers */}
          <div className="signal-section">
            <h3>Active Latency Trackers</h3>
            {Object.keys(data.latency_trackers || {}).length === 0 ? (
              <p className="small-muted">No active tasks being tracked.</p>
            ) : (
              <div className="table-wrap table-wrap--bounded">
                <table className="compact-table">
                  <thead>
                    <tr>
                      <th>worker</th>
                      <th>task</th>
                      <th>component</th>
                      <th>elapsed</th>
                      <th>solo</th>
                      <th>co-located</th>
                      <th>max_conc</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.latency_trackers).flatMap(([addr, tracker]) =>
                      tracker.slots.map((slot) => (
                        <tr key={`${addr}:${slot.task_id}`}>
                          <td title={addr}>{addr.length > 20 ? `…${addr.slice(-18)}` : addr}</td>
                          <td title={slot.task_id}>{slot.task_id.length > 10 ? `${slot.task_id.slice(0, 10)}…` : slot.task_id}</td>
                          <td>{slot.component}</td>
                          <td>{slot.elapsed_sec.toFixed(1)}s</td>
                          <td>{slot.currently_solo ? "yes" : "no"}</td>
                          <td>{slot.co_located_at_entry.join(", ") || "-"}</td>
                          <td>{slot.max_concurrent}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Activation Peaks */}
          {Object.keys(data.activation_peaks || {}).length > 0 ? (
            <div className="signal-section">
              <h3>Activation VRAM Peaks</h3>
              <div className="table-wrap table-wrap--bounded">
                <table className="compact-table">
                  <thead><tr><th>key</th><th>peak (MiB)</th></tr></thead>
                  <tbody>
                    {Object.entries(data.activation_peaks).map(([key, val]) => (
                      <tr key={key}><td>{key}</td><td>{val.toFixed(0)}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}
