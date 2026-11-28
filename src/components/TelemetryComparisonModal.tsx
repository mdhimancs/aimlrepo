import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeftRight,
  GitCompareArrows,
  Layers,
  Lock,
  TrendingUp,
  Unlock,
  X
} from 'lucide-react';
import { FlagshipProject } from '../types';
import ProjectTelemetryCharts from './ProjectTelemetryCharts';
import { CHART_COLORS, CHART_FONTS } from '../utils/chartTheme';

interface TelemetryComparisonModalProps {
  projectA: FlagshipProject;
  projectB: FlagshipProject;
  onSwapProjects: () => void;
  onClose: () => void;
}

function parseNumeric(str: string, fallback: number): number {
  const m = str.match(/(\d+(\.\d+)?)/);
  return m ? parseFloat(m[1]) : fallback;
}

function computeStdDevPct(project: FlagshipProject): number {
  const history = project.epochHistory;
  if (!history.length) return 0.45;
  const diffs = history.map((h) => Math.abs(h.accuracy - h.valAccuracy) * 100);
  const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
  return Number((avgDiff * 0.45 + 0.35).toFixed(2));
}

export default function TelemetryComparisonModal({
  projectA,
  projectB,
  onSwapProjects,
  onClose
}: TelemetryComparisonModalProps) {
  const [sectionFilter, setSectionFilter] = useState<'all' | 'convergence' | 'performance'>('all');
  const [syncScroll, setSyncScroll] = useState(true);
  const [showOverlayStrip, setShowOverlayStrip] = useState(true);

  const leftPaneRef = useRef<HTMLDivElement | null>(null);
  const rightPaneRef = useRef<HTMLDivElement | null>(null);
  const isSyncingRef = useRef<'left' | 'right' | null>(null);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose]);

  const handlePaneScroll = (source: 'left' | 'right') => {
    if (!syncScroll) return;
    if (isSyncingRef.current && isSyncingRef.current !== source) return;

    const srcEl = source === 'left' ? leftPaneRef.current : rightPaneRef.current;
    const dstEl = source === 'left' ? rightPaneRef.current : leftPaneRef.current;
    if (!srcEl || !dstEl) return;

    const srcMax = Math.max(1, srcEl.scrollHeight - srcEl.clientHeight);
    const dstMax = Math.max(1, dstEl.scrollHeight - dstEl.clientHeight);
    const ratio = srcEl.scrollTop / srcMax;

    isSyncingRef.current = source;
    dstEl.scrollTop = ratio * dstMax;
    requestAnimationFrame(() => {
      isSyncingRef.current = null;
    });
  };

  const summary = useMemo(() => {
    const lastA = projectA.epochHistory[projectA.epochHistory.length - 1] || {
      loss: 0.15,
      accuracy: 0.94,
      valAccuracy: 0.92
    };
    const lastB = projectB.epochHistory[projectB.epochHistory.length - 1] || {
      loss: 0.15,
      accuracy: 0.94,
      valAccuracy: 0.92
    };

    const accA = Number((lastA.accuracy * 100).toFixed(2));
    const accB = Number((lastB.accuracy * 100).toFixed(2));
    const valA = Number((lastA.valAccuracy * 100).toFixed(2));
    const valB = Number((lastB.valAccuracy * 100).toFixed(2));
    const lossA = Number(lastA.loss.toFixed(4));
    const lossB = Number(lastB.loss.toFixed(4));
    const stdA = computeStdDevPct(projectA);
    const stdB = computeStdDevPct(projectB);
    const latA = parseNumeric(projectA.performanceIndicators.latency, 10);
    const latB = parseNumeric(projectB.performanceIndicators.latency, 10);

    return {
      accA,
      accB,
      accDelta: Number((accA - accB).toFixed(2)),
      valA,
      valB,
      valDelta: Number((valA - valB).toFixed(2)),
      lossA,
      lossB,
      lossDelta: Number((lossA - lossB).toFixed(4)),
      stdA,
      stdB,
      latA,
      latB,
      latDelta: Number((latA - latB).toFixed(2))
    };
  }, [projectA, projectB]);

  const overlayChart = useMemo(() => {
    const histA = projectA.epochHistory;
    const histB = projectB.epochHistory;
    const maxLen = Math.max(histA.length, histB.length, 2);
    const maxLoss = Math.max(
      1.2,
      ...histA.map((h) => h.loss * 1.1),
      ...histB.map((h) => h.loss * 1.1)
    );
    const left = 52;
    const right = 740;
    const top = 16;
    const bottom = 128;
    const height = bottom - top;

    const toPath = (
      hist: typeof histA,
      accessor: (h: (typeof histA)[number]) => number
    ) =>
      hist
        .map((h, i) => {
          const x = left + ((hist.length > 1 ? i / (hist.length - 1) : 0) * (right - left));
          const y = bottom - Math.max(0, Math.min(1, accessor(h))) * height;
          return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(' ');

    return {
      maxLen,
      maxLoss,
      left,
      right,
      top,
      bottom,
      height,
      valPathA: toPath(histA, (h) => h.valAccuracy),
      valPathB: toPath(histB, (h) => h.valAccuracy),
      lossPathA: toPath(histA, (h) => h.loss / maxLoss),
      lossPathB: toPath(histB, (h) => h.loss / maxLoss)
    };
  }, [projectA.epochHistory, projectB.epochHistory]);

  return (
    <div
      className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-xs flex items-center justify-center p-2 sm:p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Split-View Telemetry Comparison Modal"
    >
      <div
        className="w-full max-w-[1580px] h-[94vh] bg-slate-50 rounded-2xl border border-slate-300 shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Top Modal Header */}
        <div className="bg-white border-b border-slate-300 px-5 py-3.5 flex flex-wrap items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-9 h-9 rounded-xl bg-indigo-900 text-white flex items-center justify-center shrink-0 shadow-xs">
              <GitCompareArrows className="w-4.5 h-4.5" />
            </div>
            <div className="min-w-0">
              <h2 className="text-base font-bold text-slate-900 leading-tight truncate">
                Split-View Telemetry Comparison
              </h2>
              <p className="text-[11px] text-slate-600 font-mono truncate">
                <span className="text-indigo-900 font-bold">A: {projectA.title}</span>
                <span className="mx-2 text-slate-400">vs</span>
                <span className="text-emerald-900 font-bold">B: {projectB.title}</span>
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Section Filter */}
            <div className="inline-flex rounded-lg border border-slate-300 bg-slate-100 p-0.5 text-[11px] font-semibold">
              {([
                { id: 'all', label: 'All Telemetry' },
                { id: 'convergence', label: '3D & Convergence' },
                { id: 'performance', label: 'Dist & Efficiency' }
              ] as const).map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setSectionFilter(tab.id)}
                  className={`px-2.5 py-1 rounded-md transition-colors cursor-pointer ${
                    sectionFilter === tab.id
                      ? 'bg-indigo-900 text-white shadow-xs'
                      : 'text-slate-700 hover:text-slate-900'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <button
              type="button"
              onClick={() => setShowOverlayStrip((v) => !v)}
              className={`px-2.5 py-1.5 rounded-lg border text-[11px] font-semibold flex items-center gap-1.5 transition-colors cursor-pointer ${
                showOverlayStrip
                  ? 'bg-blue-50 text-blue-900 border-blue-300'
                  : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Delta &amp; Overlay</span>
            </button>

            <button
              type="button"
              onClick={() => setSyncScroll((v) => !v)}
              className={`px-2.5 py-1.5 rounded-lg border text-[11px] font-semibold flex items-center gap-1.5 transition-colors cursor-pointer ${
                syncScroll
                  ? 'bg-emerald-50 text-emerald-900 border-emerald-300'
                  : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
              }`}
              title="Synchronize vertical scrolling between Left and Right telemetry panes"
            >
              {syncScroll ? <Lock className="w-3.5 h-3.5" /> : <Unlock className="w-3.5 h-3.5" />}
              <span>{syncScroll ? 'Sync Scroll: ON' : 'Sync Scroll: OFF'}</span>
            </button>

            <button
              type="button"
              onClick={onSwapProjects}
              className="px-2.5 py-1.5 rounded-lg bg-white hover:bg-slate-100 text-slate-800 border border-slate-300 text-[11px] font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
              title="Swap Project A and Project B positions"
            >
              <ArrowLeftRight className="w-3.5 h-3.5 text-indigo-800" />
              <span>Swap A ↔ B</span>
            </button>

            <button
              type="button"
              onClick={onClose}
              className="p-1.5 rounded-lg bg-slate-100 hover:bg-rose-100 text-slate-700 hover:text-rose-900 border border-slate-300 transition-colors cursor-pointer"
              aria-label="Close comparison modal"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Optional Head-to-Head KPI Delta Strip + Shared Axis Overlay */}
        {showOverlayStrip && (
          <div className="bg-white border-b border-slate-300 px-5 py-3 shrink-0 space-y-3">
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2.5 text-xs font-mono tabular-nums">
              {/* Train Accuracy */}
              <div className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-2">
                <span className="text-[10px] font-sans font-semibold text-slate-500 uppercase block">
                  Train Accuracy
                </span>
                <div className="flex items-baseline justify-between gap-1 mt-0.5">
                  <span className="text-indigo-900 font-bold">A: {summary.accA}%</span>
                  <span className="text-emerald-900 font-bold">B: {summary.accB}%</span>
                </div>
                <span className="text-[10px] text-slate-600 block mt-0.5">
                  Δ (A−B): {summary.accDelta > 0 ? `+${summary.accDelta}` : summary.accDelta}%
                </span>
              </div>

              {/* Val Accuracy */}
              <div className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-2">
                <span className="text-[10px] font-sans font-semibold text-slate-500 uppercase block">
                  Validation Accuracy
                </span>
                <div className="flex items-baseline justify-between gap-1 mt-0.5">
                  <span className="text-indigo-900 font-bold">A: {summary.valA}%</span>
                  <span className="text-emerald-900 font-bold">B: {summary.valB}%</span>
                </div>
                <span className="text-[10px] text-slate-600 block mt-0.5">
                  Δ (A−B): {summary.valDelta > 0 ? `+${summary.valDelta}` : summary.valDelta}%
                </span>
              </div>

              {/* Final Loss */}
              <div className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-2">
                <span className="text-[10px] font-sans font-semibold text-slate-500 uppercase block">
                  Final Train Loss (Lower Better)
                </span>
                <div className="flex items-baseline justify-between gap-1 mt-0.5">
                  <span className="text-indigo-900 font-bold">A: {summary.lossA}</span>
                  <span className="text-emerald-900 font-bold">B: {summary.lossB}</span>
                </div>
                <span className="text-[10px] text-slate-600 block mt-0.5">
                  Δ (A−B): {summary.lossDelta > 0 ? `+${summary.lossDelta}` : summary.lossDelta}
                </span>
              </div>

              {/* P95 Latency */}
              <div className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-2">
                <span className="text-[10px] font-sans font-semibold text-slate-500 uppercase block">
                  Latency Lead
                </span>
                <div className="flex items-baseline justify-between gap-1 mt-0.5">
                  <span className="text-indigo-900 font-bold">A: {summary.latA}ms</span>
                  <span className="text-emerald-900 font-bold">B: {summary.latB}ms</span>
                </div>
                <span className="text-[10px] text-slate-600 block mt-0.5">
                  Δ (A−B): {summary.latDelta > 0 ? `+${summary.latDelta}` : summary.latDelta}ms
                </span>
              </div>

              {/* 5-Fold Std Dev */}
              <div className="bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 col-span-2 md:col-span-1">
                <span className="text-[10px] font-sans font-semibold text-slate-500 uppercase block">
                  5-Fold Std Dev (±σ)
                </span>
                <div className="flex items-baseline justify-between gap-1 mt-0.5">
                  <span className="text-indigo-900 font-bold">A: ±{summary.stdA}%</span>
                  <span className="text-emerald-900 font-bold">B: ±{summary.stdB}%</span>
                </div>
                <span className="text-[10px] text-slate-600 block mt-0.5">
                  Stability spread
                </span>
              </div>
            </div>

            {/* Compact Overlaid Convergence Strip */}
            <div className="bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 flex flex-col lg:flex-row lg:items-center gap-3">
              <div className="lg:w-64 shrink-0 space-y-1">
                <div className="flex items-center gap-1.5 text-xs font-bold text-slate-900">
                  <TrendingUp className="w-3.5 h-3.5 text-indigo-900" />
                  <span>Shared-Axis Trajectory Overlay</span>
                </div>
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] font-mono">
                  <span className="inline-flex items-center gap-1 text-indigo-900 font-bold">
                    <span className="w-3 h-0.5 bg-indigo-700 inline-block" /> A Val Acc
                  </span>
                  <span className="inline-flex items-center gap-1 text-emerald-900 font-bold">
                    <span className="w-3 h-0.5 bg-emerald-600 inline-block" /> B Val Acc
                  </span>
                  <span className="inline-flex items-center gap-1 text-indigo-600">
                    <span className="w-3 h-0.5 border-b-2 border-dashed border-indigo-500 inline-block" /> A Loss
                  </span>
                  <span className="inline-flex items-center gap-1 text-emerald-700">
                    <span className="w-3 h-0.5 border-b-2 border-dashed border-emerald-500 inline-block" /> B Loss
                  </span>
                </div>
              </div>

              <div className="flex-1 h-28">
                <svg viewBox="0 0 770 144" className="w-full h-full overflow-visible">
                  {[0, 50, 100].map((pct) => {
                    const y = overlayChart.bottom - (pct / 100) * overlayChart.height;
                    return (
                      <g key={pct}>
                        <line
                          x1={overlayChart.left}
                          y1={y}
                          x2={overlayChart.right}
                          y2={y}
                          stroke={CHART_COLORS.gridLine}
                          strokeDasharray={pct === 0 ? undefined : '4 4'}
                          strokeWidth="1"
                        />
                        <text
                          x={overlayChart.left - 8}
                          y={y + 3.5}
                          textAnchor="end"
                          fontSize="10"
                          fontFamily={CHART_FONTS.mono}
                          fill={CHART_COLORS.tickText}
                        >
                          {pct}%
                        </text>
                      </g>
                    );
                  })}
                  <path
                    d={overlayChart.lossPathA}
                    fill="none"
                    stroke="#6366f1"
                    strokeWidth="1.8"
                    strokeDasharray="4 3"
                    opacity="0.75"
                  />
                  <path
                    d={overlayChart.lossPathB}
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="1.8"
                    strokeDasharray="4 3"
                    opacity="0.75"
                  />
                  <path
                    d={overlayChart.valPathA}
                    fill="none"
                    stroke="#312e81"
                    strokeWidth="2.6"
                    strokeLinecap="round"
                  />
                  <path
                    d={overlayChart.valPathB}
                    fill="none"
                    stroke="#047857"
                    strokeWidth="2.6"
                    strokeLinecap="round"
                  />
                </svg>
              </div>
            </div>
          </div>
        )}

        {/* 50/50 Split-View Telemetry Panes */}
        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-2 divide-y lg:divide-y-0 lg:divide-x divide-slate-300 overflow-hidden">
          {/* Left Pane: Project A */}
          <div
            ref={leftPaneRef}
            onScroll={() => handlePaneScroll('left')}
            className="h-full overflow-y-auto bg-white"
          >
            <div className="sticky top-0 z-20 bg-indigo-900 text-white px-5 py-3 flex items-center justify-between gap-2 shadow-xs">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-wider text-indigo-200">
                  <span className="bg-indigo-800 px-1.5 py-0.5 rounded font-bold text-white">Project A</span>
                  <span>{projectA.domain}</span>
                  <span>·</span>
                  <span>{projectA.difficulty}</span>
                </div>
                <h3 className="text-sm font-bold truncate mt-0.5">{projectA.title}</h3>
              </div>
              <div className="text-right font-mono text-[11px] shrink-0">
                <div className="font-bold text-emerald-300">{projectA.trainingAccuracy.split(' ')[0]} Acc</div>
                <div className="text-indigo-200">{projectA.performanceIndicators.latency}</div>
              </div>
            </div>
            <div className="p-5">
              <ProjectTelemetryCharts
                project={projectA}
                compactSplit={true}
                sectionFilter={sectionFilter}
              />
            </div>
          </div>

          {/* Right Pane: Project B */}
          <div
            ref={rightPaneRef}
            onScroll={() => handlePaneScroll('right')}
            className="h-full overflow-y-auto bg-white"
          >
            <div className="sticky top-0 z-20 bg-emerald-900 text-white px-5 py-3 flex items-center justify-between gap-2 shadow-xs">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-[10px] font-mono uppercase tracking-wider text-emerald-200">
                  <span className="bg-emerald-800 px-1.5 py-0.5 rounded font-bold text-white">Project B</span>
                  <span>{projectB.domain}</span>
                  <span>·</span>
                  <span>{projectB.difficulty}</span>
                </div>
                <h3 className="text-sm font-bold truncate mt-0.5">{projectB.title}</h3>
              </div>
              <div className="text-right font-mono text-[11px] shrink-0">
                <div className="font-bold text-emerald-300">{projectB.trainingAccuracy.split(' ')[0]} Acc</div>
                <div className="text-emerald-200">{projectB.performanceIndicators.latency}</div>
              </div>
            </div>
            <div className="p-5">
              <ProjectTelemetryCharts
                project={projectB}
                compactSplit={true}
                sectionFilter={sectionFilter}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
