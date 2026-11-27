import React, { useState } from 'react';
import ThreeDResearchManifold from './ThreeDResearchManifold';
import { FlagshipProject, SyntheticDataType } from '../types';
import {
  ML_TECHNIQUE_META,
  MlTechniqueId,
  SYNTHETIC_DATA_TYPE_META
} from '../utils/syntheticDataset';
import { CHART_COLORS, CHART_FONTS } from '../utils/chartTheme';

interface TrainingLogPoint {
  epoch: number;
  loss: number;
  accuracy: number;
  valAcc: number;
  sigma: number;
}

interface SimulatorMultiModalChartsProps {
  project: FlagshipProject;
  dataType: SyntheticDataType;
  mlTechnique: MlTechniqueId;
  trainingLogs: TrainingLogPoint[];
  projectedValAcc: number;
  tensorShape: string;
  seed: number;
  onSelectMlTechnique: (id: MlTechniqueId) => void;
}

function seededRand(seed: number) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 3D -> 2D perspective/isometric projector with interactive yaw & pitch
function project3D(
  x: number,
  y: number,
  z: number,
  yawDeg: number,
  pitchDeg: number,
  cx: number,
  cy: number,
  scale: number
) {
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const x1 = x * Math.cos(yaw) - z * Math.sin(yaw);
  const z1 = x * Math.sin(yaw) + z * Math.cos(yaw);
  const y2 = y * Math.cos(pitch) - z1 * Math.sin(pitch);
  const z2 = y * Math.sin(pitch) + z1 * Math.cos(pitch);
  const persp = 18 / (18 + z2 * 0.65);
  return {
    px: cx + x1 * scale * persp,
    py: cy - y2 * scale * persp,
    depth: z2,
    persp
  };
}

// Render a 3D extruded prism bar (front, top, right side facets) in SVG with zero label collision
function Bar3D({
  x,
  y,
  w,
  h,
  dx = 9,
  dy = -7,
  frontColor,
  topColor,
  sideColor,
  label,
  labelColor = '#065f46',
  labelOffsetY = 0,
  xLabel,
  subY = 224
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  dx?: number;
  dy?: number;
  frontColor: string;
  topColor: string;
  sideColor: string;
  label?: string;
  labelColor?: string;
  labelOffsetY?: number;
  xLabel?: string;
  subY?: number;
}) {
  const topPts = `${x},${y} ${x + dx},${y + dy} ${x + w + dx},${y + dy} ${x + w},${y}`;
  const sidePts = `${x + w},${y} ${x + w + dx},${y + dy} ${x + w + dx},${y + h + dy} ${x + w},${y + h}`;
  const labelX = x + w / 2 + dx * 0.45;
  const labelY = Math.max(18, y + dy - 6 + labelOffsetY);
  return (
    <g>
      <polygon points={topPts} fill={topColor} />
      <polygon points={sidePts} fill={sideColor} />
      <rect x={x} y={y} width={w} height={h} fill={frontColor} rx="1.5" />
      {label && (
        <>
          {labelOffsetY < -4 && (
            <line
              x1={labelX}
              y1={y + dy - 1}
              x2={labelX}
              y2={labelY + 3}
              stroke="#64748b"
              strokeWidth="0.9"
              strokeDasharray="2 2"
            />
          )}
          <text
            x={labelX}
            y={labelY}
            textAnchor="middle"
            fontSize="11.5"
            fontFamily={CHART_FONTS.mono}
            fontWeight="700"
            fill={labelColor}
            style={{ paintOrder: 'stroke', stroke: '#ffffff', strokeWidth: '3.5px', strokeLinejoin: 'round' }}
          >
            {label}
          </text>
        </>
      )}
      {xLabel && (
        <text
          x={x + w / 2}
          y={subY}
          textAnchor="middle"
          fontSize="11.5"
          fontFamily={CHART_FONTS.mono}
          fontWeight="600"
          fill={CHART_COLORS.tickText}
          style={{ paintOrder: 'stroke', stroke: '#f8fafc', strokeWidth: '2.5px' }}
        >
          {xLabel}
        </text>
      )}
    </g>
  );
}

export default function SimulatorMultiModalCharts({
  project,
  dataType,
  mlTechnique,
  trainingLogs,
  tensorShape,
  seed,
  onSelectMlTechnique
}: SimulatorMultiModalChartsProps) {
  const [yawDeg, setYawDeg] = useState(32);
  const [pitchDeg, setPitchDeg] = useState(22);

  const hashSeed =
    seed * 31 +
    project.id.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0) +
    mlTechnique.length * 17 +
    dataType.length * 11;
  const rand = seededRand(hashSeed);

  // 1. Per-Epoch Validation Accuracy Bar Data
  const epochBars =
    trainingLogs.length > 0
      ? trainingLogs.map((l) => ({
          epoch: l.epoch,
          valPct: Number((l.valAcc * 100).toFixed(1)),
          trainPct: Number((l.accuracy * 100).toFixed(1))
        }))
      : project.epochHistory.map((h) => ({
          epoch: h.epoch,
          valPct: Number(((h.valAccuracy ?? h.accuracy - 0.02) * 100).toFixed(1)),
          trainPct: Number((h.accuracy * 100).toFixed(1))
        }));

  // 2. ML Technique Validation Accuracy Bar Data
  const baseProjAcc =
    project.epochHistory[project.epochHistory.length - 1]?.accuracy || 0.94;
  const mlTechBars = (Object.keys(ML_TECHNIQUE_META) as MlTechniqueId[]).map((id) => {
    const meta = ML_TECHNIQUE_META[id];
    const valAcc = Math.min(0.994, Math.max(0.74, baseProjAcc + meta.accModifier - 0.012));
    return {
      id,
      shortName: meta.shortName.split(' ')[0],
      fullName: meta.shortName,
      valPct: Number((valAcc * 100).toFixed(1)),
      ece: meta.eceScore,
      active: id === mlTechnique
    };
  });

  // 3. Time-Series 3D Ribbon Surface (x=time, y=amplitude, z=quantile band)
  const tsPoints = Array.from({ length: 18 }, (_, i) => {
    const t = i - 6;
    const baseVal = 62 + Math.sin(i * 0.5) * 13 + i * 1.15;
    const forecast = baseVal + (t > 4 ? Math.cos(i * 0.3) * 3 : 0);
    const band = t <= 4 ? 5 : 6 + (t - 4) * 0.65;
    return {
      idx: i,
      tStep: t,
      rawVal: Number(forecast.toFixed(1)),
      x: (i - 8.5) * 0.45,
      val: (forecast - 55) / 15,
      upper: (forecast + band - 55) / 15,
      lower: (forecast - band - 55) / 15,
      anomaly: i === 5 || i === 13
    };
  });

  // 4. 3D k-NN (k=9) & K-Means Decision Manifold - 5 Vivid Cluster Families
  const centroids3D = [
    { id: 'C1', label: 'C1: Emerald Core (-2.6, 1.5, -2.0)', x: -2.6, y: 1.5, z: -2.0, color: '#10b981', ringColor: '#34d399' },
    { id: 'C2', label: 'C2: Crimson Frontier (2.5, 1.9, -1.8)', x: 2.5, y: 1.9, z: -1.8, color: '#f43f5e', ringColor: '#fb7185' },
    { id: 'C3', label: 'C3: Solar Gold (2.2, -1.6, 2.2)', x: 2.2, y: -1.6, z: 2.2, color: '#f59e0b', ringColor: '#fde047' },
    { id: 'C4', label: 'C4: Latent Azure (-2.4, -1.5, 2.0)', x: -2.4, y: -1.5, z: 2.0, color: '#0ea5e9', ringColor: '#7dd3fc' },
    { id: 'C5', label: 'C5: Violet Ring (0.2, 2.2, 0.4)', x: 0.2, y: 2.2, z: 0.4, color: '#a855f7', ringColor: '#d8b4fe' }
  ];
  const query3D = { x: 0.1, y: 0.2, z: 0.1 };
  const pts3D = Array.from({ length: 45 }, (_, idx) => {
    const c = centroids3D[idx % centroids3D.length];
    const px = c.x + (rand() - 0.5) * 2.2;
    const py = c.y + (rand() - 0.5) * 2.0;
    const pz = c.z + (rand() - 0.5) * 2.2;
    const dist = Math.hypot(px - query3D.x, py - query3D.y, pz - query3D.z);
    return { id: idx, x: px, y: py, z: pz, color: c.color, dist, clusterId: c.id };
  });
  const k9Neighbors3D = [...pts3D].sort((a, b) => a.dist - b.dist).slice(0, 9);

  // 5. 3D Dense Vector Constellation (Spaced XYZ directions for zero overlap)
  const vectorRays3D = [
    { name: 'fp_minutia', x: 2.8, y: 2.2, z: 1.2, sim: 0.95, color: '#1e3a8a' },
    { name: 'iris_gabor', x: 3.1, y: 0.6, z: -2.0, sim: 0.97, color: '#065f46' },
    { name: 'navic_geo', x: -2.5, y: 2.6, z: 1.8, sim: 0.90, color: '#4338ca' },
    { name: 'upi_gnn', x: 2.1, y: -2.2, z: 2.4, sim: 0.92, color: '#0f766e' },
    { name: 'telecom', x: -2.9, y: -1.7, z: -1.6, sim: 0.85, color: '#b45309' },
    { name: 'scheme_dbt', x: -0.9, y: 2.4, z: -2.6, sim: 0.88, color: '#9f1239' }
  ];

  // 6. 3D Tensor Voxel Cube (4 x 4 x 3 Depth-Sorted Isometric Voxels)
  const voxels3D: {
    x: number;
    y: number;
    z: number;
    act: number;
    depth: number;
    px: number;
    py: number;
    size: number;
  }[] = [];
  for (let gx = 0; gx < 4; gx++) {
    for (let gy = 0; gy < 4; gy++) {
      for (let gz = 0; gz < 3; gz++) {
        const act = Math.min(1, Math.abs(Math.sin(gx * 0.9 + gy * 0.7 + gz * 1.1)) * 0.75 + rand() * 0.25);
        const p = project3D((gx - 1.5) * 1.25, (gy - 1.5) * 1.1, (gz - 1) * 1.25, yawDeg, pitchDeg, 232, 132, 26);
        voxels3D.push({ x: gx, y: gy, z: gz, act, depth: p.depth, px: p.px, py: p.py, size: 9.5 + act * 8 });
      }
    }
  }
  voxels3D.sort((a, b) => a.depth - b.depth);

  const epochHistoryFor3D =
    trainingLogs.length > 0
      ? trainingLogs.map((l) => ({
          epoch: l.epoch,
          loss: l.loss,
          accuracy: l.accuracy,
          valAccuracy: l.valAcc
        }))
      : project.epochHistory.map((h) => ({
          epoch: h.epoch,
          loss: h.loss,
          accuracy: h.accuracy,
          valAccuracy: h.valAccuracy ?? Math.max(0.4, h.accuracy - 0.02)
        }));

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-base font-bold text-slate-900">
            Interactive 3D Multi-Technique Analytics &amp; WebGL Research Manifold
          </h3>
          <p className="text-xs text-slate-600">
            Full 3D XYZ Coordinate Frames · WebGL Loss Surfaces, 3D Val Accuracy Bar Pillars, Rotatable 3D Time-Series Ribbons, 3D <span className="font-mono font-semibold">k=9</span> / K-Means Spheres, 3D Embedding Vectors &amp; 3D Tensor Voxels.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 bg-white border border-slate-300 rounded-xl px-3 py-1.5 text-[11px] font-mono">
          <span className="font-semibold text-slate-700">3D View Yaw ({yawDeg}°):</span>
          <input
            type="range"
            min={-60}
            max={60}
            value={yawDeg}
            onChange={(e) => setYawDeg(Number(e.target.value))}
            className="w-24 accent-indigo-600"
          />
          <span className="font-semibold text-slate-700">Pitch ({pitchDeg}°):</span>
          <input
            type="range"
            min={5}
            max={45}
            value={pitchDeg}
            onChange={(e) => setPitchDeg(Number(e.target.value))}
            className="w-20 accent-indigo-600"
          />
        </div>
      </div>

      {/* Primary Interactive Three.js WebGL 3D Viewport */}
      <ThreeDResearchManifold
        project={project}
        mlTechnique={mlTechnique}
        dataType={dataType}
        epochPoints={epochHistoryFor3D}
        initialMode="loss_surface"
        heightClass="h-[460px]"
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* CHART 1: 3D Validation Accuracy Bar Chart (Per Epoch) */}
        <div className="rounded-2xl bg-white border border-slate-300 p-5 shadow-xs space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 className="text-base font-bold text-slate-900">
                1. 3D Validation Accuracy Bar Chart (Per Epoch)
              </h4>
              <p className="text-xs font-mono text-slate-600">
                X: Epoch · Y: Val Acc % · Z: Extruded Depth · SLA ≥ 95.0%
              </p>
            </div>
            <span className="text-xs font-mono text-emerald-900 bg-emerald-50 px-2.5 py-1 rounded border border-emerald-200 font-bold tabular-nums">
              Latest Val: {epochBars[epochBars.length - 1]?.valPct}%
            </span>
          </div>

          <svg viewBox="0 0 580 250" className="w-full h-72 bg-slate-50 rounded-xl border border-slate-200 overflow-visible">
            {/* 3D Isometric Floor Plane */}
            <polygon points="62,196 80,180 554,180 536,196" fill={CHART_COLORS.stageFloor} stroke={CHART_COLORS.stageStroke} strokeWidth="0.9" />
            {/* 3D Axis Rays from Origin (62, 196) */}
            <line x1="62" y1="196" x2="546" y2="196" stroke={CHART_COLORS.axisX} strokeWidth="1.5" />
            <line x1="62" y1="196" x2="62" y2="30" stroke={CHART_COLORS.axisY} strokeWidth="1.5" />
            <line x1="62" y1="196" x2="80" y2="180" stroke={CHART_COLORS.axisZ} strokeWidth="1.5" />
            <text x="68" y="22" fontSize="11.5" fontFamily={CHART_FONTS.sans} fontWeight="700" fill={CHART_COLORS.axisYLabel}>
              +Y: Val Accuracy (%)
            </text>
            <text x="546" y="242" fontSize="11.5" fontFamily={CHART_FONTS.sans} fontWeight="700" fill={CHART_COLORS.axisXLabel} textAnchor="end">
              +X: Epoch (Z=Depth)
            </text>

            {[25, 50, 75, 100].map((pct) => {
              const y = 192 - (pct / 100) * 144;
              return (
                <g key={pct}>
                  <line x1="62" y1={y} x2="542" y2={y} stroke={CHART_COLORS.gridLine} strokeDasharray="4 4" />
                  <text x="53" y={y + 4} textAnchor="end" fontSize="12" fill={CHART_COLORS.tickText} fontFamily={CHART_FONTS.mono} fontWeight="600">
                    {pct}%
                  </text>
                </g>
              );
            })}
            {epochBars.map((b, idx) => {
              const slotW = 460 / Math.max(1, epochBars.length);
              const barW = Math.min(28, slotW * 0.56);
              const x = 72 + idx * slotW + (slotW - barW) / 2;
              const barH = Math.max(8, (b.valPct / 100) * 144);
              const y = 192 - barH;
              const getBarColors = (valPct: number) => {
                if (valPct >= 96) return { front: '#00f59b', top: '#6ee7b7', side: '#047857', text: '#065f46' };
                if (valPct >= 93) return { front: '#10b981', top: '#34d399', side: '#065f46', text: '#065f46' };
                if (valPct >= 85) return { front: '#06b6d4', top: '#67e8f9', side: '#0e7490', text: '#0e7490' };
                if (valPct >= 75) return { front: '#f59e0b', top: '#fde047', side: '#b45309', text: '#92400e' };
                return { front: '#f43f5e', top: '#fda4af', side: '#be123c', text: '#9f1239' };
              };
              const c = getBarColors(b.valPct);
              const staggerY = epochBars.length > 7 && idx % 2 === 1 ? -15 : 0;
              return (
                <Bar3D
                  key={b.epoch}
                  x={x}
                  y={y}
                  w={barW}
                  h={barH}
                  dx={9}
                  dy={-7}
                  frontColor={c.front}
                  topColor={c.top}
                  sideColor={c.side}
                  label={epochBars.length <= 14 ? `${b.valPct}` : undefined}
                  labelColor={c.text}
                  labelOffsetY={staggerY}
                  xLabel={`E${b.epoch}`}
                  subY={220}
                />
              );
            })}
          </svg>
        </div>

        {/* CHART 2: 3D ML Technique Validation Accuracy Bar Pillars */}
        <div className="rounded-2xl bg-white border border-slate-300 p-5 shadow-xs space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 className="text-base font-bold text-slate-900">
                2. 3D Val Accuracy Bar Chart Across ML Techniques
              </h4>
              <p className="text-xs font-mono text-slate-600">
                X: ML Model · Y: Acc [70..100%] · Z: Depth · Click Pillar to Run
              </p>
            </div>
            <span className="text-xs font-mono text-indigo-900 bg-indigo-50 px-2.5 py-1 rounded border border-indigo-200 font-semibold">
              5-Model 3D XYZ
            </span>
          </div>

          <svg viewBox="0 0 580 250" className="w-full h-72 bg-slate-50 rounded-xl border border-slate-200 overflow-visible">
            <polygon points="62,196 84,178 554,178 532,196" fill={CHART_COLORS.stageFloor} stroke={CHART_COLORS.stageStroke} strokeWidth="0.9" />
            <line x1="62" y1="196" x2="546" y2="196" stroke={CHART_COLORS.axisX} strokeWidth="1.5" />
            <line x1="62" y1="196" x2="62" y2="30" stroke={CHART_COLORS.axisY} strokeWidth="1.5" />
            <line x1="62" y1="196" x2="84" y2="178" stroke={CHART_COLORS.axisZ} strokeWidth="1.5" />
            <text x="68" y="22" fontSize="11.5" fontFamily={CHART_FONTS.sans} fontWeight="700" fill={CHART_COLORS.axisYLabel}>
              +Y: Val Accuracy (%)
            </text>
            <text x="546" y="242" fontSize="11.5" fontFamily={CHART_FONTS.sans} fontWeight="700" fill={CHART_COLORS.axisXLabel} textAnchor="end">
              +X: ML Technique (Z=Depth)
            </text>

            {[75, 85, 95, 100].map((pct) => {
              const y = 192 - ((pct - 70) / 30) * 144;
              return (
                <g key={pct}>
                  <line x1="62" y1={y} x2="542" y2={y} stroke={CHART_COLORS.gridLine} strokeDasharray="4 4" />
                  <text x="53" y={y + 4} textAnchor="end" fontSize="12" fill={CHART_COLORS.tickText} fontFamily={CHART_FONTS.mono} fontWeight="600">
                    {pct}%
                  </text>
                </g>
              );
            })}
            {mlTechBars.map((tb, idx) => {
              const slotW = 460 / mlTechBars.length;
              const barW = 48;
              const x = 74 + idx * slotW + (slotW - barW) / 2;
              const barH = Math.max(14, ((tb.valPct - 70) / 30) * 144);
              const y = 192 - barH;
              const staggerY = idx % 2 === 1 ? -14 : 0;
              const getTechColors = (id: string, active: boolean) => {
                const map: Record<string, { front: string; top: string; side: string; text: string }> = {
                  multimodal_transformer: { front: '#4f46e5', top: '#818cf8', side: '#312e81', text: '#312e81' },
                  gbdt_catboost_xgb: { front: '#059669', top: '#34d399', side: '#065f46', text: '#065f46' },
                  heterogeneous_graphsage: { front: '#d97706', top: '#fbbf24', side: '#92400e', text: '#78350f' },
                  federated_contrastive: { front: '#0284c7', top: '#38bdf8', side: '#0369a1', text: '#0c4a6e' },
                  physics_informed_pinn: { front: '#e11d48', top: '#fb7185', side: '#9f1239', text: '#881337' }
                };
                return map[id] || { front: '#4f46e5', top: '#818cf8', side: '#312e81', text: '#312e81' };
              };
              const tc = getTechColors(tb.id, tb.active);
              return (
                <g
                  key={tb.id}
                  onClick={() => onSelectMlTechnique(tb.id)}
                  className="cursor-pointer"
                >
                  <Bar3D
                    x={x}
                    y={y}
                    w={barW}
                    h={barH}
                    dx={12}
                    dy={-8}
                    frontColor={tc.front}
                    topColor={tc.top}
                    sideColor={tc.side}
                    label={`${tb.valPct}%${tb.active ? ' ★' : ''}`}
                    labelColor={tc.text}
                    labelOffsetY={staggerY}
                    xLabel={tb.shortName}
                    subY={220}
                  />
                </g>
              );
            })}
          </svg>
        </div>

        {/* CHART 3: 3D Time-Series Forecast & Drift Surface Ribbon */}
        <div className="rounded-2xl bg-white border border-slate-300 p-5 shadow-xs space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 className="text-base font-bold text-slate-900">
                3. 3D Time-Series Forecasting Ribbon (X:Time · Y:Load · Z:σ-Band)
              </h4>
              <p className="text-xs font-mono text-slate-600">
                3D XYZ Coordinate Frame · Forecast Curtain [t-6..t+11] · Peak Anomaly Callouts
              </p>
            </div>
            <span className="text-xs font-mono text-indigo-900 bg-indigo-50 px-2.5 py-1 rounded border border-indigo-200 font-semibold">
              3D XYZ Ribbon
            </span>
          </div>

          <svg viewBox="0 0 460 265" className="w-full h-72 bg-slate-50 rounded-xl border border-slate-200 overflow-visible">
            {(() => {
              const cx = 232;
              const cy = 142;
              const sc = 27;
              const origin = project3D(-3.8, -1.3, -1.6, yawDeg, pitchDeg, cx, cy, sc);
              const xEnd = project3D(4.2, -1.3, -1.6, yawDeg, pitchDeg, cx, cy, sc);
              const yEnd = project3D(-3.8, 2.5, -1.6, yawDeg, pitchDeg, cx, cy, sc);
              const zEnd = project3D(-3.8, -1.3, 2.2, yawDeg, pitchDeg, cx, cy, sc);

              const upperBack = tsPoints.map((p) =>
                project3D(p.x, p.upper - 0.9, -1.4, yawDeg, pitchDeg, cx, cy, sc)
              );
              const lowerFront = tsPoints.map((p) =>
                project3D(p.x, p.lower - 0.9, 1.4, yawDeg, pitchDeg, cx, cy, sc)
              );
              const midCurve = tsPoints.map((p) =>
                project3D(p.x, p.val - 0.9, 0, yawDeg, pitchDeg, cx, cy, sc)
              );

              return (
                <>
                  {/* 3D Coordinate Axes */}
                  <line x1={origin.px} y1={origin.py} x2={xEnd.px} y2={xEnd.py} stroke={CHART_COLORS.axisX} strokeWidth="1.6" />
                  <line x1={origin.px} y1={origin.py} x2={yEnd.px} y2={yEnd.py} stroke={CHART_COLORS.axisY} strokeWidth="1.6" />
                  <line x1={origin.px} y1={origin.py} x2={zEnd.px} y2={zEnd.py} stroke={CHART_COLORS.axisZ} strokeWidth="1.6" />
                  <text x={origin.px - 8} y={origin.py + 14} textAnchor="end" fontSize="10.5" fontFamily={CHART_FONTS.mono} fontWeight="600" fill={CHART_COLORS.mutedText}>
                    (t-6,40,-σ)
                  </text>
                  <text x={xEnd.px} y={xEnd.py + 16} textAnchor="middle" fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.axisXLabel}>
                    +X: t+11
                  </text>
                  <text x={yEnd.px - 6} y={yEnd.py - 8} textAnchor="end" fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.axisYLabel}>
                    +Y: Load 100
                  </text>
                  <text x={zEnd.px - 8} y={zEnd.py + 14} textAnchor="end" fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.axisZLabel}>
                    +Z: +σ
                  </text>

                  {/* 3D Ribbed Quads */}
                  {tsPoints.slice(0, -1).map((p, i) => {
                    const p1 = upperBack[i];
                    const p2 = upperBack[i + 1];
                    const p3 = lowerFront[i + 1];
                    const p4 = lowerFront[i];
                    return (
                      <polygon
                        key={p.idx}
                        points={`${p1.px},${p1.py} ${p2.px},${p2.py} ${p3.px},${p3.py} ${p4.px},${p4.py}`}
                        fill={i > 9 ? '#6366f1' : '#2563eb'}
                        fillOpacity="0.34"
                        stroke="#4338ca"
                        strokeWidth="0.9"
                      />
                    );
                  })}
                  {/* Center Forecast Spine */}
                  <polyline
                    fill="none"
                    stroke="#1e1b4b"
                    strokeWidth="2.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points={midCurve.map((m) => `${m.px},${m.py}`).join(' ')}
                  />
                  {/* Anomaly Nodes in 3D with Offset Non-Overlapping Coordinates */}
                  {tsPoints.map((p, i) => {
                    if (!p.anomaly) return null;
                    const mx = midCurve[i].px;
                    const my = midCurve[i].py;
                    const labelY = Math.max(22, my - 28);
                    return (
                      <g key={`anom-${p.idx}`}>
                        <line x1={mx} y1={my} x2={mx} y2={labelY + 5} stroke="#be123c" strokeWidth="1.2" strokeDasharray="3 2" />
                        <circle
                          cx={mx}
                          cy={my}
                          r="5.5"
                          fill="#be123c"
                          stroke="#ffffff"
                          strokeWidth="1.8"
                        />
                        <text
                          x={mx}
                          y={labelY}
                          textAnchor="middle"
                          fontSize="11"
                          fontFamily={CHART_FONTS.mono}
                          fontWeight="700"
                          fill="#9f1239"
                          style={{ paintOrder: 'stroke', stroke: '#f8fafc', strokeWidth: '3.5px' }}
                        >
                          ({p.tStep >= 0 ? `t+${p.tStep}` : `t${p.tStep}`}, {p.rawVal})
                        </text>
                      </g>
                    );
                  })}
                </>
              );
            })()}
          </svg>
        </div>

        {/* CHART 4: 3D k-NN (k=9) & K-Means Decision Manifold */}
        <div className="rounded-2xl bg-white border border-slate-300 p-5 shadow-xs space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 className="text-base font-bold text-slate-900">
                4. 3D k-NN (<span className="font-mono">k=9</span>) &amp; K-Means Decision Manifold
              </h4>
              <p className="text-xs font-mono text-slate-600">
                Rotatable 3D Latent Cube (X, Y, Z) · Query Node q(0.1,0.2,0.1) &amp; Centroids
              </p>
            </div>
            <span className="text-xs font-mono text-emerald-900 bg-emerald-50 px-2.5 py-1 rounded border border-emerald-200 font-semibold">
              3D k=9 Rays
            </span>
          </div>

          <svg viewBox="0 0 460 265" className="w-full h-72 bg-slate-50 rounded-xl border border-slate-200 overflow-visible">
            {(() => {
              const cx = 230;
              const cy = 136;
              const sc = 26;
              const origin = project3D(0, 0, 0, yawDeg, pitchDeg, cx, cy, sc);
              const axX = project3D(3.8, 0, 0, yawDeg, pitchDeg, cx, cy, sc);
              const axXNeg = project3D(-3.8, 0, 0, yawDeg, pitchDeg, cx, cy, sc);
              const axY = project3D(0, 3.4, 0, yawDeg, pitchDeg, cx, cy, sc);
              const axYNeg = project3D(0, -3.0, 0, yawDeg, pitchDeg, cx, cy, sc);
              const axZ = project3D(0, 0, 3.8, yawDeg, pitchDeg, cx, cy, sc);
              const axZNeg = project3D(0, 0, -3.8, yawDeg, pitchDeg, cx, cy, sc);
              const qProj = project3D(query3D.x, query3D.y, query3D.z, yawDeg, pitchDeg, cx, cy, sc);

              const projectedAll = pts3D
                .map((p) => ({
                  ...p,
                  ...project3D(p.x, p.y, p.z, yawDeg, pitchDeg, cx, cy, sc)
                }))
                .sort((a, b) => a.depth - b.depth);

              return (
                <>
                  {/* 3D Orthogonal Negative Hairlines + Positive Colored Axes */}
                  <line x1={origin.px} y1={origin.py} x2={axXNeg.px} y2={axXNeg.py} stroke="#cbd5e1" strokeDasharray="3 3" />
                  <line x1={origin.px} y1={origin.py} x2={axYNeg.px} y2={axYNeg.py} stroke="#cbd5e1" strokeDasharray="3 3" />
                  <line x1={origin.px} y1={origin.py} x2={axZNeg.px} y2={axZNeg.py} stroke="#cbd5e1" strokeDasharray="3 3" />

                  <line x1={origin.px} y1={origin.py} x2={axX.px} y2={axX.py} stroke={CHART_COLORS.axisX} strokeWidth="1.5" />
                  <line x1={origin.px} y1={origin.py} x2={axY.px} y2={axY.py} stroke={CHART_COLORS.axisY} strokeWidth="1.5" />
                  <line x1={origin.px} y1={origin.py} x2={axZ.px} y2={axZ.py} stroke={CHART_COLORS.axisZ} strokeWidth="1.5" />

                  <text x={axX.px + 6} y={axX.py + 4} fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.axisXLabel}>+X(3.8)</text>
                  <text x={axY.px} y={axY.py - 8} textAnchor="middle" fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.axisYLabel}>+Y(3.4)</text>
                  <text x={axZ.px - 6} y={axZ.py + 14} textAnchor="end" fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.axisZLabel}>+Z(3.8)</text>

                  {/* k=9 Nearest Neighbor Colored Rays */}
                  {k9Neighbors3D.map((n) => {
                    const np = project3D(n.x, n.y, n.z, yawDeg, pitchDeg, cx, cy, sc);
                    return (
                      <line
                        key={`k9-${n.id}`}
                        x1={qProj.px}
                        y1={qProj.py}
                        x2={np.px}
                        y2={np.py}
                        stroke={n.color}
                        strokeWidth="1.8"
                        opacity="0.85"
                      />
                    );
                  })}

                  {/* Depth-sorted 3D Spheres */}
                  {projectedAll.map((p) => (
                    <circle
                      key={p.id}
                      cx={p.px}
                      cy={p.py}
                      r={Math.max(3.4, 4.8 * p.persp)}
                      fill={p.color}
                      opacity={0.88}
                    />
                  ))}

                  {/* Centroid Rings + Radially Pushed Non-Overlapping Labels */}
                  {centroids3D.map((c) => {
                    const cp = project3D(c.x, c.y, c.z, yawDeg, pitchDeg, cx, cy, sc);
                    const dxOut = cp.px >= cx ? 12 : -12;
                    const dyOut = cp.py >= cy ? 16 : -14;
                    return (
                      <g key={c.id}>
                        <circle cx={cp.px} cy={cp.py} r="7.5" fill="none" stroke={c.color} strokeWidth="2.2" />
                        <text
                          x={cp.px + dxOut}
                          y={Math.min(248, Math.max(20, cp.py + dyOut))}
                          textAnchor={cp.px >= cx ? 'start' : 'end'}
                          fontSize="10.5"
                          fontFamily={CHART_FONTS.mono}
                          fontWeight="700"
                          fill={c.color}
                          style={{ paintOrder: 'stroke', stroke: '#f8fafc', strokeWidth: '3.5px' }}
                        >
                          {c.label}
                        </text>
                      </g>
                    );
                  })}
                  {/* Query Node q: Radiant Gold Diamond with Target Halo */}
                  <circle cx={qProj.px} cy={qProj.py} r="9.0" fill="none" stroke="#6366f1" strokeWidth="1.8" strokeDasharray="3 2" />
                  <polygon
                    points={`${qProj.px},${qProj.py - 7} ${qProj.px + 7},${qProj.py} ${qProj.px},${qProj.py + 7} ${qProj.px - 7},${qProj.py}`}
                    fill="#fbbf24"
                    stroke="#b45309"
                    strokeWidth="1.5"
                  />
                  <text
                    x={qProj.px}
                    y={qProj.py - 12}
                    textAnchor="middle"
                    fontSize="11.5"
                    fontFamily={CHART_FONTS.mono}
                    fontWeight="700"
                    fill="#1e1b4b"
                    style={{ paintOrder: 'stroke', stroke: '#f8fafc', strokeWidth: '3.5px' }}
                  >
                    ★ q(0.1,0.2,0.1)
                  </text>
                </>
              );
            })()}
          </svg>
        </div>

        {/* CHART 5: 3D Dense Vector Cosine Constellation */}
        <div className="rounded-2xl bg-white border border-slate-300 p-5 shadow-xs space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 className="text-base font-bold text-slate-900">
                5. 3D Dense Vector Embedding Constellation
              </h4>
              <p className="text-xs font-mono text-slate-600">
                3D Cosine Unit Axes (±X, ±Y, ±Z) · Orthogonal FAISS L2 Projection
              </p>
            </div>
            <span className="text-xs font-mono text-blue-900 bg-blue-50 px-2.5 py-1 rounded border border-blue-200 font-semibold">
              XYZ Vector Space
            </span>
          </div>

          <svg viewBox="0 0 460 265" className="w-full h-72 bg-slate-50 rounded-xl border border-slate-200 overflow-visible">
            {(() => {
              const cx = 230;
              const cy = 136;
              const sc = 25;
              const orig = project3D(0, 0, 0, yawDeg, pitchDeg, cx, cy, sc);
              const xAxis = project3D(3.6, 0, 0, yawDeg, pitchDeg, cx, cy, sc);
              const yAxis = project3D(0, 3.4, 0, yawDeg, pitchDeg, cx, cy, sc);
              const zAxis = project3D(0, 0, 3.6, yawDeg, pitchDeg, cx, cy, sc);

              const projectedRays = vectorRays3D
                .map((v) => ({
                  ...v,
                  ...project3D(v.x, v.y, v.z, yawDeg, pitchDeg, cx, cy, sc)
                }))
                .sort((a, b) => a.depth - b.depth);

              return (
                <>
                  {/* Equator Reference Ring + Orthogonal XYZ Frame */}
                  <ellipse cx={orig.px} cy={orig.py} rx="108" ry="34" fill="none" stroke="#cbd5e1" strokeDasharray="4 3" />
                  <line x1={orig.px} y1={orig.py} x2={xAxis.px} y2={xAxis.py} stroke="#94a3b8" strokeDasharray="3 3" strokeWidth="1.2" />
                  <line x1={orig.px} y1={orig.py} x2={yAxis.px} y2={yAxis.py} stroke="#94a3b8" strokeDasharray="3 3" strokeWidth="1.2" />
                  <line x1={orig.px} y1={orig.py} x2={zAxis.px} y2={zAxis.py} stroke="#94a3b8" strokeDasharray="3 3" strokeWidth="1.2" />
                  <text x={xAxis.px + 6} y={xAxis.py + 4} fontSize="10.5" fontFamily={CHART_FONTS.mono} fontWeight="600" fill={CHART_COLORS.mutedText}>+X(1,0,0)</text>
                  <text x={yAxis.px} y={yAxis.py - 8} textAnchor="middle" fontSize="10.5" fontFamily={CHART_FONTS.mono} fontWeight="600" fill={CHART_COLORS.mutedText}>+Y(0,1,0)</text>
                  <text x={zAxis.px - 6} y={zAxis.py + 14} textAnchor="end" fontSize="10.5" fontFamily={CHART_FONTS.mono} fontWeight="600" fill={CHART_COLORS.mutedText}>+Z(0,0,1)</text>
                  <circle cx={orig.px} cy={orig.py} r="4.5" fill="#0f172a" />

                  {projectedRays.map((v) => {
                    const dx = v.px - orig.px;
                    const dy = v.py - orig.py;
                    const len = Math.max(1, Math.hypot(dx, dy));
                    const lx = v.px + (dx / len) * 13;
                    const ly = Math.min(248, Math.max(20, v.py + (dy / len) * 13));
                    return (
                      <g key={v.name}>
                        <line
                          x1={orig.px}
                          y1={orig.py}
                          x2={v.px}
                          y2={v.py}
                          stroke={v.color}
                          strokeWidth="2.8"
                          strokeLinecap="round"
                        />
                        <circle cx={v.px} cy={v.py} r="5.2" fill={v.color} stroke="#ffffff" strokeWidth="1.4" />
                        <text
                          x={lx}
                          y={ly}
                          textAnchor={dx >= 0 ? 'start' : 'end'}
                          fontSize="11.5"
                          fontFamily={CHART_FONTS.mono}
                          fontWeight="700"
                          fill={v.color}
                          style={{ paintOrder: 'stroke', stroke: '#f8fafc', strokeWidth: '3.5px' }}
                        >
                          {v.name} ({v.sim})
                        </text>
                      </g>
                    );
                  })}
                </>
              );
            })()}
          </svg>
        </div>

        {/* CHART 6: 3D Multi-Dimensional Tensor Voxel Cube */}
        <div className="rounded-2xl bg-white border border-slate-300 p-5 shadow-xs space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h4 className="text-base font-bold text-slate-900">
                6. 3D Tensor Activation Voxel Cube
              </h4>
              <p className="text-xs font-mono text-slate-600">
                3D [4×4×3] Coordinate Frame · Shape: <span className="text-indigo-800 font-semibold">{tensorShape}</span>
              </p>
            </div>
            <span className="text-xs font-mono text-slate-800 bg-slate-100 px-2.5 py-1 rounded border border-slate-300 font-semibold">
              FP16 XYZ Voxels
            </span>
          </div>

          <svg viewBox="0 0 460 265" className="w-full h-72 bg-slate-50 rounded-xl border border-slate-200 overflow-visible">
            {(() => {
              const cx = 232;
              const cy = 132;
              const sc = 26;
              const p000 = project3D(-2.3, -2.1, -1.7, yawDeg, pitchDeg, cx, cy, sc);
              const pX = project3D(2.6, -2.1, -1.7, yawDeg, pitchDeg, cx, cy, sc);
              const pY = project3D(-2.3, 2.3, -1.7, yawDeg, pitchDeg, cx, cy, sc);
              const pZ = project3D(-2.3, -2.1, 2.1, yawDeg, pitchDeg, cx, cy, sc);
              const pMax = project3D(2.1, 2.0, 1.6, yawDeg, pitchDeg, cx, cy, sc);

              return (
                <>
                  <line x1={p000.px} y1={p000.py} x2={pX.px} y2={pX.py} stroke={CHART_COLORS.axisX} strokeWidth="1.5" />
                  <line x1={p000.px} y1={p000.py} x2={pY.px} y2={pY.py} stroke={CHART_COLORS.axisY} strokeWidth="1.5" />
                  <line x1={p000.px} y1={p000.py} x2={pZ.px} y2={pZ.py} stroke={CHART_COLORS.axisZ} strokeWidth="1.5" />
                  <text x={p000.px - 8} y={p000.py + 14} textAnchor="end" fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="600" fill={CHART_COLORS.mutedText}>
                    (0,0,0)
                  </text>
                  <text x={pX.px + 8} y={pX.py + 10} fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.axisXLabel}>
                    +X: Batch[0..3]
                  </text>
                  <text x={pY.px - 6} y={pY.py - 8} textAnchor="end" fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.axisYLabel}>
                    +Y: Layer[0..3]
                  </text>
                  <text x={pZ.px - 8} y={pZ.py + 14} textAnchor="end" fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.axisZLabel}>
                    +Z: Head[0..2]
                  </text>

                  {voxels3D.map((v, i) => {
                    const s = v.size;
                    let frontC = '#4f46e5';
                    let topC = '#818cf8';
                    let sideC = '#312e81';
                    if (v.act >= 0.82) {
                      frontC = '#ef4444'; // Hot Flame Crimson
                      topC = '#f87171';
                      sideC = '#b91c1c';
                    } else if (v.act >= 0.65) {
                      frontC = '#f59e0b'; // Solar Gold / Amber
                      topC = '#fde047';
                      sideC = '#b45309';
                    } else if (v.act >= 0.45) {
                      frontC = '#10b981'; // Spring Emerald
                      topC = '#6ee7b7';
                      sideC = '#047857';
                    } else if (v.act >= 0.25) {
                      frontC = '#06b6d4'; // Electric Cyan
                      topC = '#67e8f9';
                      sideC = '#0e7490';
                    }
                    return (
                      <Bar3D
                        key={i}
                        x={v.px - s / 2}
                        y={v.py - s / 2}
                        w={s}
                        h={s}
                        dx={5}
                        dy={-4}
                        frontColor={frontC}
                        topColor={topC}
                        sideColor={sideC}
                      />
                    );
                  })}
                  <text
                    x={pMax.px + 14}
                    y={Math.max(22, pMax.py - 10)}
                    fontSize="11.5"
                    fontFamily={CHART_FONTS.mono}
                    fontWeight="700"
                    fill="#312e81"
                    style={{ paintOrder: 'stroke', stroke: '#f8fafc', strokeWidth: '3.5px' }}
                  >
                    (3,3,2)
                  </text>
                </>
              );
            })()}
          </svg>
        </div>
      </div>
    </div>
  );
}
