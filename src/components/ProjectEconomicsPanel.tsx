import React, { useMemo, useState } from 'react';
import {
  Activity,
  Building2,
  Coins,
  Cpu,
  Layers,
  Server,
  ShieldCheck,
  TrendingUp,
  Users,
  Wrench,
  Zap
} from 'lucide-react';
import { FlagshipProject, OpExCostKey } from '../types';
import {
  computeProjectEconomics,
  formatInrCompact,
  formatInrExact
} from '../utils/projectEconomics';

interface ProjectEconomicsPanelProps {
  project: FlagshipProject;
}

const OPEX_PALETTE: Record<
  OpExCostKey,
  {
    front: string;
    top: string;
    side: string;
    barClass: string;
    textClass: string;
    dotClass: string;
  }
> = {
  infra: {
    front: '#1d4ed8',
    top: '#60a5fa',
    side: '#1e3a8a',
    barClass: 'bg-blue-700',
    textClass: 'text-blue-900',
    dotClass: 'bg-blue-700'
  },
  aiOps: {
    front: '#4f46e5',
    top: '#818cf8',
    side: '#312e81',
    barClass: 'bg-indigo-600',
    textClass: 'text-indigo-900',
    dotClass: 'bg-indigo-600'
  },
  mlOps: {
    front: '#059669',
    top: '#34d399',
    side: '#065f46',
    barClass: 'bg-emerald-600',
    textClass: 'text-emerald-900',
    dotClass: 'bg-emerald-600'
  },
  energy: {
    front: '#d97706',
    top: '#fbbf24',
    side: '#92400e',
    barClass: 'bg-amber-600',
    textClass: 'text-amber-900',
    dotClass: 'bg-amber-600'
  },
  manpower: {
    front: '#be123c',
    top: '#fb7185',
    side: '#881337',
    barClass: 'bg-rose-700',
    textClass: 'text-rose-900',
    dotClass: 'bg-rose-700'
  },
  maintenance: {
    front: '#475569',
    top: '#94a3b8',
    side: '#1e293b',
    barClass: 'bg-slate-600',
    textClass: 'text-slate-800',
    dotClass: 'bg-slate-600'
  }
};

function getCategoryIcon(key: OpExCostKey) {
  switch (key) {
    case 'infra':
      return <Server className="w-4 h-4 text-blue-800 shrink-0" />;
    case 'aiOps':
      return <Activity className="w-4 h-4 text-indigo-700 shrink-0" />;
    case 'mlOps':
      return <Cpu className="w-4 h-4 text-emerald-700 shrink-0" />;
    case 'energy':
      return <Zap className="w-4 h-4 text-amber-700 shrink-0" />;
    case 'manpower':
      return <Users className="w-4 h-4 text-rose-800 shrink-0" />;
    case 'maintenance':
      return <Wrench className="w-4 h-4 text-slate-700 shrink-0" />;
  }
}

export default function ProjectEconomicsPanel({ project }: ProjectEconomicsPanelProps) {
  const econ = useMemo(() => computeProjectEconomics(project), [project]);
  const [selectedYearIdx, setSelectedYearIdx] = useState<number>(19); // Default Year 20
  const [showAll20Years, setShowAll20Years] = useState<boolean>(false);

  const milestoneYears = useMemo(() => [1, 4, 8, 12, 16, 20], []);
  const milestoneRows = useMemo(
    () => milestoneYears.map((yr) => econ.yearlySchedule[yr - 1]).filter(Boolean),
    [econ.yearlySchedule, milestoneYears]
  );

  const tableRows = useMemo(() => {
    if (showAll20Years) return econ.yearlySchedule;
    return [1, 2, 5, 10, 15, 20].map((yr) => econ.yearlySchedule[yr - 1]).filter(Boolean);
  }, [econ.yearlySchedule, showAll20Years]);

  const activeRow = econ.yearlySchedule[selectedYearIdx] || econ.yearlySchedule[19];

  // Max vertical scale for the 3D Isometric Cost Trajectory Chart
  const maxAnnualOpEx = Math.max(...milestoneRows.map((r) => r.totalOpEx), 1);
  const chartScaleMax = Math.max(econ.totalCapExUsd * 0.45, maxAnnualOpEx * 1.15);

  const stackOrder: OpExCostKey[] = [
    'infra',
    'manpower',
    'mlOps',
    'aiOps',
    'energy',
    'maintenance'
  ];

  return (
    <section className="bg-slate-100 rounded-2xl border border-slate-300 p-6 space-y-6 shadow-xs">
      {/* Header Row */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 pb-4 border-b border-slate-300">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-slate-600 mb-1">
            <span className="font-bold text-blue-900">CapEx &amp; 20-Year YoY OpEx Financial Architecture</span>
            <span aria-hidden="true">·</span>
            <span>{econ.deploymentTier}</span>
          </div>
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <Coins className="w-5 h-5 text-blue-900 shrink-0" />
            <span>Total CapEx, 20-Yr YoY OpEx &amp; Operational Cost Breakdown</span>
          </h3>
          <p className="text-xs text-slate-700 mt-1">
            Comprehensive 20-year lifecycle economics covering initial CapEx and annual YoY OpEx across Infrastructure, AIOps, MLOps, Energy, Manpower, and Maintenance.
          </p>
        </div>

        <div className="text-xs font-mono bg-white border border-slate-300 rounded-xl px-3.5 py-2 text-slate-800 shrink-0">
          <span className="text-slate-500 block text-[10px]">Estimated Unit Economics</span>
          <span className="font-bold text-emerald-900">{econ.unitCostSummary}</span>
        </div>
      </div>

      {/* Top Executive KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-1">
          <div className="text-[11px] font-mono text-slate-500 font-semibold">Total CapEx (Day 0)</div>
          <div className="text-xl font-bold font-mono tabular-nums text-blue-950">
            {formatInrCompact(econ.totalCapExUsd)}
          </div>
          <div className="text-[11px] font-mono text-slate-600 tabular-nums">
            {formatInrExact(econ.totalCapExUsd)}
          </div>
        </div>

        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-1">
          <div className="text-[11px] font-mono text-slate-500 font-semibold">OpEx YoY (Year 1)</div>
          <div className="text-xl font-bold font-mono tabular-nums text-emerald-900">
            {formatInrCompact(econ.opExYear1Usd)}/yr
          </div>
          <div className="text-[11px] font-mono text-slate-600 tabular-nums">
            Yr 20: {formatInrCompact(econ.opExYear20Usd)}/yr (+{econ.opExYoYGrowthPct}% CAGR)
          </div>
        </div>

        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-1">
          <div className="text-[11px] font-mono text-slate-500 font-semibold">Total 20-Year OpEx</div>
          <div className="text-xl font-bold font-mono tabular-nums text-indigo-950">
            {formatInrCompact(econ.totalOpEx20YrUsd)}
          </div>
          <div className="text-[11px] font-mono text-slate-600 tabular-nums">
            {formatInrExact(econ.totalOpEx20YrUsd)} (20 Yrs)
          </div>
        </div>

        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-1">
          <div className="text-[11px] font-mono text-slate-500 font-semibold">20-Year Total TCO</div>
          <div className="text-xl font-bold font-mono tabular-nums text-rose-950">
            {formatInrCompact(econ.totalTco20YrUsd)}
          </div>
          <div className="text-[11px] font-mono text-slate-600 tabular-nums">
            CapEx + 20-Yr Cumulative OpEx
          </div>
        </div>

        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-1">
          <div className="text-[11px] font-mono text-slate-500 font-semibold">Power &amp; Staffing</div>
          <div className="text-xl font-bold font-mono tabular-nums text-slate-900">
            {econ.estimatedPowerKw} kW · {econ.fteHeadcount} FTE
          </div>
          <div className="text-[11px] font-mono text-slate-600 tabular-nums">
            {econ.annualEnergyMwhY1} MWh/yr (Year 1 Power)
          </div>
        </div>
      </div>

      {/* 3D Isometric 20-Year OpEx Trajectory Chart + CapEx Asset Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* 3D Isometric Stacked OpEx Trajectory */}
        <div className="lg:col-span-7 bg-white border border-slate-300 rounded-xl p-5 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
              <TrendingUp className="w-4 h-4 text-blue-900" />
              <span>3D Stacked YoY OpEx Trajectory (20-Year Horizon)</span>
            </div>
            <span className="text-[11px] font-mono text-slate-600">
              Click any 3D pillar to inspect annual breakdown
            </span>
          </div>

          <svg viewBox="0 0 640 245" className="w-full h-60 overflow-visible">
            {/* 3D Isometric Floor & Back Wall */}
            <polygon
              points="72,188 92,172 606,172 586,188"
              fill="#f1f5f9"
              stroke="#cbd5e1"
              strokeWidth="1"
            />
            <polygon
              points="72,28 92,12 92,172 72,188"
              fill="#f8fafc"
              stroke="#e2e8f0"
              strokeWidth="1"
            />

            {/* Gridlines */}
            {[24, 65, 106, 147].map((y, idx) => (
              <line
                key={idx}
                x1="92"
                y1={y}
                x2="606"
                y2={y}
                stroke="#e2e8f0"
                strokeDasharray="4 4"
                strokeWidth="1"
              />
            ))}

            {/* 3D Coordinate Axes (+X, +Y, +Z) */}
            <line x1="72" y1="188" x2="606" y2="188" stroke="#e11d48" strokeWidth="1.4" />
            <line x1="72" y1="188" x2="72" y2="20" stroke="#059669" strokeWidth="1.4" />
            <line x1="72" y1="188" x2="92" y2="172" stroke="#2563eb" strokeWidth="1.4" />

            {/* Axis Labels positioned safely away from pillars */}
            <text x="64" y="34" textAnchor="end" fontSize="11" fill="#334155" fontFamily="monospace" fontWeight="bold">
              {formatInrCompact(chartScaleMax)}
            </text>
            <text x="64" y="109" textAnchor="end" fontSize="11" fill="#334155" fontFamily="monospace" fontWeight="bold">
              {formatInrCompact(chartScaleMax * 0.5)}
            </text>
            <text x="64" y="188" textAnchor="end" fontSize="11" fill="#334155" fontFamily="monospace" fontWeight="bold">
              ₹0
            </text>

            <text x="78" y="14" fontSize="10.5" fontWeight="bold" fill="#047857" fontFamily="monospace">
              +Y: Annual Cost (₹)
            </text>
            <text x="97" y="168" fontSize="10.5" fontWeight="bold" fill="#1d4ed8" fontFamily="monospace">
              +Z: 6-Layer OpEx Stack
            </text>
            <text x="610" y="191" fontSize="10.5" fontWeight="bold" fill="#be123c" fontFamily="monospace">
              +X
            </text>
            <text x="16" y="206" fontSize="10" fill="#475569" fontFamily="monospace" fontWeight="bold">
              (0,0,0)
            </text>

            {/* Stacked 3D Columns for Milestone Years */}
            {milestoneRows.map((row, idx) => {
              const slotW = 480 / milestoneRows.length;
              const barW = 34;
              const bx = 104 + idx * slotW + (slotW - barW) / 2;
              const totalH = Math.min(138, Math.max(18, (row.totalOpEx / chartScaleMax) * 136));
              const dx = 10;
              const dy = -8;
              const isSelected = activeRow.year === row.year;

              let currentY = 186;

              return (
                <g
                  key={row.year}
                  className="cursor-pointer"
                  onClick={() => setSelectedYearIdx(row.year - 1)}
                >
                  {stackOrder.map((key, segIdx) => {
                    const segVal = row[key];
                    const segH = Math.max(2, (segVal / row.totalOpEx) * totalH);
                    const yTop = currentY - segH;
                    const pal = OPEX_PALETTE[key];
                    const isTopSegment = segIdx === stackOrder.length - 1;
                    currentY = yTop;

                    return (
                      <g key={key}>
                        {isTopSegment && (
                          <polygon
                            points={`${bx},${yTop} ${bx + dx},${yTop + dy} ${bx + barW + dx},${yTop + dy} ${bx + barW},${yTop}`}
                            fill={pal.top}
                          />
                        )}
                        <polygon
                          points={`${bx + barW},${yTop} ${bx + barW + dx},${yTop + dy} ${bx + barW + dx},${yTop + segH + dy} ${bx + barW},${yTop + segH}`}
                          fill={pal.side}
                        />
                        <rect
                          x={bx}
                          y={yTop}
                          width={barW}
                          height={segH}
                          fill={pal.front}
                          stroke={isSelected ? '#0f172a' : 'none'}
                          strokeWidth={isSelected ? '0.6' : '0'}
                        />
                      </g>
                    );
                  })}

                  {/* Highlight ring around selected column */}
                  {isSelected && (
                    <rect
                      x={bx - 3}
                      y={186 - totalH - 2}
                      width={barW + dx + 6}
                      height={totalH + 6}
                      fill="none"
                      stroke="#1e3a8a"
                      strokeWidth="1.2"
                      strokeDasharray="3 2"
                      rx="2"
                    />
                  )}

                  {/* Zero-overlap Value Label above pillar */}
                  <text
                    x={bx + barW / 2 + dx * 0.45}
                    y={186 - totalH + dy - 6}
                    textAnchor="middle"
                    fontSize="11"
                    fontFamily="monospace"
                    fontWeight="bold"
                    fill="#0f172a"
                    style={{ paintOrder: 'stroke' }}
                    stroke="#ffffff"
                    strokeWidth="3"
                  >
                    {formatInrCompact(row.totalOpEx)}
                  </text>

                  {/* X-Axis Year Tick */}
                  <text
                    x={bx + barW / 2}
                    y="206"
                    textAnchor="middle"
                    fontSize="11.5"
                    fontFamily="monospace"
                    fontWeight={isSelected ? 'bold' : '600'}
                    fill={isSelected ? '#1e3a8a' : '#334155'}
                  >
                    Yr {row.year}
                  </text>
                  <text
                    x={bx + barW / 2}
                    y="219"
                    textAnchor="middle"
                    fontSize="10"
                    fontFamily="monospace"
                    fontWeight="bold"
                    fill="#475569"
                  >
                    {row.calendarYear}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Compact Legend */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 pt-2 border-t border-slate-200 text-[11px] font-mono">
            {econ.opExCategories.map((cat) => (
              <span key={cat.key} className="inline-flex items-center gap-1.5 text-slate-700">
                <span className={`w-2.5 h-2.5 rounded-xs ${OPEX_PALETTE[cat.key].dotClass}`} />
                <span>{cat.shortLabel}</span>
              </span>
            ))}
          </div>

          {/* Active Year Readout Strip */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 text-xs font-mono space-y-1.5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-bold text-blue-950">
                Selected Snapshot: Year {activeRow.year} ({activeRow.calendarYear})
              </span>
              <span className="text-emerald-900 font-bold tabular-nums">
                Annual OpEx: {formatInrExact(activeRow.totalOpEx)} · Cumul. TCO:{' '}
                {formatInrCompact(activeRow.cumulativeTco)}
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1 text-[11px] text-slate-700 tabular-nums">
              <span>Infra: {formatInrCompact(activeRow.infra)}</span>
              <span>AIOps: {formatInrCompact(activeRow.aiOps)}</span>
              <span>MLOps: {formatInrCompact(activeRow.mlOps)}</span>
              <span>Energy: {formatInrCompact(activeRow.energy)}</span>
              <span>Manpower: {formatInrCompact(activeRow.manpower)}</span>
              <span>Maintenance: {formatInrCompact(activeRow.maintenance)}</span>
            </div>
          </div>
        </div>

        {/* CapEx Allocation Breakdown + 6-Pillar OpEx Mix */}
        <div className="lg:col-span-5 space-y-4">
          {/* Initial CapEx Breakdown Card */}
          <div className="bg-white border border-slate-300 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                <Building2 className="w-4 h-4 text-blue-900" />
                <span>Total CapEx Allocation (Day 0)</span>
              </div>
              <span className="text-xs font-mono font-bold text-blue-900 tabular-nums">
                {formatInrExact(econ.totalCapExUsd)}
              </span>
            </div>

            <div className="space-y-2.5">
              {econ.capExBreakdown.map((item, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex items-baseline justify-between gap-2 text-xs">
                    <span className="font-semibold text-slate-800">{item.label}</span>
                    <span className="font-mono font-bold text-blue-950 tabular-nums shrink-0">
                      {formatInrCompact(item.amountUsd)} ({item.sharePct}%)
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-700 rounded-full"
                      style={{ width: `${item.sharePct}%` }}
                    />
                  </div>
                  <div className="text-[11px] text-slate-500 font-mono truncate" title={item.notes}>
                    {item.notes}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 20-Yr OpEx Share Summary */}
          <div className="bg-white border border-slate-300 rounded-xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-bold text-slate-900">
                <Layers className="w-4 h-4 text-emerald-800" />
                <span>20-Year Cumulative OpEx Mix</span>
              </div>
              <span className="text-xs font-mono font-bold text-emerald-900 tabular-nums">
                {formatInrCompact(econ.totalOpEx20YrUsd)}
              </span>
            </div>

            <div className="space-y-2">
              {econ.opExCategories.map((cat) => (
                <div key={cat.key} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium text-slate-800">{cat.shortLabel}</span>
                    <span className="font-mono font-bold tabular-nums text-slate-900">
                      {formatInrCompact(cat.total20YrUsd)} ({cat.sharePct}%)
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${OPEX_PALETTE[cat.key].barClass} rounded-full`}
                      style={{ width: `${cat.sharePct}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Detailed 6-Pillar Cards: Infra, AIOps, MLOps, Energy, Manpower, Maintenance */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {econ.opExCategories.map((cat) => (
          <div
            key={cat.key}
            className="bg-white border border-slate-300 rounded-xl p-4 space-y-2.5 flex flex-col justify-between"
          >
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  {getCategoryIcon(cat.key)}
                  <span className="text-sm font-bold text-slate-900">{cat.label}</span>
                </div>
                <span className={`text-xs font-mono font-bold ${OPEX_PALETTE[cat.key].textClass}`}>
                  {cat.sharePct}%
                </span>
              </div>
              <p className="text-xs text-slate-600 leading-relaxed">{cat.description}</p>
            </div>

            <div className="grid grid-cols-3 gap-2 pt-2.5 border-t border-slate-200 text-[11px] font-mono tabular-nums">
              <div>
                <span className="text-slate-500 block text-[10px]">Yr 1 YoY</span>
                <span className="font-bold text-slate-900">{formatInrCompact(cat.year1Usd)}</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px]">Yr 20 YoY</span>
                <span className="font-bold text-slate-900">{formatInrCompact(cat.year20Usd)}</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px]">20-Yr Total</span>
                <span className={`font-bold ${OPEX_PALETTE[cat.key].textClass}`}>
                  {formatInrCompact(cat.total20YrUsd)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Year-over-Year (YoY) 20-Year OpEx & Cumulative TCO Schedule */}
      <div className="bg-white border border-slate-300 rounded-xl p-5 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-800" />
              <span>Year-over-Year (YoY) OpEx &amp; 20-Year TCO Schedule (2026–2045)</span>
            </h4>
            <p className="text-xs text-slate-600 mt-0.5">
              Annual breakdown across Infra, AIOps, MLOps, Energy, Manpower, and Maintenance with cumulative 20-year OpEx and TCO.
            </p>
          </div>
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg border border-slate-300 shrink-0">
            <button
              type="button"
              onClick={() => setShowAll20Years(false)}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                !showAll20Years
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Milestones (6 Rows)
            </button>
            <button
              type="button"
              onClick={() => setShowAll20Years(true)}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors ${
                showAll20Years
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Full 20-Yr YoY (20 Rows)
            </button>
          </div>
        </div>

        <div className="overflow-x-auto border border-slate-300 rounded-lg">
          <table className="w-full text-left border-collapse text-[11px] font-mono tabular-nums">
            <thead>
              <tr className="bg-slate-100 text-slate-800 border-b border-slate-300">
                <th className="py-2.5 px-2.5 font-bold">Year</th>
                <th className="py-2.5 px-2.5 font-bold">Infra Cost</th>
                <th className="py-2.5 px-2.5 font-bold">AIOps Cost</th>
                <th className="py-2.5 px-2.5 font-bold">MLOps Cost</th>
                <th className="py-2.5 px-2.5 font-bold">Energy Cost</th>
                <th className="py-2.5 px-2.5 font-bold">Manpower</th>
                <th className="py-2.5 px-2.5 font-bold">Maint.</th>
                <th className="py-2.5 px-2.5 font-bold text-blue-950">Annual OpEx</th>
                <th className="py-2.5 px-2.5 font-bold">YoY Δ</th>
                <th className="py-2.5 px-2.5 font-bold text-indigo-950">Cumul. OpEx</th>
                <th className="py-2.5 px-2.5 font-bold text-rose-950">Cumul. TCO</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 text-slate-700">
              {tableRows.map((r) => (
                <tr
                  key={r.year}
                  onClick={() => setSelectedYearIdx(r.year - 1)}
                  className={`cursor-pointer transition-colors ${
                    activeRow.year === r.year ? 'bg-blue-50/70' : 'hover:bg-slate-50'
                  }`}
                >
                  <td className="py-2 px-2.5 font-bold text-slate-900 whitespace-nowrap">
                    Yr {r.year} ({r.calendarYear})
                  </td>
                  <td className="py-2 px-2.5">{formatInrCompact(r.infra)}</td>
                  <td className="py-2 px-2.5">{formatInrCompact(r.aiOps)}</td>
                  <td className="py-2 px-2.5">{formatInrCompact(r.mlOps)}</td>
                  <td className="py-2 px-2.5">{formatInrCompact(r.energy)}</td>
                  <td className="py-2 px-2.5">{formatInrCompact(r.manpower)}</td>
                  <td className="py-2 px-2.5">{formatInrCompact(r.maintenance)}</td>
                  <td className="py-2 px-2.5 font-bold text-blue-950">
                    {formatInrCompact(r.totalOpEx)}
                  </td>
                  <td className="py-2 px-2.5 text-emerald-800">
                    {r.year === 1 ? 'Base' : `+${r.yoyDeltaPct}%`}
                  </td>
                  <td className="py-2 px-2.5 font-semibold text-indigo-950">
                    {formatInrCompact(r.cumulativeOpEx)}
                  </td>
                  <td className="py-2 px-2.5 font-bold text-rose-950">
                    {formatInrCompact(r.cumulativeTco)}
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="bg-slate-100 border-t-2 border-slate-300 font-bold text-slate-900">
                <td className="py-2.5 px-2.5">20-Yr Totals</td>
                <td className="py-2.5 px-2.5">
                  {formatInrCompact(econ.opExCategories[0].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5">
                  {formatInrCompact(econ.opExCategories[1].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5">
                  {formatInrCompact(econ.opExCategories[2].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5">
                  {formatInrCompact(econ.opExCategories[3].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5">
                  {formatInrCompact(econ.opExCategories[4].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5">
                  {formatInrCompact(econ.opExCategories[5].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5 text-blue-950">
                  +{econ.opExYoYGrowthPct}% CAGR
                </td>
                <td className="py-2.5 px-2.5 text-slate-600">CapEx: {formatInrCompact(econ.totalCapExUsd)}</td>
                <td className="py-2.5 px-2.5 text-indigo-950">
                  {formatInrCompact(econ.totalOpEx20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5 text-rose-950">
                  {formatInrCompact(econ.totalTco20YrUsd)}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </section>
  );
}
