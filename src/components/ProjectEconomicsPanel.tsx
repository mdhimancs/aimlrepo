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
  formatDualCurrency,
  formatRsOnly,
  formatUsdOnly
} from '../utils/projectEconomics';
import { CHART_COLORS, CHART_FONTS } from '../utils/chartTheme';

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
          <div className="text-base font-bold font-mono tabular-nums text-blue-950">
            {formatDualCurrency(econ.totalCapExUsd)}
          </div>
          <div className="text-[11px] font-mono text-slate-600 tabular-nums">
            Day-0 Hardware &amp; Vault Fit-Out
          </div>
        </div>

        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-1">
          <div className="text-[11px] font-mono text-slate-500 font-semibold">OpEx YoY (Year 1)</div>
          <div className="text-base font-bold font-mono tabular-nums text-emerald-900">
            {formatDualCurrency(econ.opExYear1Usd)}/yr
          </div>
          <div className="text-[11px] font-mono text-slate-600 tabular-nums">
            Yr 20: {formatDualCurrency(econ.opExYear20Usd)} (+{econ.opExYoYGrowthPct}% CAGR)
          </div>
        </div>

        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-1">
          <div className="text-[11px] font-mono text-slate-500 font-semibold">Total 20-Year OpEx</div>
          <div className="text-base font-bold font-mono tabular-nums text-indigo-950">
            {formatDualCurrency(econ.totalOpEx20YrUsd)}
          </div>
          <div className="text-[11px] font-mono text-slate-600 tabular-nums">
            Cumulative 20-Yr Run Cost
          </div>
        </div>

        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-1">
          <div className="text-[11px] font-mono text-slate-500 font-semibold">20-Year Total TCO</div>
          <div className="text-base font-bold font-mono tabular-nums text-rose-950">
            {formatDualCurrency(econ.totalTco20YrUsd)}
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
            <div className="flex items-center gap-2 text-base font-bold text-slate-900">
              <TrendingUp className="w-4 h-4 text-blue-900" />
              <span>3D Stacked YoY OpEx Trajectory (20-Year Horizon)</span>
            </div>
            <span className="text-xs font-mono text-slate-600">
              Click any 3D pillar to inspect annual breakdown
            </span>
          </div>

          <svg viewBox="0 0 680 285" className="w-full h-80 overflow-visible">
            {/* 3D Isometric Floor & Back Wall */}
            <polygon
              points="84,218 106,200 644,200 622,218"
              fill={CHART_COLORS.stageFloor}
              stroke={CHART_COLORS.stageStroke}
              strokeWidth="1"
            />
            <polygon
              points="84,36 106,18 106,200 84,218"
              fill={CHART_COLORS.backPlane}
              stroke={CHART_COLORS.gridLine}
              strokeWidth="1"
            />

            {/* Gridlines */}
            {[28, 71, 114, 157].map((y, idx) => (
              <line
                key={idx}
                x1="106"
                y1={y}
                x2="644"
                y2={y}
                stroke={CHART_COLORS.gridLine}
                strokeDasharray="4 4"
                strokeWidth="1"
              />
            ))}

            {/* 3D Coordinate Axes (+X, +Y, +Z) */}
            <line x1="84" y1="218" x2="640" y2="218" stroke={CHART_COLORS.axisX} strokeWidth="1.6" />
            <line x1="84" y1="218" x2="84" y2="24" stroke={CHART_COLORS.axisY} strokeWidth="1.6" />
            <line x1="84" y1="218" x2="106" y2="200" stroke={CHART_COLORS.axisZ} strokeWidth="1.6" />

            {/* Axis Labels positioned safely away from pillars */}
            <text x="74" y="39" textAnchor="end" fontSize="12.5" fill={CHART_COLORS.tickText} fontFamily={CHART_FONTS.mono} fontWeight="700">
              {formatUsdOnly(chartScaleMax)}
            </text>
            <text x="74" y="124" textAnchor="end" fontSize="12.5" fill={CHART_COLORS.tickText} fontFamily={CHART_FONTS.mono} fontWeight="700">
              {formatUsdOnly(chartScaleMax * 0.5)}
            </text>
            <text x="74" y="218" textAnchor="end" fontSize="12.5" fill={CHART_COLORS.tickText} fontFamily={CHART_FONTS.mono} fontWeight="700">
              $0
            </text>

            <text x="90" y="15" fontSize="12" fontWeight="700" fill={CHART_COLORS.axisYLabel} fontFamily={CHART_FONTS.sans}>
              +Y: Annual OpEx ($ USD)
            </text>
            <text x="112" y="194" fontSize="11.5" fontWeight="700" fill={CHART_COLORS.axisZLabel} fontFamily={CHART_FONTS.sans}>
              +Z: 6-Layer Cost Stack
            </text>
            <text x="646" y="222" fontSize="12" fontWeight="700" fill={CHART_COLORS.axisXLabel} fontFamily={CHART_FONTS.sans}>
              +X
            </text>

            {/* Stacked 3D Columns for Milestone Years */}
            {milestoneRows.map((row, idx) => {
              const slotW = 508 / milestoneRows.length;
              const barW = 42;
              const bx = 114 + idx * slotW + (slotW - barW) / 2;
              const totalH = Math.min(162, Math.max(22, (row.totalOpEx / chartScaleMax) * 160));
              const dx = 12;
              const dy = -9;
              const isSelected = activeRow.year === row.year;

              let currentY = 216;

              return (
                <g
                  key={row.year}
                  className="cursor-pointer"
                  onClick={() => setSelectedYearIdx(row.year - 1)}
                >
                  {stackOrder.map((key, segIdx) => {
                    const segVal = row[key];
                    const segH = Math.max(2.5, (segVal / row.totalOpEx) * totalH);
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
                          strokeWidth={isSelected ? '0.8' : '0'}
                        />
                      </g>
                    );
                  })}

                  {/* Highlight ring around selected column */}
                  {isSelected && (
                    <rect
                      x={bx - 4}
                      y={216 - totalH - 3}
                      width={barW + dx + 8}
                      height={totalH + 8}
                      fill="none"
                      stroke="#1e3a8a"
                      strokeWidth="1.5"
                      strokeDasharray="4 2"
                      rx="3"
                    />
                  )}

                  {/* Zero-overlap Value Label above pillar */}
                  <text
                    x={bx + barW / 2 + dx * 0.45}
                    y={216 - totalH + dy - 7}
                    textAnchor="middle"
                    fontSize="12.5"
                    fontFamily={CHART_FONTS.mono}
                    fontWeight="700"
                    fill="#0f172a"
                    style={{ paintOrder: 'stroke' }}
                    stroke="#ffffff"
                    strokeWidth="4"
                  >
                    {formatUsdOnly(row.totalOpEx)}
                  </text>

                  {/* X-Axis Year Tick */}
                  <text
                    x={bx + barW / 2}
                    y="242"
                    textAnchor="middle"
                    fontSize="12.5"
                    fontFamily={CHART_FONTS.mono}
                    fontWeight={isSelected ? '700' : '600'}
                    fill={isSelected ? '#1e3a8a' : CHART_COLORS.tickText}
                  >
                    Yr {row.year}
                  </text>
                  <text
                    x={bx + barW / 2}
                    y="258"
                    textAnchor="middle"
                    fontSize="11.5"
                    fontFamily={CHART_FONTS.mono}
                    fontWeight="600"
                    fill={CHART_COLORS.mutedText}
                  >
                    {row.calendarYear}
                  </text>
                </g>
              );
            })}
          </svg>

          {/* Compact Legend */}
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2 pt-3 border-t border-slate-200 text-xs font-mono">
            {econ.opExCategories.map((cat) => (
              <span key={cat.key} className="inline-flex items-center gap-1.5 text-slate-800 font-semibold">
                <span className={`w-3 h-3 rounded-xs ${OPEX_PALETTE[cat.key].dotClass}`} />
                <span>{cat.shortLabel}</span>
              </span>
            ))}
          </div>

          {/* Active Year Readout Strip */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 text-xs font-mono space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="font-bold text-blue-950 font-sans text-sm">
                Selected Snapshot: Year {activeRow.year} ({activeRow.calendarYear})
              </span>
              <span className="text-emerald-900 font-bold tabular-nums">
                Annual OpEx: {formatDualCurrency(activeRow.totalOpEx)} · Cumul. TCO:{' '}
                {formatDualCurrency(activeRow.cumulativeTco)}
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-1.5 text-xs text-slate-800 tabular-nums">
              <span>Infra: {formatDualCurrency(activeRow.infra)}</span>
              <span>AIOps: {formatDualCurrency(activeRow.aiOps)}</span>
              <span>MLOps: {formatDualCurrency(activeRow.mlOps)}</span>
              <span>Energy: {formatDualCurrency(activeRow.energy)}</span>
              <span>Manpower: {formatDualCurrency(activeRow.manpower)}</span>
              <span>Maintenance: {formatDualCurrency(activeRow.maintenance)}</span>
            </div>
          </div>
        </div>

        {/* CapEx Allocation Breakdown + 6-Pillar OpEx Mix */}
        <div className="lg:col-span-5 space-y-4">
          {/* Initial CapEx Breakdown Card */}
          <div className="bg-white border border-slate-300 rounded-xl p-5 space-y-3.5">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-base font-bold text-slate-900">
                <Building2 className="w-4 h-4 text-blue-900" />
                <span>Total CapEx Allocation (Day 0)</span>
              </div>
              <span className="text-xs font-mono font-bold text-blue-900 tabular-nums shrink-0">
                {formatDualCurrency(econ.totalCapExUsd)}
              </span>
            </div>

            <div className="space-y-3">
              {econ.capExBreakdown.map((item, idx) => (
                <div key={idx} className="space-y-1.5">
                  <div className="flex items-baseline justify-between gap-2 text-[13px]">
                    <span className="font-semibold text-slate-800">{item.label}</span>
                    <span className="font-mono font-bold text-blue-950 tabular-nums shrink-0">
                      {formatDualCurrency(item.amountUsd)} ({item.sharePct}%)
                    </span>
                  </div>
                  <div className="h-2.5 w-full bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-700 rounded-full"
                      style={{ width: `${item.sharePct}%` }}
                    />
                  </div>
                  <div className="text-xs text-slate-600 truncate" title={item.notes}>
                    {item.notes}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* 20-Yr OpEx Share Summary */}
          <div className="bg-white border border-slate-300 rounded-xl p-5 space-y-3.5">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-base font-bold text-slate-900">
                <Layers className="w-4 h-4 text-emerald-800" />
                <span>20-Year Cumulative OpEx Mix</span>
              </div>
              <span className="text-xs font-mono font-bold text-emerald-900 tabular-nums shrink-0">
                {formatDualCurrency(econ.totalOpEx20YrUsd)}
              </span>
            </div>

            <div className="space-y-2.5">
              {econ.opExCategories.map((cat) => (
                <div key={cat.key} className="space-y-1.5">
                  <div className="flex items-center justify-between gap-2 text-[13px]">
                    <span className="font-semibold text-slate-800">{cat.shortLabel}</span>
                    <span className="font-mono font-bold tabular-nums text-slate-900 shrink-0">
                      {formatDualCurrency(cat.total20YrUsd)} ({cat.sharePct}%)
                    </span>
                  </div>
                  <div className="h-2.5 w-full bg-slate-200 rounded-full overflow-hidden">
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
                <span className="font-bold text-slate-900 block">{formatUsdOnly(cat.year1Usd)}</span>
                <span className="text-[9.5px] text-slate-500 block">{formatRsOnly(cat.year1Usd)}</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px]">Yr 20 YoY</span>
                <span className="font-bold text-slate-900 block">{formatUsdOnly(cat.year20Usd)}</span>
                <span className="text-[9.5px] text-slate-500 block">{formatRsOnly(cat.year20Usd)}</span>
              </div>
              <div>
                <span className="text-slate-500 block text-[10px]">20-Yr Total</span>
                <span className={`font-bold block ${OPEX_PALETTE[cat.key].textClass}`}>
                  {formatUsdOnly(cat.total20YrUsd)}
                </span>
                <span className="text-[9.5px] text-slate-500 block">{formatRsOnly(cat.total20YrUsd)}</span>
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
              Annual breakdown in USD ($) across Infra, AIOps, MLOps, Energy, Manpower, and Maintenance (hover any cell for dual $ / Rs Cr).
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
                  <td className="py-2 px-2.5" title={formatDualCurrency(r.infra)}>{formatUsdOnly(r.infra)}</td>
                  <td className="py-2 px-2.5" title={formatDualCurrency(r.aiOps)}>{formatUsdOnly(r.aiOps)}</td>
                  <td className="py-2 px-2.5" title={formatDualCurrency(r.mlOps)}>{formatUsdOnly(r.mlOps)}</td>
                  <td className="py-2 px-2.5" title={formatDualCurrency(r.energy)}>{formatUsdOnly(r.energy)}</td>
                  <td className="py-2 px-2.5" title={formatDualCurrency(r.manpower)}>{formatUsdOnly(r.manpower)}</td>
                  <td className="py-2 px-2.5" title={formatDualCurrency(r.maintenance)}>{formatUsdOnly(r.maintenance)}</td>
                  <td className="py-2 px-2.5 font-bold text-blue-950" title={formatDualCurrency(r.totalOpEx)}>
                    {formatUsdOnly(r.totalOpEx)}
                  </td>
                  <td className="py-2 px-2.5 text-emerald-800">
                    {r.year === 1 ? 'Base' : `+${r.yoyDeltaPct}%`}
                  </td>
                  <td className="py-2 px-2.5 font-semibold text-indigo-950" title={formatDualCurrency(r.cumulativeOpEx)}>
                    {formatUsdOnly(r.cumulativeOpEx)}
                  </td>
                  <td className="py-2 px-2.5 font-bold text-rose-950" title={formatDualCurrency(r.cumulativeTco)}>
                    {formatUsdOnly(r.cumulativeTco)}
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="bg-slate-100 border-t-2 border-slate-300 font-bold text-slate-900">
                <td className="py-2.5 px-2.5">20-Yr Totals</td>
                <td className="py-2.5 px-2.5" title={formatDualCurrency(econ.opExCategories[0].total20YrUsd)}>
                  {formatUsdOnly(econ.opExCategories[0].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5" title={formatDualCurrency(econ.opExCategories[1].total20YrUsd)}>
                  {formatUsdOnly(econ.opExCategories[1].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5" title={formatDualCurrency(econ.opExCategories[2].total20YrUsd)}>
                  {formatUsdOnly(econ.opExCategories[2].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5" title={formatDualCurrency(econ.opExCategories[3].total20YrUsd)}>
                  {formatUsdOnly(econ.opExCategories[3].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5" title={formatDualCurrency(econ.opExCategories[4].total20YrUsd)}>
                  {formatUsdOnly(econ.opExCategories[4].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5" title={formatDualCurrency(econ.opExCategories[5].total20YrUsd)}>
                  {formatUsdOnly(econ.opExCategories[5].total20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5 text-blue-950">
                  +{econ.opExYoYGrowthPct}% CAGR
                </td>
                <td className="py-2.5 px-2.5 text-slate-600" title={formatDualCurrency(econ.totalCapExUsd)}>
                  CapEx: {formatUsdOnly(econ.totalCapExUsd)}
                </td>
                <td className="py-2.5 px-2.5 text-indigo-950" title={formatDualCurrency(econ.totalOpEx20YrUsd)}>
                  {formatUsdOnly(econ.totalOpEx20YrUsd)}
                </td>
                <td className="py-2.5 px-2.5 text-rose-950" title={formatDualCurrency(econ.totalTco20YrUsd)}>
                  {formatUsdOnly(econ.totalTco20YrUsd)}
                </td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>
    </section>
  );
}
