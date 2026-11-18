import React, { useMemo, useState, useEffect } from 'react';
import {
  Database,
  RefreshCw,
  Play,
  Download,
  Table2,
  SlidersHorizontal
} from 'lucide-react';
import {
  FlagshipProject,
  SyntheticDataType,
  SyntheticVariant
} from '../types';
import {
  SYNTHETIC_DATA_TYPE_META,
  SYNTHETIC_VARIANT_META,
  generateSyntheticSamples,
  getSyntheticDatasetProfile,
  inferProjectDataType
} from '../utils/syntheticDataset';

interface ProjectSyntheticDatasetPanelProps {
  project: FlagshipProject;
  onRunTrainingWithDataType: (
    project: FlagshipProject,
    dataType: SyntheticDataType,
    variant: SyntheticVariant
  ) => void;
}

export default function ProjectSyntheticDatasetPanel({
  project,
  onRunTrainingWithDataType
}: ProjectSyntheticDatasetPanelProps) {
  const defaultType = useMemo(() => inferProjectDataType(project), [project]);
  const [selectedDataType, setSelectedDataType] = useState<SyntheticDataType>(defaultType);
  const [selectedVariant, setSelectedVariant] = useState<SyntheticVariant>('balanced');
  const [seed, setSeed] = useState<number>(42);

  useEffect(() => {
    setSelectedDataType(inferProjectDataType(project));
    setSelectedVariant('balanced');
    setSeed(42);
  }, [project.id]);

  const profile = useMemo(
    () => getSyntheticDatasetProfile(project, selectedDataType, selectedVariant),
    [project, selectedDataType, selectedVariant]
  );

  const sampleRows = useMemo(
    () => generateSyntheticSamples(project, selectedDataType, selectedVariant, 5, seed),
    [project, selectedDataType, selectedVariant, seed]
  );

  const handleDownloadCsv = () => {
    const fullRows = generateSyntheticSamples(project, selectedDataType, selectedVariant, 50, seed);
    if (fullRows.length === 0) return;
    const headers = Object.keys(fullRows[0]);
    const csvLines = [
      headers.join(','),
      ...fullRows.map((row) =>
        headers.map((h) => `"${String(row[h] ?? '').replace(/"/g, '""')}"`).join(',')
      )
    ];
    const blob = new Blob([csvLines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${project.id}_${selectedDataType}_${selectedVariant}_seed${seed}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-slate-100 hover:bg-slate-200/80 p-6 rounded-xl border border-slate-300 space-y-5 transition-all duration-200 shadow-xs">
      {/* Header Row */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-300">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-rose-900 mb-1">
            <span className="bg-rose-100 px-2 py-0.5 rounded font-bold border border-rose-200">
              [Synthetic Dataset &amp; Modality Spec]
            </span>
            <span className="text-blue-900 font-semibold">
              Default Modality: {SYNTHETIC_DATA_TYPE_META[profile.defaultDataType].shortLabel}
            </span>
            <span>·</span>
            <span className="text-emerald-900 font-semibold">{profile.noiseSigma}</span>
          </div>
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Database className="w-4 h-4 text-blue-900" />
            <span>{profile.datasetName}</span>
          </h3>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={() => setSeed((s) => s + 17)}
            className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-50 text-slate-800 border border-slate-300 text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs"
          >
            <RefreshCw className="w-3.5 h-3.5 text-blue-900" />
            <span>Resample (Seed {seed})</span>
          </button>

          <button
            type="button"
            onClick={handleDownloadCsv}
            className="px-3 py-1.5 rounded-lg bg-white hover:bg-slate-50 text-slate-800 border border-slate-300 text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs"
          >
            <Download className="w-3.5 h-3.5 text-emerald-900" />
            <span>Export 50 Rows (CSV)</span>
          </button>

          <button
            type="button"
            onClick={() => onRunTrainingWithDataType(project, selectedDataType, selectedVariant)}
            className="px-4 py-1.5 rounded-lg bg-blue-900 hover:bg-blue-800 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs"
          >
            <Play className="w-3.5 h-3.5" />
            <span>Run Training on {SYNTHETIC_DATA_TYPE_META[selectedDataType].shortLabel}</span>
          </button>
        </div>
      </div>

      {/* Data Type & Synthetic Distribution Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white p-3.5 rounded-xl border border-slate-300 space-y-2">
          <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-blue-900" />
            <span>Select Data Type / Input Tensor Modality</span>
          </label>
          <select
            value={selectedDataType}
            onChange={(e) => setSelectedDataType(e.target.value as SyntheticDataType)}
            className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900"
          >
            {(Object.keys(SYNTHETIC_DATA_TYPE_META) as SyntheticDataType[]).map((dt) => (
              <option key={dt} value={dt}>
                {SYNTHETIC_DATA_TYPE_META[dt].label}
                {dt === profile.defaultDataType ? ' (Project Native)' : ''}
              </option>
            ))}
          </select>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-300 space-y-2">
          <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
            <Table2 className="w-3.5 h-3.5 text-rose-900" />
            <span>Synthetic Dataset Distribution Profile</span>
          </label>
          <select
            value={selectedVariant}
            onChange={(e) => setSelectedVariant(e.target.value as SyntheticVariant)}
            className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900"
          >
            {(Object.keys(SYNTHETIC_VARIANT_META) as SyntheticVariant[]).map((v) => (
              <option key={v} value={v}>
                {SYNTHETIC_VARIANT_META[v].label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Telemetry Spec Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
        <div className="bg-white p-3 rounded-xl border border-slate-300">
          <span className="text-[10px] text-slate-500 uppercase block">Tensor / Batch Shape</span>
          <span className="text-blue-900 font-bold">{profile.tensorShape}</span>
        </div>
        <div className="bg-white p-3 rounded-xl border border-slate-300">
          <span className="text-[10px] text-slate-500 uppercase block">Synthetic Corpus Size</span>
          <span className="text-slate-900 font-bold">{profile.sampleCountLabel}</span>
        </div>
        <div className="bg-white p-3 rounded-xl border border-slate-300">
          <span className="text-[10px] text-slate-500 uppercase block">Target Objective</span>
          <span className="text-emerald-900 font-bold">{profile.recommendedLoss}</span>
        </div>
        <div className="bg-white p-3 rounded-xl border border-slate-300">
          <span className="text-[10px] text-slate-500 uppercase block">Injection Profile</span>
          <span className="text-rose-900 font-bold">{SYNTHETIC_VARIANT_META[selectedVariant].label.split(' ')[0]} ({profile.noiseSigma})</span>
        </div>
      </div>

      {/* Schema Pill Row */}
      <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
        <span className="text-slate-700 font-bold">Schema:</span>
        {profile.columns.map((col) => (
          <span
            key={col.name}
            className={`px-2.5 py-1 rounded-md border ${
              col.role === 'target'
                ? 'bg-emerald-100 text-emerald-900 border-emerald-300 font-semibold'
                : col.role === 'tensor'
                ? 'bg-blue-100 text-blue-900 border-blue-300 font-semibold'
                : 'bg-white text-slate-800 border-slate-300'
            }`}
          >
            {col.name}: <span className="opacity-80">{col.dtype}</span>
          </span>
        ))}
      </div>

      {/* Generated Synthetic Batch Preview Table */}
      <div className="bg-white rounded-xl border border-slate-300 overflow-hidden shadow-xs">
        <div className="px-4 py-2.5 bg-slate-200/80 border-b border-slate-300 flex items-center justify-between text-xs">
          <span className="font-bold text-slate-900">
            Live Synthetic Sample Preview (First 5 Records · {SYNTHETIC_DATA_TYPE_META[selectedDataType].shortLabel})
          </span>
          <span className="font-mono text-[11px] text-slate-600">{SYNTHETIC_VARIANT_META[selectedVariant].desc}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="bg-slate-100 border-b border-slate-300 text-slate-700">
                {profile.columns.map((col) => (
                  <th key={col.name} className="py-2 px-3.5 font-semibold whitespace-nowrap">
                    {col.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {sampleRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50 transition-colors">
                  {profile.columns.map((col) => (
                    <td
                      key={col.name}
                      className={`py-2 px-3.5 whitespace-nowrap ${
                        col.role === 'target'
                          ? 'text-emerald-900 font-bold'
                          : col.role === 'tensor'
                          ? 'text-blue-900'
                          : 'text-slate-800'
                      }`}
                    >
                      {String(row[col.name])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
