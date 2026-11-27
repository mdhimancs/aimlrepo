import React, { useMemo, useState, useEffect } from 'react';
import {
  Database,
  RefreshCw,
  Play,
  Download,
  Table2,
  SlidersHorizontal,
  ShieldCheck,
  AlertTriangle,
  Cpu,
  Clock,
  HardDrive,
  Zap,
  Info,
  CheckCircle2,
  Copy,
  Check,
  ExternalLink,
  Layers,
  Activity,
  FileText
} from 'lucide-react';
import {
  DataCleaningChallenge,
  DataSourceMetadata,
  FlagshipProject,
  MLTechniqueComplexity,
  SyntheticColumnSchema,
  SyntheticDataType,
  SyntheticVariant
} from '../types';
import {
  MlTechniqueId,
  ML_TECHNIQUE_META,
  SYNTHETIC_DATA_TYPE_META,
  SYNTHETIC_VARIANT_META,
  generateSyntheticSamples,
  getDataCleaningChallenges,
  getDataSourceInfo,
  getMLTechniqueComplexity,
  getSyntheticDatasetProfile,
  inferBestMlTechnique,
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
  const defaultTechnique = useMemo(() => inferBestMlTechnique(project), [project]);

  const [selectedDataType, setSelectedDataType] = useState<SyntheticDataType>(defaultType);
  const [selectedVariant, setSelectedVariant] = useState<SyntheticVariant>('balanced');
  const [selectedTechnique, setSelectedTechnique] = useState<MlTechniqueId>(defaultTechnique);
  const [seed, setSeed] = useState<number>(42);
  const [activeSampleRowTab, setActiveSampleRowTab] = useState<'row1' | 'row2' | 'both'>('both');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  useEffect(() => {
    setSelectedDataType(inferProjectDataType(project));
    setSelectedVariant('balanced');
    setSelectedTechnique(inferBestMlTechnique(project));
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

  const dataSourceInfo: DataSourceMetadata = useMemo(
    () => getDataSourceInfo(project, selectedDataType),
    [project, selectedDataType]
  );

  const dataCleaningChallenges: DataCleaningChallenge[] = useMemo(
    () => getDataCleaningChallenges(project, selectedDataType),
    [project, selectedDataType]
  );

  const mlComplexity: MLTechniqueComplexity = useMemo(
    () => getMLTechniqueComplexity(selectedTechnique, selectedDataType),
    [selectedTechnique, selectedDataType]
  );

  const row1 = sampleRows[0] || {};
  const row2 = sampleRows[1] || {};

  const handleCopyJson = (data: unknown, key: string) => {
    try {
      navigator.clipboard.writeText(JSON.stringify(data, null, 2));
      setCopiedKey(key);
      setTimeout(() => setCopiedKey(null), 2200);
    } catch {
      // Fallback
    }
  };

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

  const getColumnSemanticDetails = (col: SyntheticColumnSchema) => {
    if (col.role === 'target') {
      return {
        description: 'Ground-truth supervised classification / regression target label',
        range: 'Discrete {0, 1} or Continuous ℝ',
        nullability: '0.0% (Zero Missing · Enforced)'
      };
    }
    if (col.role === 'tensor') {
      return {
        description: 'High-dimensional feature representation / embedding matrix',
        range: '[-1.80, +2.40] L2-normalized vector',
        nullability: '0.0% (Zero Missing · Formatted)'
      };
    }
    if (col.role === 'metadata') {
      return {
        description: 'Deterministic cryptographically salted entity token / UUID',
        range: 'Hex / Alphanumeric token string',
        nullability: '0.0% (Primary Key · Unique)'
      };
    }
    if (col.name.includes('navic') || col.name.includes('geo')) {
      return {
        description: 'Geospatial NavIC lat/lon coordinates & velocity telemetry',
        range: 'Lat [8.4, 37.6], Lon [68.7, 97.25]',
        nullability: '0.0% (RTK Validated · Zero Drop)'
      };
    }
    return {
      description: 'Continuous normalized telemetry / behavioral feature signal',
      range: '[-2.50, +3.80] Standardized N(0, 1)',
      nullability: '0.0% (Imputed via MissForest)'
    };
  };

  return (
    <div className="bg-slate-100 hover:bg-slate-200/70 p-6 rounded-xl border border-slate-300 space-y-6 transition-all duration-200 shadow-xs">
      {/* 1. Header Row */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-slate-300">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono text-rose-900 mb-1">
            <span className="bg-rose-100 px-2 py-0.5 rounded font-bold border border-rose-200">
              [Synthetic Dataset, ML Complexity &amp; Preprocessing Architecture]
            </span>
            <span className="text-blue-900 font-semibold">
              Native Modality: {SYNTHETIC_DATA_TYPE_META[profile.defaultDataType].shortLabel}
            </span>
            <span>·</span>
            <span className="text-emerald-900 font-semibold">{profile.noiseSigma}</span>
            <span>·</span>
            <span className="text-purple-900 font-semibold">1.50B Cohort (Δ ±2.4%/yr)</span>
          </div>
          <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <Database className="w-5 h-5 text-blue-900" />
            <span>{profile.datasetName}</span>
          </h3>
          <p className="text-xs text-slate-600 mt-0.5">
            Comprehensive dataset provenance, sample column schemas, sample row inspections, domain data cleaning hurdles, and ML technique asymptotic complexity.
          </p>
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

      {/* 2. Interactive Modality & Distribution Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-white p-3.5 rounded-xl border border-slate-300 space-y-1.5">
          <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
            <SlidersHorizontal className="w-3.5 h-3.5 text-blue-900" />
            <span>Input Modality / Tensor Representation</span>
          </label>
          <select
            value={selectedDataType}
            onChange={(e) => setSelectedDataType(e.target.value as SyntheticDataType)}
            className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900"
          >
            {(Object.keys(SYNTHETIC_DATA_TYPE_META) as SyntheticDataType[]).map((dt) => (
              <option key={dt} value={dt}>
                {SYNTHETIC_DATA_TYPE_META[dt].label}
                {dt === profile.defaultDataType ? ' (Project Native)' : ''}
              </option>
            ))}
          </select>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-300 space-y-1.5">
          <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
            <Table2 className="w-3.5 h-3.5 text-rose-900" />
            <span>Synthetic Cohort Distribution Profile</span>
          </label>
          <select
            value={selectedVariant}
            onChange={(e) => setSelectedVariant(e.target.value as SyntheticVariant)}
            className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900"
          >
            {(Object.keys(SYNTHETIC_VARIANT_META) as SyntheticVariant[]).map((v) => (
              <option key={v} value={v}>
                {SYNTHETIC_VARIANT_META[v].label}
              </option>
            ))}
          </select>
        </div>

        <div className="bg-white p-3.5 rounded-xl border border-slate-300 space-y-1.5">
          <label className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-purple-900" />
            <span>Active ML Technique Architecture</span>
          </label>
          <select
            value={selectedTechnique}
            onChange={(e) => setSelectedTechnique(e.target.value as MlTechniqueId)}
            className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900"
          >
            {(Object.keys(ML_TECHNIQUE_META) as MlTechniqueId[]).map((tech) => (
              <option key={tech} value={tech}>
                {ML_TECHNIQUE_META[tech].name}
                {tech === defaultTechnique ? ' (Recommended)' : ''}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 3. SOURCE OF THE DATA & SYNTHETIC PROVENANCE */}
      <div className="bg-white rounded-xl border border-slate-300 p-5 space-y-3.5 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2.5 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-blue-100 border border-blue-200 flex items-center justify-center">
              <ShieldCheck className="w-4 h-4 text-blue-900" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-900">
                Source of Data &amp; Synthetic Generator Provenance
              </h4>
              <p className="text-[11px] text-slate-500 font-mono">
                Authoritative real-world benchmark citations, generative modeling technique &amp; differential privacy bounds
              </p>
            </div>
          </div>
          {dataSourceInfo.sourceUrl && (
            <a
              href={dataSourceInfo.sourceUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 text-xs font-semibold text-blue-900 hover:text-blue-700 bg-blue-50 hover:bg-blue-100/70 border border-blue-200 px-2.5 py-1 rounded-md transition-colors"
            >
              <span>Benchmark Reference</span>
              <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider block font-semibold mb-1">
              Benchmark Origin &amp; Ground Truth
            </span>
            <div className="font-semibold text-slate-900 text-xs leading-snug">
              {dataSourceInfo.benchmarkOrigin}
            </div>
            <div className="text-[11px] text-slate-500 mt-1 italic font-sans leading-tight">
              {dataSourceInfo.benchmarkCitation}
            </div>
          </div>

          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider block font-semibold mb-1">
              Synthetic Generative Engine
            </span>
            <div className="font-semibold text-blue-950 text-xs leading-snug">
              {dataSourceInfo.syntheticGenerator}
            </div>
            <div className="text-[11px] text-slate-600 mt-1 leading-tight">
              {dataSourceInfo.generationTechnique}
            </div>
          </div>

          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider block font-semibold mb-1">
              Privacy &amp; Statistical Fidelity
            </span>
            <div className="font-mono font-bold text-emerald-900 text-xs">
              {dataSourceInfo.fidelityScore}
            </div>
            <div className="text-[11px] font-mono text-slate-700 mt-1 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
              {dataSourceInfo.differentialPrivacy}
            </div>
          </div>

          <div className="bg-slate-50 p-3 rounded-lg border border-slate-200">
            <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider block font-semibold mb-1">
              License &amp; Cohort Scale
            </span>
            <div className="font-semibold text-slate-900 text-xs">
              {dataSourceInfo.dataLicense}
            </div>
            <div className="text-[11px] font-mono text-purple-900 mt-1 font-semibold">
              {dataSourceInfo.realWorldCohort}
            </div>
          </div>
        </div>
      </div>

      {/* 4. COMPLEXITY OF THE ML TECHNIQUE */}
      <div className="bg-white rounded-xl border border-slate-300 p-5 space-y-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2.5 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-purple-100 border border-purple-200 flex items-center justify-center">
              <Cpu className="w-4 h-4 text-purple-900" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <span>Computational Complexity of ML Technique:</span>
                <span className="text-purple-900 font-mono text-xs bg-purple-50 px-2 py-0.5 rounded border border-purple-200">
                  {mlComplexity.techniqueName}
                </span>
              </h4>
              <p className="text-[11px] text-slate-500 font-mono">
                Asymptotic time O(·) &amp; space O(·) complexity, forward pass FLOPs, parameter scaling, and hardware acceleration
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[11px] font-mono font-bold text-slate-700 bg-slate-100 px-2.5 py-1 rounded border border-slate-300">
              SLA: {mlComplexity.slaTarget}
            </span>
          </div>
        </div>

        {/* 4 Asymptotic Metric Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
          {/* Training Time */}
          <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider font-semibold">
                Training Time Complexity
              </span>
              <Clock className="w-3.5 h-3.5 text-blue-900" />
            </div>
            <div className="font-mono text-sm font-bold text-blue-900 bg-blue-50/80 px-2 py-1 rounded border border-blue-200 break-all">
              {mlComplexity.trainingTimeComplexity}
            </div>
            <p className="text-[11px] text-slate-600 leading-tight">
              {mlComplexity.trainingTimeExplanation}
            </p>
          </div>

          {/* Inference Latency */}
          <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider font-semibold">
                Inference Latency Complexity
              </span>
              <Zap className="w-3.5 h-3.5 text-amber-900" />
            </div>
            <div className="font-mono text-sm font-bold text-amber-900 bg-amber-50/80 px-2 py-1 rounded border border-amber-200 break-all">
              {mlComplexity.inferenceTimeComplexity}
            </div>
            <p className="text-[11px] text-slate-600 leading-tight">
              {mlComplexity.inferenceLatencyExplanation}
            </p>
          </div>

          {/* Space / Memory Complexity */}
          <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider font-semibold">
                Space / Memory Complexity
              </span>
              <HardDrive className="w-3.5 h-3.5 text-emerald-900" />
            </div>
            <div className="font-mono text-sm font-bold text-emerald-900 bg-emerald-50/80 px-2 py-1 rounded border border-emerald-200 break-all">
              {mlComplexity.spaceComplexity}
            </div>
            <p className="text-[11px] text-slate-600 leading-tight">
              {mlComplexity.memoryFootprintExplanation}
            </p>
          </div>

          {/* Parameters & FLOPs */}
          <div className="bg-slate-50 p-3.5 rounded-xl border border-slate-200 space-y-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider font-semibold">
                Compute &amp; FLOPs Scale
              </span>
              <Activity className="w-3.5 h-3.5 text-rose-900" />
            </div>
            <div className="font-mono text-sm font-bold text-rose-900 bg-rose-50/80 px-2 py-1 rounded border border-rose-200">
              {mlComplexity.flopsPerSample}
            </div>
            <div className="text-[11px] font-semibold text-slate-900">
              {mlComplexity.parameterScale}
            </div>
            <div className="text-[10px] font-mono text-slate-500">
              {ML_TECHNIQUE_META[selectedTechnique].objectiveFormula}
            </div>
          </div>
        </div>

        {/* Scaling Bottleneck & Hardware Acceleration Footnote */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs pt-1">
          <div className="bg-amber-50/60 p-3 rounded-lg border border-amber-200 text-slate-800 space-y-1">
            <div className="font-bold text-amber-950 flex items-center gap-1.5 text-xs">
              <AlertTriangle className="w-3.5 h-3.5 text-amber-900" />
              <span>Primary Algorithmic Scaling Bottleneck &amp; Mitigation</span>
            </div>
            <p className="text-[11px] text-slate-700 leading-relaxed">
              {mlComplexity.scalingBottleneck}
            </p>
          </div>

          <div className="bg-blue-50/60 p-3 rounded-lg border border-blue-200 text-slate-800 space-y-1">
            <div className="font-bold text-blue-950 flex items-center gap-1.5 text-xs">
              <Cpu className="w-3.5 h-3.5 text-blue-900" />
              <span>Target Acceleration Hardware &amp; Kernel Stack</span>
            </div>
            <p className="text-[11px] text-slate-700 leading-relaxed font-mono">
              {mlComplexity.accelerationHardware}
            </p>
          </div>
        </div>
      </div>

      {/* 5. DATA CLEANING & PREPROCESSING CHALLENGES */}
      <div className="bg-white rounded-xl border border-slate-300 p-5 space-y-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2.5 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-rose-100 border border-rose-200 flex items-center justify-center">
              <AlertTriangle className="w-4 h-4 text-rose-900" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-900">
                Production Data Cleaning &amp; Preprocessing Challenges
              </h4>
              <p className="text-[11px] text-slate-500 font-mono">
                Domain-specific real-world hurdles: sensor degradation, severe class imbalance, high-cardinality PII masking, and temporal drift
              </p>
            </div>
          </div>
          <span className="text-xs font-mono text-slate-600 bg-slate-100 px-2.5 py-1 rounded border border-slate-200">
            4 Critical Production Gates
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
          {dataCleaningChallenges.map((challenge, idx) => {
            const severityColor =
              challenge.severity === 'Critical'
                ? 'bg-rose-100 text-rose-900 border-rose-300'
                : challenge.severity === 'High'
                ? 'bg-amber-100 text-amber-900 border-amber-300'
                : 'bg-blue-100 text-blue-900 border-blue-300';

            return (
              <div
                key={idx}
                className="bg-slate-50 p-4 rounded-xl border border-slate-200 space-y-2.5 text-xs hover:border-slate-300 transition-colors"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="space-y-0.5">
                    <span className="text-[10px] font-mono text-slate-500 uppercase font-semibold block">
                      {challenge.category}
                    </span>
                    <h5 className="font-bold text-slate-900 text-xs leading-snug">
                      {challenge.title}
                    </h5>
                  </div>
                  <span className={`text-[10px] font-mono font-bold px-2 py-0.5 rounded border shrink-0 ${severityColor}`}>
                    {challenge.severity}
                  </span>
                </div>

                <div className="space-y-1.5 pt-1 border-t border-slate-200 text-[11px]">
                  <div>
                    <span className="font-bold text-slate-800">Root Cause: </span>
                    <span className="text-slate-600">{challenge.rootCause}</span>
                  </div>
                  <div>
                    <span className="font-bold text-slate-800">Operational Impact: </span>
                    <span className="text-slate-600">{challenge.impactArea}</span>
                  </div>
                  <div className="bg-white p-2 rounded-lg border border-slate-200 space-y-0.5">
                    <div className="font-bold text-emerald-950 flex items-center gap-1 text-[11px]">
                      <CheckCircle2 className="w-3 h-3 text-emerald-800" />
                      <span>Production Remediation Algorithm:</span>
                    </div>
                    <p className="text-[11px] text-slate-700 font-mono">
                      {challenge.remediation}
                    </p>
                  </div>
                </div>

                <div className="flex items-center justify-between gap-2 pt-1 text-[10px] font-mono text-slate-500 border-t border-slate-200">
                  <span className="bg-slate-200/80 px-2 py-0.5 rounded text-slate-800 font-semibold truncate">
                    {challenge.pipelineStage}
                  </span>
                  <span className="text-emerald-900 font-bold shrink-0">
                    {challenge.retentionLift}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* 6. SAMPLE SYNTHETIC DATA COLUMNS & 1-2 SAMPLE ROWS DEEP INSPECTION */}
      <div className="bg-white rounded-xl border border-slate-300 p-5 space-y-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2.5 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-lg bg-emerald-100 border border-emerald-200 flex items-center justify-center">
              <FileText className="w-4 h-4 text-emerald-900" />
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-900">
                Sample Synthetic Data Columns &amp; Sample Rows Inspection
              </h4>
              <p className="text-[11px] text-slate-500 font-mono">
                Column-by-column schema specifications with 1–2 concrete sample records for the active modality
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="inline-flex rounded-lg border border-slate-300 bg-slate-100 p-0.5 text-xs font-semibold">
              <button
                type="button"
                onClick={() => setActiveSampleRowTab('both')}
                className={`px-2.5 py-1 rounded-md transition-colors ${
                  activeSampleRowTab === 'both' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Rows 1 &amp; 2
              </button>
              <button
                type="button"
                onClick={() => setActiveSampleRowTab('row1')}
                className={`px-2.5 py-1 rounded-md transition-colors ${
                  activeSampleRowTab === 'row1' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Sample Row 1
              </button>
              <button
                type="button"
                onClick={() => setActiveSampleRowTab('row2')}
                className={`px-2.5 py-1 rounded-md transition-colors ${
                  activeSampleRowTab === 'row2' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Sample Row 2
              </button>
            </div>

            <button
              type="button"
              onClick={() => handleCopyJson([row1, row2], 'both_rows')}
              className="px-2.5 py-1 rounded-md bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-300 text-xs font-semibold flex items-center gap-1 transition-colors"
            >
              {copiedKey === 'both_rows' ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-700" />
                  <span className="text-emerald-700">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-slate-500" />
                  <span>Copy Both (JSON)</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Detailed Columns Table */}
        <div className="overflow-x-auto rounded-lg border border-slate-200">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-100 border-b border-slate-200 text-slate-700 font-mono text-[11px]">
                <th className="py-2.5 px-3 font-semibold">Column Name</th>
                <th className="py-2.5 px-3 font-semibold">Data Type (Dtype)</th>
                <th className="py-2.5 px-3 font-semibold">Role</th>
                <th className="py-2.5 px-3 font-semibold">Semantic Meaning</th>
                <th className="py-2.5 px-3 font-semibold">Value Domain / Range</th>
                <th className="py-2.5 px-3 font-semibold">Missingness Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 font-mono">
              {profile.columns.map((col) => {
                const details = getColumnSemanticDetails(col);
                const roleBadgeClass =
                  col.role === 'target'
                    ? 'bg-emerald-100 text-emerald-900 border-emerald-300'
                    : col.role === 'tensor'
                    ? 'bg-blue-100 text-blue-900 border-blue-300'
                    : col.role === 'metadata'
                    ? 'bg-purple-100 text-purple-900 border-purple-300'
                    : 'bg-slate-100 text-slate-800 border-slate-300';

                return (
                  <tr key={col.name} className="hover:bg-slate-50 transition-colors">
                    <td className="py-2 px-3 font-bold text-slate-900 whitespace-nowrap">
                      {col.name}
                    </td>
                    <td className="py-2 px-3 text-blue-900 font-semibold whitespace-nowrap">
                      {col.dtype}
                    </td>
                    <td className="py-2 px-3 whitespace-nowrap">
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded border uppercase ${roleBadgeClass}`}>
                        {col.role}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-slate-700 font-sans text-xs">
                      {details.description}
                    </td>
                    <td className="py-2 px-3 text-slate-600 text-[11px] whitespace-nowrap">
                      {details.range}
                    </td>
                    <td className="py-2 px-3 text-emerald-900 font-semibold text-[11px] whitespace-nowrap">
                      {details.nullability}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* 1 or 2 Sample Rows Deep Inspection Card */}
        <div className="space-y-3 pt-1">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-blue-900" />
              <span>Sample Rows Value Comparison (Record #1 vs Record #2)</span>
            </span>
            <span className="text-[11px] font-mono text-slate-500">
              Live Seed {seed} · {SYNTHETIC_DATA_TYPE_META[selectedDataType].shortLabel}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
            {/* Sample Row 1 */}
            {(activeSampleRowTab === 'row1' || activeSampleRowTab === 'both') && (
              <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 space-y-2.5">
                <div className="flex items-center justify-between pb-2 border-b border-slate-200">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-blue-900 text-white font-mono text-[10px] font-bold flex items-center justify-center">
                      1
                    </span>
                    <span className="font-bold text-xs text-slate-900">
                      Sample Row #01 (Baseline / Nominal)
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleCopyJson(row1, 'row1')}
                    className="p-1 rounded hover:bg-slate-200 text-slate-600 transition-colors"
                    title="Copy Row 1 JSON"
                  >
                    {copiedKey === 'row1' ? (
                      <Check className="w-3.5 h-3.5 text-emerald-700" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>

                <div className="space-y-1.5 text-xs font-mono">
                  {profile.columns.map((col) => (
                    <div
                      key={col.name}
                      className="flex items-start justify-between gap-2 py-1 px-2 rounded bg-white border border-slate-200"
                    >
                      <span className="text-slate-500 font-semibold text-[11px] shrink-0">
                        {col.name}:
                      </span>
                      <span
                        className={`text-right break-all text-[11px] font-bold ${
                          col.role === 'target'
                            ? 'text-emerald-900'
                            : col.role === 'tensor'
                            ? 'text-blue-900'
                            : 'text-slate-900'
                        }`}
                      >
                        {String(row1[col.name] ?? '—')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Sample Row 2 */}
            {(activeSampleRowTab === 'row2' || activeSampleRowTab === 'both') && (
              <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 space-y-2.5">
                <div className="flex items-center justify-between pb-2 border-b border-slate-200">
                  <div className="flex items-center gap-2">
                    <span className="w-5 h-5 rounded-full bg-amber-900 text-white font-mono text-[10px] font-bold flex items-center justify-center">
                      2
                    </span>
                    <span className="font-bold text-xs text-slate-900">
                      Sample Row #02 (Perturbed / Anomaly Alert)
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleCopyJson(row2, 'row2')}
                    className="p-1 rounded hover:bg-slate-200 text-slate-600 transition-colors"
                    title="Copy Row 2 JSON"
                  >
                    {copiedKey === 'row2' ? (
                      <Check className="w-3.5 h-3.5 text-emerald-700" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>

                <div className="space-y-1.5 text-xs font-mono">
                  {profile.columns.map((col) => (
                    <div
                      key={col.name}
                      className="flex items-start justify-between gap-2 py-1 px-2 rounded bg-white border border-slate-200"
                    >
                      <span className="text-slate-500 font-semibold text-[11px] shrink-0">
                        {col.name}:
                      </span>
                      <span
                        className={`text-right break-all text-[11px] font-bold ${
                          col.role === 'target'
                            ? 'text-rose-900'
                            : col.role === 'tensor'
                            ? 'text-purple-900'
                            : 'text-slate-900'
                        }`}
                      >
                        {String(row2[col.name] ?? '—')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 7. Generated Synthetic 5-Row Batch Table (Interactive) */}
      <div className="bg-white rounded-xl border border-slate-300 overflow-hidden shadow-xs">
        <div className="px-4 py-2.5 bg-slate-200/80 border-b border-slate-300 flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-xs">
          <span className="font-bold text-slate-900 flex items-center gap-1.5">
            <Table2 className="w-3.5 h-3.5 text-blue-900" />
            <span>
              Live Synthetic Batch Table (First 5 Records · {SYNTHETIC_DATA_TYPE_META[selectedDataType].shortLabel})
            </span>
          </span>
          <span className="font-mono text-[11px] text-slate-600 truncate">
            {SYNTHETIC_VARIANT_META[selectedVariant].desc}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs font-mono">
            <thead>
              <tr className="bg-slate-100 border-b border-slate-300 text-slate-700">
                <th className="py-2 px-3 font-semibold text-slate-500 w-10">#</th>
                {profile.columns.map((col) => (
                  <th key={col.name} className="py-2 px-3 font-semibold whitespace-nowrap">
                    {col.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {sampleRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-slate-50 transition-colors">
                  <td className="py-2 px-3 text-slate-400 font-bold">{idx + 1}</td>
                  {profile.columns.map((col) => (
                    <td
                      key={col.name}
                      className={`py-2 px-3 whitespace-nowrap ${
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
