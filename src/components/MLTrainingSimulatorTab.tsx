import React, { useState } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  GitBranch,
  Layers,
  Play,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TrendingUp
} from 'lucide-react';
import SimulatorMultiModalCharts from './SimulatorMultiModalCharts';
import { FlagshipProject, SyntheticDataType, SyntheticVariant } from '../types';
import {
  ENCRYPTION_MECHANISM_META,
  EncryptionProfileId,
  LEARNING_RATE_PRESETS,
  ML_TECHNIQUE_META,
  MlTechniqueId,
  PROGRAMMING_ARCHITECTURE_META,
  ProgrammingArchId,
  SYNTHETIC_DATA_TYPE_META,
  SYNTHETIC_VARIANT_META,
  computeHorizonProjections,
  computeLatencyP95P99,
  generateSyntheticSamples,
  getSyntheticDatasetProfile,
  inferBestLearningRate,
  inferBestMlTechnique,
  inferBestProgrammingArch,
  inferProjectDataType
} from '../utils/syntheticDataset';
import { computeProjectEconomics, formatInrCompact } from '../utils/projectEconomics';

export interface SimConfigState {
  projectId: string;
  dataType: SyntheticDataType;
  syntheticVariant: SyntheticVariant;
  sampleCount: number;
  annualDeltaPct: number;
  horizonYears: number;
  seed: number;
  learningRate: string;
  batchSize: string;
  epochs: number;
  optimizer: string;
  architecture: string;
  programmingArch: ProgrammingArchId;
  mlTechnique: MlTechniqueId;
  encryptionProfile: EncryptionProfileId;
}

export interface TrainingLogEntry {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAcc: number;
  sigma: number;
  throughput: number;
  latencyMs: number;
  p99LatencyMs: number;
  dataTypeLabel: string;
  progArchLabel: string;
  mlTechLabel: string;
}

interface MLTrainingSimulatorTabProps {
  allProjects: FlagshipProject[];
  researchProjects: FlagshipProject[];
  repoProjects: FlagshipProject[];
  starredRepoIds: string[];
  simConfig: SimConfigState;
  setSimConfig: React.Dispatch<React.SetStateAction<SimConfigState>>;
  isTraining: boolean;
  trainingEpoch: number;
  trainingLogs: TrainingLogEntry[];
  onStartTraining: (override?: SimConfigState) => void;
}

type BenchmarkView =
  | 'datatype'
  | 'distribution'
  | 'horizon'
  | 'architecture'
  | 'ml_technique'
  | 'learning_rates'
  | 'encryption';

export default function MLTrainingSimulatorTab({
  allProjects,
  researchProjects,
  repoProjects,
  starredRepoIds,
  simConfig,
  setSimConfig,
  isTraining,
  trainingEpoch,
  trainingLogs,
  onStartTraining
}: MLTrainingSimulatorTabProps) {
  const [activeBenchTab, setActiveBenchTab] = useState<BenchmarkView>('datatype');

  const activeSimProject =
    allProjects.find((p) => p.id === simConfig.projectId) || allProjects[0];
  const activeProfile = getSyntheticDatasetProfile(
    activeSimProject,
    simConfig.dataType,
    simConfig.syntheticVariant
  );
  const previewSamples = generateSyntheticSamples(
    activeSimProject,
    simConfig.dataType,
    simConfig.syntheticVariant,
    4,
    simConfig.seed
  );

  const isCrossModality = simConfig.dataType !== activeProfile.defaultDataType;
  const bestProgArch = inferBestProgrammingArch(activeSimProject);
  const bestMlTech = inferBestMlTechnique(activeSimProject);
  const bestLrPreset = inferBestLearningRate(activeSimProject);

  const activeTypeMeta = SYNTHETIC_DATA_TYPE_META[simConfig.dataType];
  const activeVarMeta = SYNTHETIC_VARIANT_META[simConfig.syntheticVariant];
  const activeProgMeta = PROGRAMMING_ARCHITECTURE_META[simConfig.programmingArch];
  const activeMlMeta = ML_TECHNIQUE_META[simConfig.mlTechnique];
  const activeEncMeta = ENCRYPTION_MECHANISM_META[simConfig.encryptionProfile];
  const activeLrPreset =
    LEARNING_RATE_PRESETS.find((p) => Number(simConfig.learningRate) === p.numericLr) ||
    bestLrPreset;

  const baseProjAcc =
    activeSimProject.epochHistory[activeSimProject.epochHistory.length - 1]?.accuracy || 0.945;

  const latestLog = trainingLogs.length > 0 ? trainingLogs[trainingLogs.length - 1] : null;
  const progressPct = Math.min(100, Math.round((trainingEpoch / Math.max(1, simConfig.epochs)) * 100));

  const projectedAcc = Math.min(
    0.996,
    Math.max(
      0.76,
      baseProjAcc +
        activeTypeMeta.accBias +
        activeVarMeta.accOffset +
        activeMlMeta.accModifier -
        (isCrossModality ? 0.028 : 0)
    )
  );
  const activeLatMetrics = computeLatencyP95P99(
    simConfig.programmingArch,
    simConfig.dataType,
    simConfig.mlTechnique,
    simConfig.encryptionProfile
  );
  const projectedLatencyMs = activeLatMetrics.p95Ms;
  const projectedP99LatencyMs = activeLatMetrics.p99Ms;
  const projectedThroughput = Math.round(
    activeTypeMeta.throughputBase * activeProgMeta.throughputMultiplier * activeEncMeta.throughputMultiplier
  );

  const displayAcc = latestLog ? latestLog.accuracy : projectedAcc;
  const displayValAcc = latestLog
    ? latestLog.valAcc
    : Math.max(0.72, projectedAcc - 0.011 * activeVarMeta.noiseMultiplier);
  const displayLoss = latestLog
    ? latestLog.loss
    : Number(
        (
          (activeSimProject.epochHistory[activeSimProject.epochHistory.length - 1]?.loss || 0.14) *
          activeTypeMeta.lossMult *
          activeVarMeta.lossPenalty *
          activeMlMeta.lossModifier
        ).toFixed(4)
      );
  const displayThroughput = latestLog ? latestLog.throughput : projectedThroughput;
  const displayLatency = latestLog ? latestLog.latencyMs : projectedLatencyMs;
  const displayP99Latency = latestLog ? latestLog.p99LatencyMs : projectedP99LatencyMs;
  const activeProjectEconomics = computeProjectEconomics(activeSimProject);

  const horizonPoints = computeHorizonProjections(
    1500,
    simConfig.annualDeltaPct,
    simConfig.horizonYears,
    displayValAcc
  );
  const year20Point = horizonPoints[horizonPoints.length - 1];

  const handleSelectTargetProject = (projectId: string) => {
    const nextProj = allProjects.find((p) => p.id === projectId) || allProjects[0];
    const autoDataType = inferProjectDataType(nextProj);
    const autoProgArch = inferBestProgrammingArch(nextProj);
    const autoMlTech = inferBestMlTechnique(nextProj);
    const autoLr = inferBestLearningRate(nextProj);
    const nextConfig: SimConfigState = {
      ...simConfig,
      projectId: nextProj.id,
      architecture: nextProj.techStack.slice(0, 2).join(' + '),
      dataType: autoDataType,
      programmingArch: autoProgArch,
      mlTechnique: autoMlTech,
      learningRate: autoLr.lrValue
    };
    setSimConfig(nextConfig);
    onStartTraining(nextConfig);
  };

  const applyAndRun = (patch: Partial<SimConfigState>) => {
    const nextConfig: SimConfigState = { ...simConfig, ...patch };
    setSimConfig(nextConfig);
    onStartTraining(nextConfig);
  };

  // 5-stage simulation pipeline state
  const stageIdx = isTraining
    ? Math.min(4, Math.floor((trainingEpoch / Math.max(1, simConfig.epochs)) * 5))
    : trainingLogs.length > 0
      ? 5
      : 0;

  const pipelineStages = [
    '01. 1.50B Sharded Ingestion',
    '02. Zero-Copy & TLS Init',
    '03. Forward/Loss Opt',
    '04. 20-Yr Δ Drift Calib',
    '05. SLA & Crypto Audit'
  ];

  const slaPass = displayLatency <= 45;
  const generalizationGapPct = Number(((displayAcc - displayValAcc) * 100).toFixed(2));

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-indigo-800 font-semibold mb-1.5">
            <span className="bg-indigo-100 px-2 py-0.5 rounded border border-indigo-200">
              [Research &amp; Scale Simulation Engine]
            </span>
            <span>1.50B Population Baseline</span>
            <span>·</span>
            <span>20-Year Horizon (2026–2046) @ Δ ±{simConfig.annualDeltaPct}%/yr</span>
          </div>
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
            ML Simulator
          </h2>
          <p className="text-slate-600 text-sm mt-1 max-w-4xl">
            Selecting a Target Project automatically locks its optimal Data Type, Programming Architecture, ML Technique, and Encryption Profile and runs live 1.50-billion-scale simulation epochs.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <span className="px-3 py-1.5 rounded-lg bg-white border border-slate-300 text-xs font-mono font-semibold text-slate-800 shadow-xs">
            Cohort: {(simConfig.sampleCount / 1_000_000_000).toFixed(2)}B Records
          </span>
          <span className="px-3 py-1.5 rounded-lg bg-emerald-50 border border-emerald-300 text-xs font-mono font-semibold text-emerald-900 shadow-xs">
            CIDR SLA Target: &lt; 200 ms (NavIC + PQC Verified)
          </span>
        </div>
      </div>

      {/* Main Split: Left Configuration Controls + Right Live Simulation & Outcomes */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Simulator Controls Panel */}
        <div className="lg:col-span-4 bg-slate-100 border border-slate-300 rounded-2xl p-5 space-y-4 shadow-xs">
          <div className="flex items-center justify-between pb-3 border-b border-slate-300">
            <h3 className="text-sm font-bold text-slate-900">
              Experiment Configuration
            </h3>
            <span className="text-[11px] font-mono text-indigo-800 bg-indigo-100 px-2 py-0.5 rounded font-semibold">
              Auto-Select Active
            </span>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Target Project (Auto-Selects Optimal Data Type &amp; Stack)
            </label>
            <select
              value={simConfig.projectId}
              onChange={(e) => handleSelectTargetProject(e.target.value)}
              className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2.5 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
            >
              <optgroup label="Research Projects (Flagship + Starred)">
                {researchProjects.map((p) => (
                  <option key={p.id} value={p.id}>
                    {starredRepoIds.includes(p.id) ? '★ ' : ''}
                    {p.title}
                  </option>
                ))}
              </optgroup>
              <optgroup label="ML Case Studies Repository">
                {repoProjects.map((p) => (
                  <option key={p.id} value={p.id}>
                    [{p.sourceName || 'Case Study'}] {p.title}
                  </option>
                ))}
              </optgroup>
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Data Type{' '}
              <span className="text-[11px] font-mono text-indigo-700">
                {isCrossModality ? '(Cross-Modality Stress)' : '(Auto-Matched ★)'}
              </span>
            </label>
            <select
              value={simConfig.dataType}
              onChange={(e) =>
                applyAndRun({ dataType: e.target.value as SyntheticDataType })
              }
              className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
            >
              {(Object.keys(SYNTHETIC_DATA_TYPE_META) as SyntheticDataType[]).map((dt) => (
                <option key={dt} value={dt}>
                  {SYNTHETIC_DATA_TYPE_META[dt].label}
                  {dt === activeProfile.defaultDataType ? ' ★ Recommended' : ''}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Synthetic Dataset Distribution (1.50B / 20-Yr)
            </label>
            <select
              value={simConfig.syntheticVariant}
              onChange={(e) => {
                const variant = e.target.value as SyntheticVariant;
                applyAndRun({
                  syntheticVariant: variant,
                  sampleCount: variant === 'scaled_100k' ? 2410000000 : 1500000000
                });
              }}
              className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
            >
              {(Object.keys(SYNTHETIC_VARIANT_META) as SyntheticVariant[]).map((v) => (
                <option key={v} value={v}>
                  {SYNTHETIC_VARIANT_META[v].label}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                1.50B Horizon Scale
              </label>
              <select
                value={simConfig.sampleCount}
                onChange={(e) =>
                  applyAndRun({ sampleCount: Number(e.target.value) })
                }
                className="w-full bg-white border border-slate-300 rounded-xl px-2.5 py-2 text-xs text-slate-900 font-mono focus:outline-none focus:border-blue-900 shadow-xs"
              >
                <option value={1500000000}>1,500,000,000 (1.50B Yr 0 Base)</option>
                <option value={1690000000}>1,690,000,000 (1.69B Yr +5)</option>
                <option value={1900000000}>1,900,000,000 (1.90B Yr +10)</option>
                <option value={2140000000}>2,140,000,000 (2.14B Yr +15)</option>
                <option value={2410000000}>2,410,000,000 (2.41B Yr +20)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Annual Δ Growth Rate
              </label>
              <select
                value={simConfig.annualDeltaPct}
                onChange={(e) =>
                  setSimConfig({ ...simConfig, annualDeltaPct: Number(e.target.value) })
                }
                className="w-full bg-white border border-slate-300 rounded-xl px-2.5 py-2 text-xs text-slate-900 font-mono focus:outline-none focus:border-blue-900 shadow-xs"
              >
                <option value={1.2}>Δ ±1.2% / yr (Low)</option>
                <option value={2.4}>Δ ±2.4% / yr (Standard)</option>
                <option value={3.5}>Δ ±3.5% / yr (High Surge)</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Encryption (Data at Rest &amp; In Transit){' '}
              <span className="text-[11px] font-mono text-emerald-700">
                {simConfig.encryptionProfile === 'pqc_hybrid_vault' ? '(20-Yr PQC ★)' : ''}
              </span>
            </label>
            <select
              value={simConfig.encryptionProfile}
              onChange={(e) =>
                applyAndRun({ encryptionProfile: e.target.value as EncryptionProfileId })
              }
              className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
            >
              {(Object.keys(ENCRYPTION_MECHANISM_META) as EncryptionProfileId[]).map((encId) => (
                <option key={encId} value={encId}>
                  {ENCRYPTION_MECHANISM_META[encId].name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Programming Architecture{' '}
              <span className="text-[11px] font-mono text-indigo-700">
                {simConfig.programmingArch === bestProgArch ? '(Optimal ★)' : ''}
              </span>
            </label>
            <select
              value={simConfig.programmingArch}
              onChange={(e) =>
                applyAndRun({ programmingArch: e.target.value as ProgrammingArchId })
              }
              className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
            >
              {(Object.keys(PROGRAMMING_ARCHITECTURE_META) as ProgrammingArchId[]).map((archId) => (
                <option key={archId} value={archId}>
                  {PROGRAMMING_ARCHITECTURE_META[archId].name}
                  {archId === bestProgArch ? ' ★ Recommended' : ''}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              ML Learning Technique{' '}
              <span className="text-[11px] font-mono text-indigo-700">
                {simConfig.mlTechnique === bestMlTech ? '(Optimal ★)' : ''}
              </span>
            </label>
            <select
              value={simConfig.mlTechnique}
              onChange={(e) =>
                applyAndRun({ mlTechnique: e.target.value as MlTechniqueId })
              }
              className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
            >
              {(Object.keys(ML_TECHNIQUE_META) as MlTechniqueId[]).map((techId) => (
                <option key={techId} value={techId}>
                  {ML_TECHNIQUE_META[techId].name}
                  {techId === bestMlTech ? ' ★ Recommended' : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Optimizer</label>
              <select
                value={simConfig.optimizer}
                onChange={(e) => setSimConfig({ ...simConfig, optimizer: e.target.value })}
                className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
              >
                <option value="AdamW">AdamW (Decoupled)</option>
                <option value="Lion">Lion (Sign-Momentum)</option>
                <option value="SGD">SGD Nesterov</option>
              </select>
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1">Batch Size</label>
              <select
                value={simConfig.batchSize}
                onChange={(e) => setSimConfig({ ...simConfig, batchSize: e.target.value })}
                className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs text-slate-900 font-mono focus:outline-none focus:border-blue-900 shadow-xs"
              >
                <option value="16">16 (Low VRAM)</option>
                <option value="32">32 (Balanced)</option>
                <option value="64">64 (Throughput)</option>
                <option value="128">128 (H100 / FP16)</option>
              </select>
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">
              Suggested Learning Rate &amp; Schedule{' '}
              <span className="text-[11px] font-mono text-indigo-700">
                {Number(simConfig.learningRate) === bestLrPreset.numericLr ? '(Optimal ★)' : ''}
              </span>
            </label>
            <select
              value={
                LEARNING_RATE_PRESETS.some((p) => Number(simConfig.learningRate) === p.numericLr)
                  ? simConfig.learningRate
                  : 'custom'
              }
              onChange={(e) => {
                if (e.target.value !== 'custom') {
                  applyAndRun({ learningRate: e.target.value });
                }
              }}
              className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs mb-2"
            >
              {LEARNING_RATE_PRESETS.map((lr) => (
                <option key={lr.id} value={lr.lrValue}>
                  {lr.label}
                  {lr.id === bestLrPreset.id ? ' ★ Recommended' : ''}
                </option>
              ))}
              {!LEARNING_RATE_PRESETS.some((p) => Number(simConfig.learningRate) === p.numericLr) && (
                <option value="custom">Custom LR ({simConfig.learningRate})</option>
              )}
            </select>

            <div className="grid grid-cols-2 gap-3 items-center">
              <div>
                <label className="block text-[11px] font-semibold text-slate-600 mb-1">Custom LR Override</label>
                <input
                  type="text"
                  value={simConfig.learningRate}
                  onChange={(e) => setSimConfig({ ...simConfig, learningRate: e.target.value })}
                  className="w-full bg-white border border-slate-300 rounded-xl px-3 py-1.5 text-xs text-slate-900 font-mono focus:outline-none focus:border-blue-900 shadow-xs"
                />
              </div>
              <div>
                <label className="block text-[11px] font-semibold text-slate-600 mb-1">
                  Epochs ({simConfig.epochs})
                </label>
                <input
                  type="range"
                  min={6}
                  max={20}
                  step={2}
                  value={simConfig.epochs}
                  onChange={(e) => setSimConfig({ ...simConfig, epochs: Number(e.target.value) })}
                  className="w-full accent-blue-900"
                />
              </div>
            </div>
            <p className="text-[11px] font-mono text-slate-600 mt-1.5">
              Schedule: <span className="font-semibold text-indigo-900">{activeLrPreset.schedule}</span>
            </p>
          </div>

          <button
            onClick={() => onStartTraining()}
            disabled={isTraining}
            className="w-full py-3 bg-blue-900 hover:bg-blue-800 disabled:opacity-50 text-white font-semibold rounded-xl transition-all shadow-md flex items-center justify-center gap-2 text-sm"
          >
            {isTraining ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>
                  Simulating Epoch {trainingEpoch}/{simConfig.epochs} ({progressPct}%)...
                </span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                <span>Run 1.50B Simulation ({activeTypeMeta.shortLabel})</span>
              </>
            )}
          </button>
        </div>

        {/* Right Column: Live Progress, Research Outcome & Convergence */}
        <div className="lg:col-span-8 space-y-5">
          {/* Multi-Stage Live Execution Pipeline */}
          <div className="rounded-2xl bg-white border border-slate-300 p-5 shadow-xs space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  Live Simulation Progress &amp; Convergence Pipeline
                </h3>
                <p className="text-xs text-slate-600 mt-0.5">
                  {activeSimProject.title} · {activeProgMeta.shortName} · {activeMlMeta.shortName} · {activeEncMeta.shortName}
                </p>
              </div>
              <span
                className={`px-2.5 py-1 rounded-lg text-xs font-mono font-semibold border ${
                  isTraining
                    ? 'bg-indigo-50 text-indigo-800 border-indigo-200'
                    : trainingLogs.length > 0
                      ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                      : 'bg-slate-100 text-slate-700 border-slate-300'
                }`}
              >
                {isTraining
                  ? `Running (${progressPct}%)`
                  : trainingLogs.length > 0
                    ? 'Convergence Verified (100%)'
                    : 'Ready to Simulate'}
              </span>
            </div>

            {/* 5-Stage Progress Bar */}
            <div className="space-y-2">
              <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-600 transition-all duration-300"
                  style={{ width: `${trainingLogs.length > 0 ? progressPct : 0}%` }}
                />
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                {pipelineStages.map((label, idx) => {
                  const completed = stageIdx > idx || (!isTraining && trainingLogs.length > 0);
                  const active = isTraining && stageIdx === idx;
                  return (
                    <div
                      key={label}
                      className={`px-2.5 py-1.5 rounded-lg border text-[11px] font-mono font-semibold flex items-center gap-1.5 ${
                        completed
                          ? 'bg-emerald-50 border-emerald-300 text-emerald-900'
                          : active
                            ? 'bg-indigo-50 border-indigo-400 text-indigo-900'
                            : 'bg-slate-50 border-slate-200 text-slate-500'
                      }`}
                    >
                      <span
                        className={`w-2 h-2 rounded-full shrink-0 ${
                          completed
                            ? 'bg-emerald-600'
                            : active
                              ? 'bg-indigo-600 animate-pulse'
                              : 'bg-slate-300'
                        }`}
                      />
                      <span className="truncate">{label}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* 6 KPI Cards */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="text-[11px] text-slate-500 block font-mono">Epoch</span>
                <span className="text-lg font-bold font-mono text-slate-900">
                  {trainingEpoch} / {simConfig.epochs}
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="text-[11px] text-slate-500 block font-mono">Final / Live Loss</span>
                <span className="text-lg font-bold font-mono text-blue-900">{displayLoss}</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="text-[11px] text-slate-500 block font-mono">Val Accuracy</span>
                <span className="text-lg font-bold font-mono text-emerald-800">
                  {(displayValAcc * 100).toFixed(2)}%
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="text-[11px] text-slate-500 block font-mono">1.50B Throughput</span>
                <span className="text-lg font-bold font-mono text-indigo-900">
                  {displayThroughput.toLocaleString()}/s
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="text-[11px] text-slate-500 block font-mono">P95 / P99 Latency</span>
                <span className="text-base font-bold font-mono text-slate-900">
                  {displayLatency} / {displayP99Latency} ms
                </span>
              </div>
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200">
                <span className="text-[11px] text-slate-500 block font-mono">VRAM / RAM</span>
                <span className="text-lg font-bold font-mono text-slate-800">
                  {(activeTypeMeta.vramGb + activeProgMeta.memoryFootprintGb).toFixed(2)} GB
                </span>
              </div>
            </div>

            {/* Project CapEx, OpEx YoY & 20-Year TCO Strip */}
            <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                <span className="font-bold text-slate-900">
                  CapEx &amp; 20-Yr OpEx Economics ({activeProjectEconomics.deploymentTier})
                </span>
                <div className="flex flex-wrap items-center gap-3 tabular-nums">
                  <span className="font-bold text-blue-950">
                    CapEx: {formatInrCompact(activeProjectEconomics.totalCapExUsd)}
                  </span>
                  <span className="font-bold text-emerald-900">
                    OpEx YoY (Y1→Y20): {formatInrCompact(activeProjectEconomics.opExYear1Usd)} →{' '}
                    {formatInrCompact(activeProjectEconomics.opExYear20Usd)}
                  </span>
                  <span className="font-bold text-indigo-950">
                    20-Yr OpEx: {formatInrCompact(activeProjectEconomics.totalOpEx20YrUsd)}
                  </span>
                  <span className="font-bold text-rose-950">
                    20-Yr TCO: {formatInrCompact(activeProjectEconomics.totalTco20YrUsd)}
                  </span>
                </div>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 text-[11px] font-mono tabular-nums">
                {activeProjectEconomics.opExCategories.map((cat) => (
                  <div key={cat.key} className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5">
                    <span className="text-slate-500 block text-[10px] truncate">{cat.shortLabel}</span>
                    <span className="font-bold text-slate-900">
                      {formatInrCompact(cat.total20YrUsd)}
                    </span>
                    <span className="text-slate-500 text-[9.5px] block">
                      Y1: {formatInrCompact(cat.year1Usd)} ({cat.sharePct}%)
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Convergence Curve Card */}
          <div className="rounded-2xl bg-slate-100 border border-slate-300 p-5 shadow-xs space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h4 className="text-sm font-bold text-slate-900">
                  Optimization Trajectory ({activeTypeMeta.shortLabel} · {activeVarMeta.skewProfile})
                </h4>
                <p className="text-[11px] font-mono text-slate-600">
                  Loss Objective: {activeMlMeta.objectiveFormula} · Target Shape: {activeProfile.tensorShape}
                </p>
              </div>
              <div className="flex items-center gap-4 text-[11px] font-mono">
                <span className="text-blue-900 font-semibold">— Train Loss</span>
                <span className="text-emerald-800 font-semibold">— Train Acc</span>
                <span className="text-rose-800 font-semibold">-- Val Acc</span>
              </div>
            </div>

            {trainingLogs.length > 1 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-1.5">
                  <div className="text-[11px] font-mono font-semibold text-slate-700">
                    3D Loss &amp; Accuracy Convergence Ribbon
                  </div>
                  <svg viewBox="0 0 380 175" className="w-full h-44 bg-white rounded-xl border border-slate-200">
                    {/* 3D isometric stage floor & back plane */}
                    <polygon points="52,130 66,118 362,118 348,130" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="0.8" />
                    <polygon points="52,22 66,10 66,118 52,130" fill="#f8fafc" stroke="#e2e8f0" strokeWidth="0.8" />
                    <line x1="66" y1="16" x2="362" y2="16" stroke="#e2e8f0" strokeDasharray="3 3" />
                    <line x1="66" y1="67" x2="362" y2="67" stroke="#e2e8f0" strokeDasharray="3 3" />
                    {/* 3D Coordinate Axes (+X, +Y, +Z) */}
                    <line x1="52" y1="130" x2="356" y2="130" stroke="#e11d48" strokeWidth="1.2" />
                    <line x1="52" y1="130" x2="52" y2="18" stroke="#059669" strokeWidth="1.2" />
                    <line x1="52" y1="130" x2="66" y2="118" stroke="#2563eb" strokeWidth="1.2" />
                    <text x="46" y="25" textAnchor="end" fontSize="8.5" fill="#475569" fontFamily="monospace">100%</text>
                    <text x="46" y="75" textAnchor="end" fontSize="8.5" fill="#475569" fontFamily="monospace">50%</text>
                    <text x="46" y="131" textAnchor="end" fontSize="8.5" fill="#475569" fontFamily="monospace">0%</text>
                    <text x="54" y="12" fontSize="7.5" fontWeight="bold" fill="#059669" fontFamily="monospace">+Y: Score/Loss</text>
                    <text x="72" y="114" fontSize="7.5" fontWeight="bold" fill="#2563eb" fontFamily="monospace">+Z: Depth</text>
                    {(() => {
                      const maxL = Math.max(1.5, ...trainingLogs.map((l) => l.loss * 1.05));
                      const stepX = 282 / Math.max(1, trainingLogs.length - 1);
                      const dx = 12;
                      const dy = -10;
                      const lossArr = trainingLogs.map((l, i) => ({
                        x: 56 + i * stepX,
                        y: 128 - (l.loss / maxL) * 98
                      }));
                      const accArr = trainingLogs.map((l, i) => ({
                        x: 56 + i * stepX,
                        y: 128 - l.accuracy * 98
                      }));
                      const valArr = trainingLogs.map((l, i) => ({
                        x: 56 + i * stepX,
                        y: 128 - l.valAcc * 98
                      }));
                      const lossRibbon = [
                        ...lossArr.map((p) => `${p.x + dx},${p.y + dy}`),
                        ...[...lossArr].reverse().map((p) => `${p.x},${p.y}`)
                      ].join(' ');
                      const accRibbon = [
                        ...accArr.map((p) => `${p.x + dx * 0.5},${p.y + dy * 0.5}`),
                        ...[...accArr].reverse().map((p) => `${p.x},${p.y}`)
                      ].join(' ');
                      const accArea = [
                        ...accArr.map((p) => `${p.x},${p.y}`),
                        `${accArr[accArr.length - 1].x},128`,
                        `${accArr[0].x},128`
                      ].join(' ');

                      return (
                        <>
                          <polygon points={accArea} fill="#10b981" fillOpacity="0.1" />
                          <polygon points={lossRibbon} fill="#3b82f6" fillOpacity="0.22" stroke="#1e3a8a" strokeWidth="0.6" />
                          <polygon points={accRibbon} fill="#10b981" fillOpacity="0.25" stroke="#065f46" strokeWidth="0.6" />
                          <polyline
                            fill="none"
                            stroke="#1e3a8a"
                            strokeWidth="2.4"
                            points={lossArr.map((p) => `${p.x + dx},${p.y + dy}`).join(' ')}
                          />
                          <polyline
                            fill="none"
                            stroke="#065f46"
                            strokeWidth="2.5"
                            points={accArr.map((p) => `${p.x},${p.y}`).join(' ')}
                          />
                          <polyline
                            fill="none"
                            stroke="#9f1239"
                            strokeWidth="2"
                            strokeDasharray="5 3"
                            points={valArr.map((p) => `${p.x},${p.y}`).join(' ')}
                          />
                        </>
                      );
                    })()}
                    <text x="12" y="148" fontSize="7.5" fill="#64748b" fontFamily="monospace">(0,0,0)</text>
                    <text x="58" y="160" textAnchor="middle" fontSize="8.5" fill="#475569" fontFamily="monospace">E1</text>
                    <text x="195" y="160" textAnchor="middle" fontSize="7.5" fontWeight="bold" fill="#e11d48" fontFamily="monospace">+X: Epoch</text>
                    <text x="338" y="160" textAnchor="middle" fontSize="8.5" fill="#475569" fontFamily="monospace">
                      E{trainingLogs[trainingLogs.length - 1]?.epoch}
                    </text>
                  </svg>
                </div>

                {/* Live 3D Validation Accuracy Bar Chart */}
                <div className="space-y-1.5">
                  <div className="text-[11px] font-mono font-semibold text-emerald-900">
                    3D Validation Accuracy Bar Chart (% per Epoch)
                  </div>
                  <svg viewBox="0 0 380 175" className="w-full h-44 bg-white rounded-xl border border-slate-200">
                    {/* 3D Isometric base grid */}
                    <polygon points="52,130 62,122 362,122 352,130" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="0.8" />
                    <line x1="62" y1="22" x2="362" y2="22" stroke="#e2e8f0" strokeDasharray="3 3" />
                    <line x1="62" y1="72" x2="362" y2="72" stroke="#e2e8f0" strokeDasharray="3 3" />
                    {/* 3D Coordinate Axes (+X, +Y, +Z) */}
                    <line x1="52" y1="130" x2="356" y2="130" stroke="#e11d48" strokeWidth="1.2" />
                    <line x1="52" y1="130" x2="52" y2="18" stroke="#059669" strokeWidth="1.2" />
                    <line x1="52" y1="130" x2="62" y2="122" stroke="#2563eb" strokeWidth="1.2" />
                    <text x="46" y="25" textAnchor="end" fontSize="8.5" fill="#475569" fontFamily="monospace">100%</text>
                    <text x="46" y="75" textAnchor="end" fontSize="8.5" fill="#475569" fontFamily="monospace">50%</text>
                    <text x="46" y="131" textAnchor="end" fontSize="8.5" fill="#475569" fontFamily="monospace">0%</text>
                    <text x="54" y="12" fontSize="7.5" fontWeight="bold" fill="#059669" fontFamily="monospace">+Y: Val %</text>
                    <text x="364" y="133" fontSize="7.5" fontWeight="bold" fill="#e11d48" fontFamily="monospace">+X</text>
                    <text x="68" y="118" fontSize="7.5" fontWeight="bold" fill="#2563eb" fontFamily="monospace">+Z</text>
                    <text x="12" y="148" fontSize="7.5" fill="#64748b" fontFamily="monospace">(0,0,0)</text>
                    {trainingLogs.map((l, idx) => {
                      const slotW = 288 / Math.max(1, trainingLogs.length);
                      const barW = Math.min(15, slotW * 0.54);
                      const x = 56 + idx * slotW + (slotW - barW) / 2;
                      const barH = Math.max(5, l.valAcc * 92);
                      const y = 128 - barH;
                      const dx = 6;
                      const dy = -5;
                      const isHigh = l.valAcc >= 0.94;
                      const frontCol = isHigh ? '#059669' : '#4f46e5';
                      const topCol = isHigh ? '#34d399' : '#818cf8';
                      const sideCol = isHigh ? '#065f46' : '#312e81';
                      const topPts = `${x},${y} ${x + dx},${y + dy} ${x + barW + dx},${y + dy} ${x + barW},${y}`;
                      const sidePts = `${x + barW},${y} ${x + barW + dx},${y + dy} ${x + barW + dx},${y + barH + dy} ${x + barW},${y + barH}`;
                      const labelOffsetY = idx % 2 === 1 ? -11 : 0;
                      const labelY = y + dy - 4 + labelOffsetY;
                      return (
                        <g key={l.epoch}>
                          <polygon points={topPts} fill={topCol} />
                          <polygon points={sidePts} fill={sideCol} />
                          <rect x={x} y={y} width={barW} height={barH} fill={frontCol} rx="1" />
                          {trainingLogs.length <= 12 && (
                            <>
                              {idx % 2 === 1 && (
                                <line
                                  x1={x + barW / 2 + dx * 0.4}
                                  y1={y + dy}
                                  x2={x + barW / 2 + dx * 0.4}
                                  y2={labelY + 2}
                                  stroke="#94a3b8"
                                  strokeWidth="0.7"
                                  strokeDasharray="1.5 1.5"
                                />
                              )}
                              <text
                                x={x + barW / 2 + dx * 0.4}
                                y={labelY}
                                textAnchor="middle"
                                fontSize="7.5"
                                fontFamily="monospace"
                                fontWeight="bold"
                                fill="#065f46"
                                style={{ paintOrder: 'stroke' }}
                                stroke="#ffffff"
                                strokeWidth="2.5"
                              >
                                {(l.valAcc * 100).toFixed(0)}%
                              </text>
                            </>
                          )}
                          <text
                            x={x + barW / 2}
                            y="156"
                            textAnchor="middle"
                            fontSize="8"
                            fontFamily="monospace"
                            fill="#475569"
                          >
                            E{l.epoch}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
              </div>
            ) : (
              <div className="h-36 bg-white rounded-xl border border-slate-200 flex flex-col items-center justify-center text-xs text-slate-600 font-mono gap-2">
                <span>Click "Run 1.50B Simulation" or select any Target Project to plot real-time Loss Curves &amp; Val Accuracy Bar Charts.</span>
              </div>
            )}
          </div>

          {/* World-Class Research Outcome & SLA Verdict Banner */}
          <div className="rounded-2xl bg-white border border-slate-300 p-5 shadow-xs space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-5 h-5 text-emerald-700" />
                <h4 className="text-sm font-bold text-slate-900">
                  Research Outcome &amp; 20-Year Production SLA Verdict
                </h4>
              </div>
              <span
                className={`px-2.5 py-1 rounded-md text-xs font-mono font-bold border ${
                  slaPass
                    ? 'bg-emerald-50 text-emerald-900 border-emerald-300'
                    : 'bg-amber-50 text-amber-900 border-amber-300'
                }`}
              >
                {slaPass ? 'CIDR <200ms SLA PASSED · PQC HARDENED' : 'HIGH LATENCY WARNING'}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                <span className="font-mono text-[11px] text-slate-500 block">
                  Generalization &amp; Calibration
                </span>
                <p className="font-bold text-slate-900">
                  Gap: {generalizationGapPct}% · ECE: {activeMlMeta.eceScore}
                </p>
                <p className="text-slate-600 text-[11px]">
                  {isCrossModality
                    ? 'Cross-modality stress evaluated; native modality gives higher TAR.'
                    : 'Aligned native modality achieves optimal calibration stability.'}
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                <span className="font-mono text-[11px] text-slate-500 block">
                  Runtime &amp; Concurrency Fit
                </span>
                <p className="font-bold text-slate-900">
                  {activeProgMeta.shortName}: P95 {displayLatency}ms · P99 {displayP99Latency}ms
                </p>
                <p className="text-slate-600 text-[11px]">
                  {activeProgMeta.concurrencyModel} · {activeProgMeta.costIndex} TCO
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                <span className="font-mono text-[11px] text-slate-500 block">
                  2046 (Year +20) Readiness
                </span>
                <p className="font-bold text-slate-900">
                  {year20Point.populationNominalB}B Pop (Max {year20Point.populationUpperB}B)
                </p>
                <p className="text-slate-600 text-[11px]">
                  {year20Point.peakQps.toLocaleString()} QPS · Acc: {year20Point.expectedAccPct}%
                </p>
              </div>

              <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
                <span className="font-mono text-[11px] text-slate-500 block">
                  Encryption (Rest &amp; Transit)
                </span>
                <p className="font-bold text-slate-900">
                  {activeEncMeta.shortName} (+{activeEncMeta.latencyOverheadMs}ms)
                </p>
                <p className="text-slate-600 text-[11px] truncate" title={`${activeEncMeta.atRestCipher} | ${activeEncMeta.inTransitProtocol}`}>
                  Rest: {activeEncMeta.atRestCipher.split('+')[0].trim()} · Wire: {activeEncMeta.inTransitProtocol.split('+')[0].trim()}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Multi-Technique Visual Analytics Suite (Val Acc Bar, Time-Series, k=9 / K-Means, Vectors, Tensors) */}
      <SimulatorMultiModalCharts
        project={activeSimProject}
        dataType={simConfig.dataType}
        mlTechnique={simConfig.mlTechnique}
        trainingLogs={trainingLogs}
        projectedValAcc={displayValAcc}
        tensorShape={activeProfile.tensorShape}
        seed={simConfig.seed}
        onSelectMlTechnique={(id) => applyAndRun({ mlTechnique: id })}
      />

      {/* Interactive Multi-Benchmark Comparison Suite */}
      <div className="rounded-2xl bg-white border border-slate-300 shadow-xs overflow-hidden">
        <div className="bg-slate-100 border-b border-slate-300 px-5 py-4 flex flex-col lg:flex-row lg:items-center justify-between gap-3">
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Multi-Dimensional Research Comparison &amp; 20-Year Scaling Benchmarks
            </h3>
            <p className="text-xs text-slate-600">
              Compare Data Types, Synthetic Distributions, 1.50B → 20-Year Growth Trajectories, Programming Architectures, ML Techniques, and Encryption Mechanisms for{' '}
              <span className="font-semibold text-slate-900">{activeSimProject.title}</span>.
            </p>
          </div>

          <div className="flex flex-wrap gap-1.5">
            {[
              { id: 'datatype', label: '1. Data Types', icon: Layers },
              { id: 'distribution', label: '2. Distributions (1.50B)', icon: BarChart3 },
              { id: 'horizon', label: '3. 20-Yr Growth (Δ ±%)', icon: TrendingUp },
              { id: 'architecture', label: '4. Open-Source HFT Arch', icon: Cpu },
              { id: 'ml_technique', label: '5. ML Techniques', icon: Sparkles },
              { id: 'learning_rates', label: '6. Learning Rates & LR Schedules', icon: Activity },
              { id: 'encryption', label: '7. Encryption (Rest / Transit)', icon: ShieldCheck }
            ].map((tab) => {
              const Icon = tab.icon;
              const active = activeBenchTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveBenchTab(tab.id as BenchmarkView)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all border ${
                    active
                      ? 'bg-indigo-600 text-white border-indigo-600 shadow-xs'
                      : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-200/70'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="p-5">
          {/* 1. DATA TYPE COMPARISON */}
          {activeBenchTab === 'datatype' && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
                <span>
                  Evaluating all 7 tensor modalities against{' '}
                  <strong className="text-slate-900">{activeSimProject.title}</strong>. Click{' '}
                  <span className="font-mono text-indigo-700 font-semibold">Simulate</span> on any row to switch &amp; run immediately.
                </span>
                <span className="font-mono text-[11px] text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  Auto-Recommended: {SYNTHETIC_DATA_TYPE_META[activeProfile.defaultDataType].shortLabel}
                </span>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 font-mono">
                      <th className="py-2.5 px-3">Data Type</th>
                      <th className="py-2.5 px-3">Tensor Shape</th>
                      <th className="py-2.5 px-3">Est. Accuracy</th>
                      <th className="py-2.5 px-3">Est. Loss</th>
                      <th className="py-2.5 px-3">1.50B Throughput</th>
                      <th className="py-2.5 px-3">P95 / P99 Latency</th>
                      <th className="py-2.5 px-3">VRAM</th>
                      <th className="py-2.5 px-3">Match</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {(Object.keys(SYNTHETIC_DATA_TYPE_META) as SyntheticDataType[]).map((dt) => {
                      const m = SYNTHETIC_DATA_TYPE_META[dt];
                      const isNative = dt === activeProfile.defaultDataType;
                      const isSelected = dt === simConfig.dataType;
                      const estAcc = Math.min(
                        0.995,
                        Math.max(0.75, baseProjAcc + m.accBias - (isNative ? 0 : 0.032))
                      );
                      const estLoss = Number(((isNative ? 0.14 : 0.24) * m.lossMult).toFixed(3));
                      const estTput = Math.round(
                        m.throughputBase * activeProgMeta.throughputMultiplier * activeEncMeta.throughputMultiplier
                      );
                      const dtLat = computeLatencyP95P99(
                        simConfig.programmingArch,
                        dt,
                        simConfig.mlTechnique,
                        simConfig.encryptionProfile
                      );

                      return (
                        <tr
                          key={dt}
                          className={
                            isSelected
                              ? 'bg-indigo-50/70'
                              : isNative
                                ? 'bg-emerald-50/40 hover:bg-slate-50'
                                : 'hover:bg-slate-50'
                          }
                        >
                          <td className="py-2.5 px-3 font-semibold text-slate-900">
                            {m.shortLabel}
                            {isNative && (
                              <span className="ml-1.5 text-[10px] font-mono bg-emerald-100 text-emerald-900 px-1.5 py-0.5 rounded border border-emerald-300">
                                ★ Native
                              </span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-[11px] text-slate-600">{m.tensorShape}</td>
                          <td className="py-2.5 px-3 font-mono font-bold text-emerald-800">
                            {(estAcc * 100).toFixed(2)}%
                          </td>
                          <td className="py-2.5 px-3 font-mono text-blue-900">{estLoss}</td>
                          <td className="py-2.5 px-3 font-mono text-slate-800">{estTput.toLocaleString()}/s</td>
                          <td className="py-2.5 px-3 font-mono text-slate-800 whitespace-nowrap">
                            <span>{dtLat.p95Ms}</span>
                            <span className="text-slate-400 mx-1">/</span>
                            <span className="text-rose-800 font-semibold">{dtLat.p99Ms} ms</span>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-700">{m.vramGb} GB</td>
                          <td className="py-2.5 px-3 text-[11px] font-mono">
                            {isNative ? 'Optimal Fit' : 'Cross-Modal Adapter'}
                          </td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={() => applyAndRun({ dataType: dt })}
                              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold border transition-colors ${
                                isSelected
                                  ? 'bg-indigo-600 text-white border-indigo-600'
                                  : 'bg-white text-indigo-800 border-indigo-300 hover:bg-indigo-50'
                              }`}
                            >
                              {isSelected ? 'Active' : 'Simulate'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 2. SYNTHETIC DATASET DISTRIBUTION COMPARISON */}
          {activeBenchTab === 'distribution' && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
                <span>
                  Comparing 1.50B-scale synthetic sampling regimes across minority fraud stress, 20-year covariate aging drift, and longitudinal growth.
                </span>
                <span className="font-mono text-[11px] text-slate-700">
                  Active: <strong>{activeVarMeta.label}</strong>
                </span>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 font-mono">
                      <th className="py-2.5 px-3">Distribution Regime</th>
                      <th className="py-2.5 px-3">Class / Cohort Skew</th>
                      <th className="py-2.5 px-3">Perturbation σ</th>
                      <th className="py-2.5 px-3">ECE Calibration</th>
                      <th className="py-2.5 px-3">Accuracy Delta</th>
                      <th className="py-2.5 px-3">Loss Multiplier</th>
                      <th className="py-2.5 px-3">Verdict</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {(Object.keys(SYNTHETIC_VARIANT_META) as SyntheticVariant[]).map((v) => {
                      const vm = SYNTHETIC_VARIANT_META[v];
                      const isSel = v === simConfig.syntheticVariant;
                      const estAcc = Math.min(0.995, Math.max(0.74, baseProjAcc + vm.accOffset));
                      return (
                        <tr key={v} className={isSel ? 'bg-indigo-50/70' : 'hover:bg-slate-50'}>
                          <td className="py-2.5 px-3">
                            <div className="font-semibold text-slate-900">{vm.label}</div>
                            <div className="text-[11px] text-slate-500 mt-0.5">{vm.desc}</div>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-[11px] text-slate-700">{vm.skewProfile}</td>
                          <td className="py-2.5 px-3 font-mono text-slate-800">±{vm.sigmaBase}σ</td>
                          <td className="py-2.5 px-3 font-mono text-blue-900">{vm.eceCalibration}</td>
                          <td className="py-2.5 px-3 font-mono font-bold text-emerald-800">
                            {(estAcc * 100).toFixed(2)}% ({vm.accOffset >= 0 ? '+' : ''}
                            {(vm.accOffset * 100).toFixed(1)}%)
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-800">{vm.lossPenalty.toFixed(2)}x</td>
                          <td className="py-2.5 px-3 font-mono text-[11px] text-slate-700">{vm.stabilityVerdict}</td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={() =>
                                applyAndRun({
                                  syntheticVariant: v,
                                  sampleCount: v === 'scaled_100k' ? 2410000000 : 1500000000
                                })
                              }
                              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold border ${
                                isSel
                                  ? 'bg-indigo-600 text-white border-indigo-600'
                                  : 'bg-white text-indigo-800 border-indigo-300 hover:bg-indigo-50'
                              }`}
                            >
                              {isSel ? 'Active' : 'Apply & Run'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 3. 1.50B POPULATION & 20-YEAR GROWTH HORIZON */}
          {activeBenchTab === 'horizon' && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                <span className="text-slate-600">
                  Longitudinal cohort scaling from <strong className="text-slate-900">1.50 Billion (2026)</strong> across{' '}
                  <strong className="text-slate-900">20 years (2046)</strong> with{' '}
                  <strong className="text-indigo-800">±{simConfig.annualDeltaPct}% annual delta</strong> bounds and periodic 3-year continual recalibration.
                </span>
                <div className="flex items-center gap-3 font-mono text-[11px]">
                  <span className="text-rose-800 font-semibold">-- Upper (+{simConfig.annualDeltaPct}%/yr)</span>
                  <span className="text-indigo-800 font-semibold">— Nominal Cohort</span>
                  <span className="text-emerald-800 font-semibold">-- Lower (-{(simConfig.annualDeltaPct * 0.45).toFixed(1)}%/yr)</span>
                </div>
              </div>

              {/* 3D Extruded Cohort Fan & Milestone Pillar Chart */}
              <svg viewBox="0 0 700 212" className="w-full h-52 bg-slate-50 rounded-xl border border-slate-200">
                {/* 3D stage floor & back grid */}
                <polygon points="62,162 80,146 668,146 650,162" fill="#e2e8f0" fillOpacity="0.45" stroke="#cbd5e1" strokeWidth="0.9" />
                <line x1="80" y1="32" x2="668" y2="32" stroke="#e2e8f0" strokeDasharray="3 3" />
                <line x1="80" y1="88" x2="668" y2="88" stroke="#e2e8f0" strokeDasharray="3 3" />
                {/* 3D Coordinate Axes (+X, +Y, +Z) */}
                <line x1="62" y1="162" x2="660" y2="162" stroke="#e11d48" strokeWidth="1.3" />
                <line x1="62" y1="162" x2="62" y2="20" stroke="#059669" strokeWidth="1.3" />
                <line x1="62" y1="162" x2="80" y2="146" stroke="#2563eb" strokeWidth="1.3" />
                <text x="54" y="36" textAnchor="end" fontSize="9" fill="#475569" fontFamily="monospace">2.60B</text>
                <text x="54" y="92" textAnchor="end" fontSize="9" fill="#475569" fontFamily="monospace">1.95B</text>
                <text x="54" y="160" textAnchor="end" fontSize="9" fill="#475569" fontFamily="monospace">1.30B</text>
                <text x="68" y="15" fontSize="8" fontWeight="bold" fill="#059669" fontFamily="monospace">+Y: Cohort (B)</text>
                <text x="86" y="142" fontSize="8" fontWeight="bold" fill="#2563eb" fontFamily="monospace">+Z: ±Δ Envelope</text>
                <text x="14" y="178" fontSize="8" fill="#64748b" fontFamily="monospace">(0,0,0)</text>
                <text x="664" y="165" fontSize="8" fontWeight="bold" fill="#e11d48" fontFamily="monospace">+X: Yr</text>
                {(() => {
                  const maxPop = Math.max(2550, ...horizonPoints.map((p) => p.populationUpperM));
                  const minPop = 1200;
                  const mapY = (val: number) => 158 - ((val - minPop) / (maxPop - minPop)) * 116;
                  const stepX = 562 / Math.max(1, horizonPoints.length - 1);
                  const dx = 14;
                  const dy = -12;

                  const upperArr = horizonPoints.map((p, i) => ({ x: 68 + i * stepX + dx, y: mapY(p.populationUpperM) + dy }));
                  const nomArr = horizonPoints.map((p, i) => ({ x: 68 + i * stepX + dx * 0.5, y: mapY(p.populationNominalM) + dy * 0.5 }));
                  const lowerArr = horizonPoints.map((p, i) => ({ x: 68 + i * stepX, y: mapY(p.populationLowerM) }));

                  const upperBand = [
                    ...upperArr.map((p) => `${p.x},${p.y}`),
                    ...[...nomArr].reverse().map((p) => `${p.x},${p.y}`)
                  ].join(' ');
                  const lowerBand = [
                    ...nomArr.map((p) => `${p.x},${p.y}`),
                    ...[...lowerArr].reverse().map((p) => `${p.x},${p.y}`)
                  ].join(' ');

                  const milestones = horizonPoints.filter((p) => [0, 5, 10, 15, 20].includes(p.yearOffset));

                  return (
                    <>
                      <polygon points={upperBand} fill="#be123c" fillOpacity="0.14" />
                      <polygon points={lowerBand} fill="#4f46e5" fillOpacity="0.16" />

                      {/* 3D Extruded Milestone Cohort Pillars */}
                      {milestones.map((m) => {
                        const idx = horizonPoints.findIndex((hp) => hp.yearOffset === m.yearOffset);
                        const bx = 68 + idx * stepX - 7;
                        const by = mapY(m.populationNominalM);
                        const bh = Math.max(8, 160 - by);
                        const bw = 16;
                        const pdx = 7;
                        const pdy = -6;
                        // Position label cleanly above upper curve to never cross curve lines
                        const upperY = upperArr[idx]?.y ?? by;
                        const labelY = Math.max(18, Math.min(by + pdy - 6, upperY - 9));
                        return (
                          <g key={m.yearOffset}>
                            <polygon
                              points={`${bx},${by} ${bx + pdx},${by + pdy} ${bx + bw + pdx},${by + pdy} ${bx + bw},${by}`}
                              fill="#818cf8"
                            />
                            <polygon
                              points={`${bx + bw},${by} ${bx + bw + pdx},${by + pdy} ${bx + bw + pdx},${by + bh + pdy} ${bx + bw},${by + bh}`}
                              fill="#312e81"
                            />
                            <rect x={bx} y={by} width={bw} height={bh} fill="#4f46e5" fillOpacity="0.88" rx="1" />
                            <line
                              x1={bx + bw / 2 + 3}
                              y1={by + pdy}
                              x2={bx + bw / 2 + 3}
                              y2={labelY + 2}
                              stroke="#6366f1"
                              strokeWidth="0.7"
                              strokeDasharray="2 2"
                            />
                            <text
                              x={bx + bw / 2 + 3}
                              y={labelY}
                              textAnchor="middle"
                              fontSize="8.5"
                              fontWeight="bold"
                              fontFamily="monospace"
                              fill="#1e1b4b"
                              style={{ paintOrder: 'stroke' }}
                              stroke="#f8fafc"
                              strokeWidth="3"
                            >
                              {m.populationNominalB}B
                            </text>
                          </g>
                        );
                      })}

                      <polyline fill="none" stroke="#9f1239" strokeWidth="2" strokeDasharray="5 3" points={upperArr.map((p) => `${p.x},${p.y}`).join(' ')} />
                      <polyline fill="none" stroke="#3730a3" strokeWidth="2.8" points={nomArr.map((p) => `${p.x},${p.y}`).join(' ')} />
                      <polyline fill="none" stroke="#065f46" strokeWidth="2" strokeDasharray="4 3" points={lowerArr.map((p) => `${p.x},${p.y}`).join(' ')} />
                    </>
                  );
                })()}
                {[0, 5, 10, 15, 20].map((yr) => (
                  <text
                    key={yr}
                    x={68 + (yr / 20) * 562}
                    y="192"
                    textAnchor="middle"
                    fontSize="9"
                    fill="#475569"
                    fontFamily="monospace"
                  >
                    {2026 + yr} (Yr {yr})
                  </text>
                ))}
              </svg>

              {/* Milestone Projection Table */}
              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left border-collapse text-xs font-mono">
                  <thead>
                    <tr className="bg-slate-100 text-slate-700 border-b border-slate-200">
                      <th className="py-2 px-3">Horizon Milestone</th>
                      <th className="py-2 px-3">Nominal Cohort</th>
                      <th className="py-2 px-3">Upper (+Δ) / Lower (-Δ)</th>
                      <th className="py-2 px-3">Daily Auth Calls</th>
                      <th className="py-2 px-3">Peak QPS</th>
                      <th className="py-2 px-3">Hot Tier TB</th>
                      <th className="py-2 px-3">Warm+Cold Tier</th>
                      <th className="py-2 px-3">Recalibrated Acc</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {horizonPoints
                      .filter((p) => [0, 5, 10, 15, 20].includes(p.yearOffset))
                      .map((p) => (
                        <tr key={p.yearOffset} className="hover:bg-slate-50">
                          <td className="py-2 px-3 font-bold text-slate-900">
                            Year +{p.yearOffset} ({p.calendarYear})
                          </td>
                          <td className="py-2 px-3 font-bold text-indigo-800">{p.populationNominalB}B</td>
                          <td className="py-2 px-3 text-slate-700">
                            {p.populationUpperB}B / {p.populationLowerB}B
                          </td>
                          <td className="py-2 px-3 text-slate-800">{p.dailyAuthCallsB}B / day</td>
                          <td className="py-2 px-3 text-blue-900 font-semibold">{p.peakQps.toLocaleString()}</td>
                          <td className="py-2 px-3 text-slate-700">{p.hotStorageTb.toLocaleString()} TB</td>
                          <td className="py-2 px-3 text-slate-700">{p.warmColdStorageTb.toLocaleString()} TB</td>
                          <td className="py-2 px-3 font-bold text-emerald-800">{p.expectedAccPct}%</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 4. OPEN-SOURCE LOW-LATENCY HIGH-FREQUENCY PROGRAMMING ARCHITECTURE COMPARISON */}
          {activeBenchTab === 'architecture' && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
                <span>
                  100% Open-Source (Apache-2.0 / MIT / BSD), platform-free low-latency high-frequency architectures using kernel-bypass <code className="font-mono text-indigo-800">io_uring</code>, LMAX ring buffers, Seastar/DPDK, Arrow Flight, and OpenVINO/TVM with zero proprietary vendor lock-in.
                </span>
                <span className="font-mono text-[11px] text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  HFT FOSS Optimal: {PROGRAMMING_ARCHITECTURE_META[bestProgArch].shortName}
                </span>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 font-mono">
                      <th className="py-2.5 px-3">Open-Source HFT Stack &amp; License</th>
                      <th className="py-2.5 px-3">Low-Latency Bypass &amp; Platform Portability</th>
                      <th className="py-2.5 px-3">1.50B Throughput</th>
                      <th className="py-2.5 px-3">P95 / P99 Latency</th>
                      <th className="py-2.5 px-3">RAM</th>
                      <th className="py-2.5 px-3">Cost Index</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {(Object.keys(PROGRAMMING_ARCHITECTURE_META) as ProgrammingArchId[]).map((archId) => {
                      const arch = PROGRAMMING_ARCHITECTURE_META[archId];
                      const isSel = archId === simConfig.programmingArch;
                      const isBest = archId === bestProgArch;
                      const archTput = Math.round(
                        activeTypeMeta.throughputBase * arch.throughputMultiplier * activeEncMeta.throughputMultiplier
                      );
                      const archLatStats = computeLatencyP95P99(
                        archId,
                        simConfig.dataType,
                        simConfig.mlTechnique,
                        simConfig.encryptionProfile
                      );
                      return (
                        <tr key={archId} className={isSel ? 'bg-indigo-50/70' : 'hover:bg-slate-50'}>
                          <td className="py-2.5 px-3">
                            <div className="font-semibold text-slate-900">
                              {arch.name}
                              {isBest && (
                                <span className="ml-1.5 text-[10px] font-mono bg-emerald-100 text-emerald-900 px-1.5 py-0.5 rounded border border-emerald-300">
                                  ★ HFT Optimal
                                </span>
                              )}
                            </div>
                            <div className="text-[11px] font-mono text-emerald-800 mt-0.5">
                              {arch.license} · {arch.suitability}
                            </div>
                          </td>
                          <td className="py-2.5 px-3 text-[11px] text-slate-700 max-w-xs">
                            <div className="font-semibold text-slate-800">{arch.hftMechanism}</div>
                            <div className="text-indigo-800 font-mono text-[10px] mt-0.5">{arch.platformLockIn}</div>
                          </td>
                          <td className="py-2.5 px-3 font-mono font-bold text-indigo-900 whitespace-nowrap">
                            {archTput.toLocaleString()}/s
                          </td>
                          <td className="py-2.5 px-3 font-mono whitespace-nowrap">
                            <div className="font-bold text-emerald-800">
                              P95: {archLatStats.p95Ms} ms · <span className="text-rose-800">P99: {archLatStats.p99Ms} ms</span>
                            </div>
                            <div className="text-[10px] text-slate-500">±{arch.p99JitterUs} μs tail jitter</div>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-800 whitespace-nowrap">{arch.memoryFootprintGb} GB</td>
                          <td className="py-2.5 px-3 font-mono text-slate-700 whitespace-nowrap">{arch.costIndex}</td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={() => applyAndRun({ programmingArch: archId })}
                              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold border ${
                                isSel
                                  ? 'bg-indigo-600 text-white border-indigo-600'
                                  : 'bg-white text-indigo-800 border-indigo-300 hover:bg-indigo-50'
                              }`}
                            >
                              {isSel ? 'Active' : 'Benchmark'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 5. ML TECHNIQUES COMPARISON */}
          {activeBenchTab === 'ml_technique' && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
                <span>
                  Comparative ablation across modern ML learning techniques on{' '}
                  <strong className="text-slate-900">{activeSimProject.title}</strong> at 1.50 Billion population scale.
                </span>
                <span className="font-mono text-[11px] text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  Recommended: {ML_TECHNIQUE_META[bestMlTech].shortName}
                </span>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 font-mono">
                      <th className="py-2.5 px-3">ML Learning Technique</th>
                      <th className="py-2.5 px-3">Loss / Objective Formula</th>
                      <th className="py-2.5 px-3">Expected Acc</th>
                      <th className="py-2.5 px-3">ECE Calib</th>
                      <th className="py-2.5 px-3">P95 / P99 Latency</th>
                      <th className="py-2.5 px-3">1.50B GPU Hrs</th>
                      <th className="py-2.5 px-3">Research Strengths</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {(Object.keys(ML_TECHNIQUE_META) as MlTechniqueId[]).map((techId) => {
                      const m = ML_TECHNIQUE_META[techId];
                      const isSel = techId === simConfig.mlTechnique;
                      const isBest = techId === bestMlTech;
                      const estAcc = Math.min(0.996, Math.max(0.75, baseProjAcc + m.accModifier));
                      const mlLat = computeLatencyP95P99(
                        simConfig.programmingArch,
                        simConfig.dataType,
                        techId,
                        simConfig.encryptionProfile
                      );
                      return (
                        <tr key={techId} className={isSel ? 'bg-indigo-50/70' : 'hover:bg-slate-50'}>
                          <td className="py-2.5 px-3 font-semibold text-slate-900">
                            {m.name}
                            {isBest && (
                              <span className="ml-1.5 text-[10px] font-mono bg-emerald-100 text-emerald-900 px-1.5 py-0.5 rounded border border-emerald-300">
                                ★ Best Fit
                              </span>
                            )}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-[11px] text-slate-700">{m.objectiveFormula}</td>
                          <td className="py-2.5 px-3 font-mono font-bold text-emerald-800">
                            {(estAcc * 100).toFixed(2)}%
                          </td>
                          <td className="py-2.5 px-3 font-mono text-blue-900">{m.eceScore}</td>
                          <td className="py-2.5 px-3 font-mono text-slate-800 whitespace-nowrap">
                            {mlLat.p95Ms} / <span className="text-rose-800 font-semibold">{mlLat.p99Ms} ms</span>
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-800">{m.trainHours1B5} hrs</td>
                          <td className="py-2.5 px-3 text-[11px] text-slate-600 max-w-xs">{m.strengths}</td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={() => applyAndRun({ mlTechnique: techId })}
                              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold border ${
                                isSel
                                  ? 'bg-indigo-600 text-white border-indigo-600'
                                  : 'bg-white text-indigo-800 border-indigo-300 hover:bg-indigo-50'
                              }`}
                            >
                              {isSel ? 'Active' : 'Simulate'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 6. SUGGESTED LEARNING RATES & SCHEDULES COMPARISON */}
          {activeBenchTab === 'learning_rates' && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
                <span>
                  Curated learning rate presets and annealing schedules for <strong className="text-slate-900">{activeSimProject.title}</strong> ({activeMlMeta.shortName}). Click <span className="font-mono text-indigo-700 font-semibold">Simulate LR</span> to benchmark convergence.
                </span>
                <span className="font-mono text-[11px] text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  Suggested LR: {bestLrPreset.lrValue} ({bestLrPreset.schedule})
                </span>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 font-mono">
                      <th className="py-2.5 px-3">Suggested LR &amp; Schedule</th>
                      <th className="py-2.5 px-3">Best Applicable Domains / Models</th>
                      <th className="py-2.5 px-3">Rec. Batch</th>
                      <th className="py-2.5 px-3">Speed</th>
                      <th className="py-2.5 px-3">Stability Risk</th>
                      <th className="py-2.5 px-3">Proj. Acc</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {LEARNING_RATE_PRESETS.map((lr) => {
                      const isSel = Number(simConfig.learningRate) === lr.numericLr;
                      const isBest = lr.id === bestLrPreset.id;
                      const estAcc = Math.min(0.996, Math.max(0.75, projectedAcc + lr.accBonus));
                      return (
                        <tr key={lr.id} className={isSel ? 'bg-indigo-50/70' : 'hover:bg-slate-50'}>
                          <td className="py-2.5 px-3">
                            <div className="font-semibold text-slate-900 font-mono">
                              {lr.lrValue}
                              {isBest && (
                                <span className="ml-1.5 text-[10px] font-mono bg-emerald-100 text-emerald-900 px-1.5 py-0.5 rounded border border-emerald-300">
                                  ★ Suggested
                                </span>
                              )}
                            </div>
                            <div className="text-[11px] text-indigo-800 mt-0.5">{lr.schedule}</div>
                          </td>
                          <td className="py-2.5 px-3 text-[11px] text-slate-600 max-w-xs">{lr.bestFor}</td>
                          <td className="py-2.5 px-3 font-mono text-slate-800 whitespace-nowrap">{lr.recommendedBatch}</td>
                          <td className="py-2.5 px-3 text-slate-800 whitespace-nowrap">{lr.convergenceSpeed}</td>
                          <td className="py-2.5 px-3 text-[11px] font-mono text-slate-700">{lr.stabilityRisk}</td>
                          <td className="py-2.5 px-3 font-mono font-bold text-emerald-800">
                            {(estAcc * 100).toFixed(2)}%
                          </td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={() => applyAndRun({ learningRate: lr.lrValue })}
                              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold border ${
                                isSel
                                  ? 'bg-indigo-600 text-white border-indigo-600'
                                  : 'bg-white text-indigo-800 border-indigo-300 hover:bg-indigo-50'
                              }`}
                            >
                              {isSel ? 'Active' : 'Simulate LR'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* 7. ENCRYPTION MECHANISMS (DATA AT REST & DATA IN TRANSIT) */}
          {activeBenchTab === 'encryption' && (
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-600">
                <span>
                  End-to-end cryptographic protection for <strong className="text-slate-900">Data at Rest</strong> (NVMe hot cache, Parquet WORM cold archive, VID vaults) and <strong className="text-slate-900">Data in Transit</strong> (mTLS 1.3, Post-Quantum ML-KEM-1024, WireGuard, RA-TLS Enclaves) across 1.50B–2.41B scale.
                </span>
                <span className="font-mono text-[11px] text-emerald-800 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                  20-Yr Sovereign Default: {ENCRYPTION_MECHANISM_META.pqc_hybrid_vault.shortName}
                </span>
              </div>

              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 font-mono">
                      <th className="py-2.5 px-3">Encryption Profile</th>
                      <th className="py-2.5 px-3">Data at Rest Mechanism</th>
                      <th className="py-2.5 px-3">Data in Transit Mechanism</th>
                      <th className="py-2.5 px-3">Key / HSM &amp; 20-Yr Safe</th>
                      <th className="py-2.5 px-3">P95 / P99 Total</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {(Object.keys(ENCRYPTION_MECHANISM_META) as EncryptionProfileId[]).map((encId) => {
                      const enc = ENCRYPTION_MECHANISM_META[encId];
                      const isSel = encId === simConfig.encryptionProfile;
                      const latStats = computeLatencyP95P99(
                        simConfig.programmingArch,
                        simConfig.dataType,
                        simConfig.mlTechnique,
                        encId
                      );
                      return (
                        <tr key={encId} className={isSel ? 'bg-indigo-50/70' : 'hover:bg-slate-50'}>
                          <td className="py-2.5 px-3">
                            <div className="font-semibold text-slate-900">
                              {enc.shortName}
                              {encId === 'pqc_hybrid_vault' && (
                                <span className="ml-1.5 text-[10px] font-mono bg-emerald-100 text-emerald-900 px-1.5 py-0.5 rounded border border-emerald-300">
                                  ★ 20-Yr PQC
                                </span>
                              )}
                            </div>
                            <div className="text-[11px] text-slate-500 mt-0.5">{enc.suitability}</div>
                          </td>
                          <td className="py-2.5 px-3 text-[11px] font-mono text-slate-800 max-w-xs">{enc.atRestCipher}</td>
                          <td className="py-2.5 px-3 text-[11px] font-mono text-blue-900 max-w-xs">{enc.inTransitProtocol}</td>
                          <td className="py-2.5 px-3 text-[11px]">
                            <div className="text-slate-800 font-semibold">{enc.quantumSafe20Yr}</div>
                            <div className="text-slate-500 font-mono text-[10px] mt-0.5">{enc.keyManagement}</div>
                          </td>
                          <td className="py-2.5 px-3 font-mono whitespace-nowrap">
                            <div className="font-bold text-emerald-800">
                              {latStats.p95Ms} / <span className="text-rose-800">{latStats.p99Ms} ms</span>
                            </div>
                            <div className="text-[10px] text-slate-500">+{enc.latencyOverheadMs}ms cipher cost</div>
                          </td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={() => applyAndRun({ encryptionProfile: encId })}
                              className={`px-2.5 py-1 rounded-md text-[11px] font-semibold border ${
                                isSel
                                  ? 'bg-indigo-600 text-white border-indigo-600'
                                  : 'bg-white text-indigo-800 border-indigo-300 hover:bg-indigo-50'
                              }`}
                            >
                              {isSel ? 'Active' : 'Apply & Run'}
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Row: Synthetic 1.50B Sample Tensor Batch + Epoch Telemetry Log */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Active Synthetic Sample Probe Table */}
        <div className="lg:col-span-5 rounded-2xl bg-slate-100 border border-slate-300 p-5 space-y-3 shadow-xs">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="flex items-center gap-2 text-[11px] font-mono">
                <Database className="w-3.5 h-3.5 text-rose-800" />
                <span className="font-bold text-rose-900">1.50B Synthetic Batch Probe</span>
                <span className="text-slate-500">·</span>
                <span className="text-blue-900 font-semibold">{activeProfile.noiseSigma}</span>
              </div>
              <h4 className="text-sm font-bold text-slate-900 mt-1">{activeProfile.datasetName}</h4>
            </div>
          </div>

          <div className="overflow-x-auto bg-white rounded-xl border border-slate-300">
            <table className="w-full text-left border-collapse text-[11px] font-mono">
              <thead>
                <tr className="bg-slate-200/60 border-b border-slate-300 text-slate-700">
                  {activeProfile.columns.map((c) => (
                    <th key={c.name} className="py-1.5 px-2.5 font-semibold whitespace-nowrap">
                      {c.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {previewSamples.map((row, idx) => (
                  <tr key={idx} className="hover:bg-slate-50">
                    {activeProfile.columns.map((c) => (
                      <td
                        key={c.name}
                        className={`py-1.5 px-2.5 whitespace-nowrap ${
                          c.role === 'target' ? 'text-emerald-900 font-bold' : 'text-slate-700'
                        }`}
                      >
                        {String(row[c.name])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Epoch Telemetry Log Table */}
        <div className="lg:col-span-7 rounded-2xl bg-slate-100 border border-slate-300 overflow-hidden shadow-xs">
          <div className="px-5 py-3.5 border-b border-slate-300 bg-slate-200/80 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-800" />
              <h3 className="text-sm font-bold text-slate-900">Epoch Telemetry &amp; SLA Log</h3>
            </div>
            <span className="text-[11px] font-mono text-blue-900 font-semibold">
              {activeProgMeta.shortName} · {activeMlMeta.shortName}
            </span>
          </div>
          <div className="overflow-x-auto max-h-64 bg-white">
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead>
                <tr className="bg-slate-100 text-slate-700 border-b border-slate-200">
                  <th className="py-2.5 px-3 font-semibold">Epoch</th>
                  <th className="py-2.5 px-3 font-semibold">Train Loss</th>
                  <th className="py-2.5 px-3 font-semibold">Train Acc</th>
                  <th className="py-2.5 px-3 font-semibold">Val Acc</th>
                  <th className="py-2.5 px-3 font-semibold">±σ Dev</th>
                  <th className="py-2.5 px-3 font-semibold">Throughput</th>
                  <th className="py-2.5 px-3 font-semibold">P95 Lat</th>
                  <th className="py-2.5 px-3 font-semibold">P99 Lat</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {trainingLogs.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="py-8 text-center text-slate-500">
                      Run a simulation or pick any Target Project to populate live epoch telemetry.
                    </td>
                  </tr>
                ) : (
                  trainingLogs.map((log) => (
                    <tr key={log.epoch} className="hover:bg-slate-50 transition-colors">
                      <td className="py-2 px-3 text-slate-900 font-bold">{log.epoch}</td>
                      <td className="py-2 px-3 text-blue-900 font-semibold tabular-nums">{log.loss}</td>
                      <td className="py-2 px-3 text-emerald-800 font-semibold tabular-nums">
                        {(log.accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 text-rose-800 font-semibold tabular-nums">
                        {(log.valAcc * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 text-slate-700 tabular-nums">±{log.sigma}%</td>
                      <td className="py-2 px-3 text-indigo-900 tabular-nums">
                        {log.throughput.toLocaleString()}/s
                      </td>
                      <td className="py-2 px-3 text-slate-800 tabular-nums">{log.latencyMs} ms</td>
                      <td className="py-2 px-3 text-rose-900 font-semibold tabular-nums">{log.p99LatencyMs} ms</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
