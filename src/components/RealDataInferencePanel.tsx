import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  ArrowRightLeft,
  BarChart3,
  CheckCircle2,
  Cpu,
  DollarSign,
  Download,
  FileSpreadsheet,
  GitCompare,
  GitMerge,
  Play,
  Save,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Upload
} from 'lucide-react';
import { FlagshipProject, SyntheticDataType } from '../types';
import { CHART_COLORS, CHART_FONTS } from '../utils/chartTheme';
import { formatDualCurrency } from '../utils/projectEconomics';
import {
  EncryptionProfileId,
  ML_TECHNIQUE_META,
  MlTechniqueId,
  ProgrammingArchId
} from '../utils/syntheticDataset';
import {
  BatchInferenceReport,
  getComplementaryPair,
  getComplementarityExplanation,
  getRealSamplePresetsForProject,
  parseRealInputData,
  runRealDataMultiMethodInference
} from '../utils/realDataInference';
import { FittedBrowserModels, trainInBrowserModels } from '../utils/realMlTrainer';
import {
  ExperimentRunSnapshot,
  clearAllExperimentRuns,
  computeSha256Short,
  deleteExperimentRun,
  diffExperimentRuns,
  loadExperimentRuns,
  saveExperimentRun
} from '../utils/experimentTracker';

interface RealDataInferencePanelProps {
  project: FlagshipProject;
  dataType: SyntheticDataType;
  mlTechnique: MlTechniqueId;
  programmingArch: ProgrammingArchId;
  encryptionProfile: EncryptionProfileId;
  isTraining: boolean;
  trainingEpochsCompleted: number;
  targetEpochs: number;
  trainedValAcc: number;
  trainedLoss: number;
  onRequestTrainModel: () => void;
}

type InputSourceMode = 'presets' | 'paste' | 'upload';
type AnalysisSubTab = 'inference' | 'profiler' | 'evaluation' | 'experiments';

export default function RealDataInferencePanel({
  project,
  dataType,
  mlTechnique,
  programmingArch,
  encryptionProfile,
  isTraining,
  trainingEpochsCompleted,
  targetEpochs,
  trainedValAcc,
  trainedLoss,
  onRequestTrainModel
}: RealDataInferencePanelProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const presets = useMemo(
    () => getRealSamplePresetsForProject(project, dataType),
    [project, dataType]
  );
  const recommendedComp = useMemo(
    () => getComplementaryPair(mlTechnique, dataType),
    [mlTechnique, dataType]
  );

  const [inputMode, setInputMode] = useState<InputSourceMode>('presets');
  const [selectedPresetId, setSelectedPresetId] = useState<string>(presets[0]?.id || '');
  const [rawInput, setRawInput] = useState<string>(presets[0]?.payload || '');
  const [uploadedFileName, setUploadedFileName] = useState<string>('');
  const [complementaryMethod, setComplementaryMethod] = useState<MlTechniqueId>(recommendedComp);
  const [selectedRecordIdx, setSelectedRecordIdx] = useState<number>(0);

  // Interactive DS controls: Threshold tau, Conformal alpha, USD / INR Cost Matrix
  const [decisionThreshold, setDecisionThreshold] = useState<number>(0.5);
  const [conformalAlpha, setConformalAlpha] = useState<number>(0.1);
  const [fpCostInr, setFpCostInr] = useState<number>(3);
  const [fnCostInr, setFnCostInr] = useState<number>(100);

  // Real In-Browser ML Fitting State (AdamW Neural Logistic + GBDT Residual Stumps)
  const [fittedModels, setFittedModels] = useState<FittedBrowserModels | null>(null);
  const [browserTrainError, setBrowserTrainError] = useState<string | null>(null);

  // Reports & Sub-Views
  const [activeSubTab, setActiveSubTab] = useState<AnalysisSubTab>('inference');
  const [report, setReport] = useState<BatchInferenceReport | null>(null);
  const [inputError, setInputError] = useState<string | null>(null);
  const [pendingRunAfterTrain, setPendingRunAfterTrain] = useState<boolean>(false);

  // Experiment Persistence & A/B Diff
  const [savedRuns, setSavedRuns] = useState<ExperimentRunSnapshot[]>(() => loadExperimentRuns());
  const [runAId, setRunAId] = useState<string>('');
  const [runBId, setRunBId] = useState<string>('');
  const [lastDatasetHash, setLastDatasetHash] = useState<string>('—');
  const [lastConfigHash, setLastConfigHash] = useState<string>('—');

  useEffect(() => {
    const first = presets[0];
    if (first) {
      setSelectedPresetId(first.id);
      setRawInput(first.payload);
      setUploadedFileName('');
      setInputError(null);
      setReport(null);
      setFittedModels(null);
      setBrowserTrainError(null);
      setSelectedRecordIdx(0);
    }
  }, [presets]);

  useEffect(() => {
    setComplementaryMethod((prev) => (prev === mlTechnique ? recommendedComp : prev));
  }, [mlTechnique, recommendedComp]);

  useEffect(() => {
    if (savedRuns.length >= 2 && (!runAId || !runBId)) {
      setRunAId(savedRuns[1].runId);
      setRunBId(savedRuns[0].runId);
    } else if (savedRuns.length === 1 && !runBId) {
      setRunBId(savedRuns[0].runId);
    }
  }, [savedRuns, runAId, runBId]);

  const liveParsePreview = useMemo(
    () => parseRealInputData(rawInput, dataType),
    [rawInput, dataType]
  );

  const persistSnapshotForReport = async (
    rep: BatchInferenceReport,
    usedFitted: boolean,
    inputStr: string
  ) => {
    const dHash = await computeSha256Short(inputStr.trim());
    const cHash = await computeSha256Short(
      JSON.stringify({
        projectId: project.id,
        dataType,
        primaryMethod: rep.primaryMethodId,
        complementaryMethod: rep.complementaryMethodId,
        decisionThreshold: rep.costSummary.decisionThreshold,
        conformalAlpha: rep.conformalAlpha,
        fpCostInr: rep.costSummary.fpCostInr,
        fnCostInr: rep.costSummary.fnCostInr,
        usedFitted
      })
    );
    setLastDatasetHash(dHash);
    setLastConfigHash(cHash);

    const snapshot: ExperimentRunSnapshot = {
      runId: `run_${Date.now().toString(36)}_${cHash.slice(0, 4)}`,
      createdAt: new Date().toLocaleTimeString(),
      projectId: project.id,
      projectTitle: project.title.split(':')[0].slice(0, 32),
      dataType,
      primaryMethod: rep.primaryMethodId,
      complementaryMethod: rep.complementaryMethodId,
      datasetSha256: dHash,
      configSha256: cHash,
      inBrowserFitted: usedFitted,
      recordCount: rep.totalRecords,
      trainedValAccPct: usedFitted && rep.fittedModelSummary
        ? rep.fittedModelSummary.testEnsembleAccPct
        : rep.trainedValAccPct,
      meanConfidencePct: rep.meanConfidencePct,
      meanConsensusPct: rep.meanConsensusPct,
      conformalCoverageTargetPct: Math.round((1 - rep.conformalAlpha) * 100),
      meanConformalSetSize: rep.meanConformalSetSize,
      decisionThreshold: rep.costSummary.decisionThreshold,
      fpCostInr: rep.costSummary.fpCostInr,
      fnCostInr: rep.costSummary.fnCostInr,
      estimatedNetValueInr: rep.costSummary.netSavingsInr,
      batchP95LatencyMs: rep.batchP95LatencyMs,
      maxPsiDrift: rep.datasetProfile.overallPsiMax,
      brierScore: rep.groundTruthEval?.brierScore ?? rep.fittedModelSummary?.testBrierScore,
      prAucPct: rep.groundTruthEval ? Number((rep.groundTruthEval.prAuc * 100).toFixed(1)) : undefined,
      empiricalEcePct: rep.groundTruthEval?.empiricalEcePct,
      alertCount: rep.positiveCount,
      reviewCount: rep.reviewCount,
      clearCount: rep.nominalCount
    };

    const nextList = saveExperimentRun(snapshot);
    setSavedRuns(nextList);
    if (nextList.length >= 2) {
      setRunAId(nextList[1].runId);
      setRunBId(nextList[0].runId);
    } else if (nextList.length === 1) {
      setRunBId(nextList[0].runId);
    }
  };

  const executeInference = (overrides?: {
    inputText?: string;
    compId?: MlTechniqueId;
    threshold?: number;
    alpha?: number;
    fpCost?: number;
    fnCost?: number;
    fittedOverride?: FittedBrowserModels | null;
    autoSaveRun?: boolean;
  }) => {
    const inputText = overrides?.inputText ?? rawInput;
    const compId = overrides?.compId ?? complementaryMethod;
    const effectiveComp = compId === mlTechnique ? getComplementaryPair(mlTechnique, dataType) : compId;
    const activeFitted =
      overrides?.fittedOverride !== undefined ? overrides.fittedOverride : fittedModels;

    const res = runRealDataMultiMethodInference({
      rawInput: inputText,
      project,
      dataType,
      primaryMethod: mlTechnique,
      complementaryMethod: effectiveComp,
      programmingArch,
      encryptionProfile,
      trainedValAcc: activeFitted
        ? activeFitted.finalTestMetrics.ensembleAccPct / 100
        : trainedValAcc,
      trainedLoss: activeFitted
        ? activeFitted.epochHistory[activeFitted.epochHistory.length - 1]?.valLoss ?? trainedLoss
        : trainedLoss,
      trainedEpochs: activeFitted
        ? activeFitted.epochHistory.length
        : trainingEpochsCompleted > 0
          ? trainingEpochsCompleted
          : targetEpochs,
      decisionThreshold: overrides?.threshold ?? decisionThreshold,
      conformalAlpha: overrides?.alpha ?? conformalAlpha,
      fpCostInr: overrides?.fpCost ?? fpCostInr,
      fnCostInr: overrides?.fnCost ?? fnCostInr,
      fittedModels: activeFitted
    });

    if (res.error || !res.report) {
      setInputError(res.error || 'Unable to parse real input payload.');
      setReport(null);
      return;
    }

    setInputError(null);
    setReport(res.report);
    setSelectedRecordIdx(0);

    if (overrides?.autoSaveRun) {
      void persistSnapshotForReport(res.report, Boolean(activeFitted), inputText);
    }
  };

  useEffect(() => {
    if (!isTraining && trainingEpochsCompleted > 0 && pendingRunAfterTrain) {
      setPendingRunAfterTrain(false);
      executeInference({ autoSaveRun: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining, trainingEpochsCompleted, pendingRunAfterTrain]);

  const handleFitInBrowserModels = () => {
    const parsed = parseRealInputData(rawInput, dataType);
    if (parsed.error) {
      setBrowserTrainError(parsed.error);
      return;
    }
    const fitRes = trainInBrowserModels(parsed, { epochs: 18, lr: 0.09 });
    if (fitRes.error || !fitRes.fitted) {
      setBrowserTrainError(fitRes.error || 'In-browser model fitting failed.');
      return;
    }
    setBrowserTrainError(null);
    setFittedModels(fitRes.fitted);
    executeInference({ fittedOverride: fitRes.fitted, autoSaveRun: true });
  };

  const handleSelectPreset = (presetId: string) => {
    const found = presets.find((p) => p.id === presetId);
    if (!found) return;
    setSelectedPresetId(found.id);
    setRawInput(found.payload);
    setUploadedFileName('');
    setInputError(null);
    setFittedModels(null);
    setBrowserTrainError(null);
    if (trainingEpochsCompleted > 0 && !isTraining) {
      executeInference({ inputText: found.payload, fittedOverride: null });
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const content = await file.text();
      setUploadedFileName(file.name);
      setRawInput(content);
      setInputMode('paste');
      setInputError(null);
      setFittedModels(null);
      setBrowserTrainError(null);
      if (trainingEpochsCompleted > 0 && !isTraining) {
        executeInference({ inputText: content, fittedOverride: null });
      }
    } catch {
      setInputError('Failed to read selected file. Ensure it is UTF-8 CSV, TSV, JSON, or TXT.');
    }
  };

  const handleTrainAndEvaluate = () => {
    setPendingRunAfterTrain(true);
    onRequestTrainModel();
  };

  const handleExportResults = (fmt: 'csv' | 'json') => {
    if (!report) return;
    let blob: Blob;
    let filename = '';
    const safeSlug = project.id.replace(/[^a-z0-9_-]/gi, '_').slice(0, 32);

    if (fmt === 'json') {
      blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
      filename = `${safeSlug}_real_inference_report.json`;
    } else {
      const header = [
        'record_id',
        'ensemble_prediction',
        'ensemble_probability',
        'calibrated_confidence_pct',
        'conformal_set',
        'consensus_pct',
        'decision_tier',
        'recommendation',
        'ood_drift_z',
        'top_driver'
      ].join(',');

      const lines = report.recordResults.map((r) => {
        const esc = (s: string) => `"${String(s).replace(/"/g, '""')}"`;
        const driverStr = r.topDrivers[0]
          ? `${r.topDrivers[0].feature}=${r.topDrivers[0].valueDisplay}`
          : 'none';
        return [
          esc(r.recordId),
          esc(r.ensembleResult.predictedLabel),
          r.ensembleResult.probability,
          r.ensembleResult.confidencePct,
          esc(`{${r.conformal.predictionSet.join(' | ')}}`),
          r.ensembleResult.consensusPct,
          r.decisionTier,
          esc(r.recommendationTitle),
          r.oodDriftZ,
          esc(driverStr)
        ].join(',');
      });
      blob = new Blob([[header, ...lines].join('\n')], { type: 'text/csv;charset=utf-8' });
      filename = `${safeSlug}_real_inference_results.csv`;
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const isTrained = trainingEpochsCompleted > 0 && !isTraining;
  const activeRecord =
    report && report.recordResults[selectedRecordIdx]
      ? report.recordResults[selectedRecordIdx]
      : report?.recordResults[0] || null;

  const runA = savedRuns.find((r) => r.runId === runAId) || null;
  const runB = savedRuns.find((r) => r.runId === runBId) || null;
  const abDiffRows = runA && runB ? diffExperimentRuns(runA, runB) : [];

  return (
    <section
      aria-label="Post-Training Real-Data Inference, Statistical Profiling & Experiment Workbench"
      className="rounded-2xl bg-white border border-slate-300 shadow-xs overflow-hidden"
    >
      {/* Top Bar */}
      <div className="bg-slate-100 border-b border-slate-300 px-5 py-4 flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600">
            <span className="font-semibold text-indigo-800">Real Data Science &amp; Inference Workbench</span>
            <span aria-hidden="true">·</span>
            <span className="font-mono tabular-nums">
              {fittedModels
                ? `In-Browser Fitted (AdamW + ${fittedModels.gbdtStumps.length} GBDT Stumps · Test ${(fittedModels.finalTestMetrics.ensembleAccPct).toFixed(1)}%)`
                : isTraining
                  ? `Training Epoch ${trainingEpochsCompleted}/${targetEpochs}`
                  : isTrained
                    ? `Trained (${trainingEpochsCompleted} Epochs · Val ${(trainedValAcc * 100).toFixed(2)}%)`
                    : 'Baseline Ready'}
            </span>
          </div>
          <h3 className="text-base font-bold text-slate-900">
            Real-Data Training, Conformal Confidence, KS/PSI Profiling, Threshold Cost Tuning &amp; A/B Runs
          </h3>
          <p className="text-xs text-slate-600 max-w-4xl">
            Fit real in-browser AdamW + GBDT models on uploaded CSV/JSON cohorts, calibrate split-conformal prediction sets, audit KS/PSI/VIF drift, optimize USD/INR cost thresholds, and compare experiments across SHA-256 checksums.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          {!isTrained && (
            <button
              type="button"
              onClick={handleTrainAndEvaluate}
              disabled={isTraining}
              className="px-3.5 py-2 rounded-lg text-xs font-semibold bg-slate-900 hover:bg-slate-800 disabled:bg-slate-400 text-white flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
            >
              <Play className="w-3.5 h-3.5" />
              <span>{isTraining ? 'Training...' : 'Run Sim Training'}</span>
            </button>
          )}
          <button
            type="button"
            onClick={handleFitInBrowserModels}
            disabled={isTraining || liveParsePreview.records.length < 3 || liveParsePreview.numericColumns.length === 0}
            className="px-3.5 py-2 rounded-lg text-xs font-semibold bg-emerald-800 hover:bg-emerald-700 disabled:bg-slate-300 disabled:text-slate-600 text-white flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
          >
            <Cpu className="w-3.5 h-3.5" />
            <span>Fit Real MLP + GBDT on Data</span>
          </button>
          <button
            type="button"
            onClick={() => executeInference({ autoSaveRun: true })}
            disabled={isTraining || liveParsePreview.records.length === 0}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 disabled:text-slate-600 text-white flex items-center gap-1.5 transition-colors whitespace-nowrap shrink-0"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Run Real Inference</span>
          </button>
        </div>
      </div>

      {/* Main Content Body */}
      <div className="p-5 space-y-6">
        {/* Step 1: Real Input & Complementary Method Configuration */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
          <div className="lg:col-span-7 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="inline-flex items-center gap-1 p-1 bg-slate-100 rounded-lg border border-slate-200">
                <button
                  type="button"
                  onClick={() => setInputMode('presets')}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                    inputMode === 'presets'
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Labeled Real Cohorts
                </button>
                <button
                  type="button"
                  onClick={() => setInputMode('paste')}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                    inputMode === 'paste'
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  Paste CSV / JSON / Text
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setInputMode('upload');
                    fileInputRef.current?.click();
                  }}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors flex items-center gap-1.5 whitespace-nowrap ${
                    inputMode === 'upload'
                      ? 'bg-white text-slate-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  <Upload className="w-3.5 h-3.5" />
                  <span>Upload File</span>
                </button>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.tsv,.json,.txt,.log"
                onChange={handleFileUpload}
                className="hidden"
              />

              <span className="text-xs font-mono tabular-nums text-slate-600">
                {liveParsePreview.error
                  ? 'Awaiting valid input'
                  : `${liveParsePreview.format.toUpperCase()} · ${liveParsePreview.records.length} rows · ${liveParsePreview.numericColumns.length} num cols`}
              </span>
            </div>

            {inputMode === 'presets' && (
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {presets.map((p) => {
                  const active = p.id === selectedPresetId;
                  return (
                    <button
                      key={p.id}
                      type="button"
                      onClick={() => handleSelectPreset(p.id)}
                      className={`text-left p-2.5 rounded-xl border text-xs transition-colors ${
                        active
                          ? 'bg-indigo-50/80 border-indigo-400 text-slate-900'
                          : 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <div className="font-semibold line-clamp-1">{p.label}</div>
                      <div className="text-[11px] text-slate-500 mt-0.5 line-clamp-2">
                        {p.description}
                      </div>
                    </button>
                  );
                })}
              </div>
            )}

            {uploadedFileName && (
              <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700">
                <span className="flex items-center gap-2 font-mono">
                  <FileSpreadsheet className="w-4 h-4 text-indigo-700" />
                  Loaded file: <strong className="text-slate-900">{uploadedFileName}</strong>
                </span>
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="text-indigo-700 hover:underline font-medium"
                >
                  Replace File
                </button>
              </div>
            )}

            <div>
              <label htmlFor="real-data-textarea" className="sr-only">
                Real CSV, JSON, numeric stream, or unstructured text payload
              </label>
              <textarea
                id="real-data-textarea"
                rows={6}
                value={rawInput}
                onChange={(e) => {
                  setRawInput(e.target.value);
                  setInputError(null);
                  setFittedModels(null);
                }}
                placeholder="Paste real CSV rows with headers (include target_class for Confusion/PR-AUC eval), JSON array, or text..."
                className="w-full rounded-xl border border-slate-300 bg-slate-50 focus:bg-white focus:border-indigo-500 focus:outline-none p-3 font-mono text-xs text-slate-900 leading-relaxed"
              />
            </div>

            {(inputError || browserTrainError) && (
              <div className="rounded-xl border border-rose-300 bg-rose-50 px-3.5 py-2.5 text-xs text-rose-900 flex items-center gap-2">
                <ShieldAlert className="w-4 h-4 shrink-0 text-rose-700" />
                <span>{inputError || browserTrainError}</span>
              </div>
            )}
          </div>

          {/* Right: Complementary Method Pairing + Threshold & Conformal Controls */}
          <div className="lg:col-span-5 rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-3.5">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <GitMerge className="w-4 h-4 text-indigo-700" />
                <h4 className="text-sm font-bold text-slate-900">
                  Method Pairing, Threshold τ &amp; Conformal α
                </h4>
              </div>
              <span className="text-[11px] font-mono text-slate-500">
                1/ECE Stacking
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
              <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                <div className="text-[11px] text-slate-500 font-mono">Primary Method</div>
                <div className="font-bold text-slate-900 mt-0.5">
                  {ML_TECHNIQUE_META[mlTechnique].shortName}
                </div>
                <div className="text-[11px] text-slate-600 font-mono tabular-nums mt-0.5">
                  ECE {ML_TECHNIQUE_META[mlTechnique].eceScore}
                </div>
              </div>

              <div className="p-2.5 rounded-lg bg-white border border-slate-200">
                <label
                  htmlFor="complementary-method-select"
                  className="text-[11px] text-slate-500 font-mono block"
                >
                  Complementary Pair
                </label>
                <select
                  id="complementary-method-select"
                  value={complementaryMethod}
                  onChange={(e) => {
                    const nextComp = e.target.value as MlTechniqueId;
                    setComplementaryMethod(nextComp);
                    if (report) executeInference({ compId: nextComp });
                  }}
                  className="mt-1 w-full bg-slate-50 border border-slate-300 rounded-md px-2 py-1 text-xs font-semibold text-slate-900 focus:outline-none focus:border-indigo-500"
                >
                  {(Object.keys(ML_TECHNIQUE_META) as MlTechniqueId[])
                    .filter((id) => id !== mlTechnique)
                    .map((id) => (
                      <option key={id} value={id}>
                        {ML_TECHNIQUE_META[id].shortName}
                        {id === recommendedComp ? ' (Recommended)' : ''}
                      </option>
                    ))}
                </select>
              </div>
            </div>

            {/* Interactive Threshold & Conformal Controls */}
            <div className="rounded-lg bg-white border border-slate-200 p-3 space-y-2.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-slate-800 flex items-center gap-1.5">
                  <SlidersHorizontal className="w-3.5 h-3.5 text-indigo-700" />
                  <span>Decision Threshold τ = {decisionThreshold.toFixed(2)}</span>
                </span>
                {report && report.costSummary.optimalThresholdTau !== decisionThreshold && (
                  <button
                    type="button"
                    onClick={() => {
                      const opt = report.costSummary.optimalThresholdTau;
                      setDecisionThreshold(opt);
                      executeInference({ threshold: opt });
                    }}
                    className="text-[11px] font-mono text-indigo-700 font-semibold hover:underline"
                  >
                    Apply Cost-Optimal τ*={report.costSummary.optimalThresholdTau.toFixed(2)}
                  </button>
                )}
              </div>
              <input
                type="range"
                min={0.1}
                max={0.9}
                step={0.05}
                value={decisionThreshold}
                onChange={(e) => {
                  const nextTau = Number(e.target.value);
                  setDecisionThreshold(nextTau);
                  if (report) executeInference({ threshold: nextTau });
                }}
                className="w-full accent-indigo-600"
              />

              <div className="grid grid-cols-3 gap-2 pt-1">
                <label className="block">
                  <span className="text-[10px] font-mono text-slate-500 block">Conformal Coverage</span>
                  <select
                    value={conformalAlpha}
                    onChange={(e) => {
                      const nextAlpha = Number(e.target.value);
                      setConformalAlpha(nextAlpha);
                      if (report) executeInference({ alpha: nextAlpha });
                    }}
                    className="mt-0.5 w-full rounded border border-slate-300 bg-slate-50 px-1.5 py-1 font-mono text-[11px] text-slate-900"
                  >
                    <option value={0.05}>95% (α=0.05)</option>
                    <option value={0.1}>90% (α=0.10)</option>
                    <option value={0.2}>80% (α=0.20)</option>
                  </select>
                </label>

                <label className="block">
                  <span className="text-[10px] font-mono text-slate-500 block">FP Cost ($)</span>
                  <input
                    type="number"
                    min={0}
                    step={1}
                    value={fpCostInr}
                    onChange={(e) => {
                      const val = Math.max(0, Number(e.target.value) || 0);
                      setFpCostInr(val);
                      if (report) executeInference({ fpCost: val });
                    }}
                    className="mt-0.5 w-full rounded border border-slate-300 bg-slate-50 px-1.5 py-1 font-mono tabular-nums text-[11px] text-slate-900"
                  />
                </label>

                <label className="block">
                  <span className="text-[10px] font-mono text-slate-500 block">FN Loss ($)</span>
                  <input
                    type="number"
                    min={0}
                    step={10}
                    value={fnCostInr}
                    onChange={(e) => {
                      const val = Math.max(0, Number(e.target.value) || 0);
                      setFnCostInr(val);
                      if (report) executeInference({ fnCost: val });
                    }}
                    className="mt-0.5 w-full rounded border border-slate-300 bg-slate-50 px-1.5 py-1 font-mono tabular-nums text-[11px] text-slate-900"
                  />
                </label>
              </div>
            </div>

            {fittedModels && (
              <div className="rounded-lg bg-emerald-50 border border-emerald-200 p-2.5 text-[11px] text-emerald-950 font-mono space-y-1">
                <div className="font-bold">
                  Real Browser Weights Fitted ({fittedModels.fittedAt}) · Split {fittedModels.splitCounts.train}/{fittedModels.splitCounts.valCalibration}/{fittedModels.splitCounts.test}
                </div>
                <div>
                  AdamW Epoch {fittedModels.epochHistory.length} Loss {fittedModels.epochHistory[fittedModels.epochHistory.length - 1]?.trainLoss} → Test Ensemble {fittedModels.finalTestMetrics.ensembleAccPct}% (Brier {fittedModels.finalTestMetrics.brierScore})
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Empty State */}
        {!report && (
          <div className="rounded-xl border border-dashed border-slate-300 bg-slate-50/70 p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="space-y-1 text-center sm:text-left">
              <div className="text-sm font-bold text-slate-900">
                Real-Data Engine Ready ({liveParsePreview.records.length} parsed record{liveParsePreview.records.length === 1 ? '' : 's'})
              </div>
              <p className="text-xs text-slate-600">
                Click <strong>Fit Real MLP + GBDT on Data</strong> to run genuine browser gradient descent on Train/Cal/Test splits, or <strong>Run Real Inference</strong> to evaluate immediately.
              </p>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={handleFitInBrowserModels}
                disabled={isTraining || liveParsePreview.records.length < 3 || liveParsePreview.numericColumns.length === 0}
                className="px-3.5 py-2 rounded-lg text-xs font-semibold bg-emerald-800 hover:bg-emerald-700 disabled:bg-slate-300 text-white whitespace-nowrap"
              >
                Fit Real Models
              </button>
              <button
                type="button"
                onClick={() => executeInference({ autoSaveRun: true })}
                disabled={isTraining || liveParsePreview.records.length === 0}
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white whitespace-nowrap"
              >
                Score Dataset
              </button>
            </div>
          </div>
        )}

        {/* Active Report Workbench */}
        {report && activeRecord && (
          <div className="space-y-5 pt-3 border-t border-slate-200">
            {/* Navigation Sub-Tabs for Clean Information Architecture */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
              <div className="inline-flex flex-wrap items-center gap-1 p-1 bg-slate-100 rounded-lg border border-slate-200">
                {(
                  [
                    { id: 'inference', label: '1. Inference, Conformal & Cost Tuning' },
                    { id: 'profiler', label: `2. Dataset Profiler (KS · PSI · VIF)` },
                    { id: 'evaluation', label: '3. Ground-Truth Eval & Reliability' },
                    { id: 'experiments', label: `4. Saved Runs & A/B Diff (${savedRuns.length})` }
                  ] as const
                ).map((t) => (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => setActiveSubTab(t.id)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors whitespace-nowrap ${
                      activeSubTab === t.id
                        ? 'bg-white text-slate-900 shadow-xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {t.label}
                  </button>
                ))}
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <button
                  type="button"
                  onClick={() => {
                    if (report) void persistSnapshotForReport(report, report.usedInBrowserFittedWeights, rawInput);
                  }}
                  className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-xs font-medium text-slate-800 flex items-center gap-1.5 whitespace-nowrap"
                >
                  <Save className="w-3.5 h-3.5 text-indigo-700" />
                  <span>Save Snapshot</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleExportResults('csv')}
                  className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-xs font-medium text-slate-800 flex items-center gap-1.5 whitespace-nowrap"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>CSV</span>
                </button>
                <button
                  type="button"
                  onClick={() => handleExportResults('json')}
                  className="px-3 py-1.5 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 text-xs font-medium text-slate-800 flex items-center gap-1.5 whitespace-nowrap"
                >
                  <Download className="w-3.5 h-3.5" />
                  <span>JSON</span>
                </button>
              </div>
            </div>

            {/* SUB-TAB 1: INFERENCE, CONFORMAL SETS & COST OPTIMIZATION */}
            {activeSubTab === 'inference' && (
              <div className="space-y-5">
                {/* 4 KPI Strip */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-1">
                    <span className="text-[11px] font-mono text-slate-500">Batch Decision Split (τ={report.costSummary.decisionThreshold.toFixed(2)})</span>
                    <div className="text-base font-bold text-slate-900 font-mono tabular-nums">
                      {report.positiveCount} Alert · {report.reviewCount} Review · {report.nominalCount} Clear
                    </div>
                    <p className="text-[11px] text-slate-600">
                      {report.disagreementCount === 0
                        ? 'Primary & complementary aligned'
                        : `${report.disagreementCount} pair conflict${report.disagreementCount > 1 ? 's' : ''}`}
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-1">
                    <span className="text-[11px] font-mono text-slate-500">Conformal Coverage ({Math.round((1 - report.conformalAlpha) * 100)}%)</span>
                    <div className="text-base font-bold text-indigo-800 font-mono tabular-nums">
                      Mean Conf {report.meanConfidencePct}% · Set Size {report.meanConformalSetSize}
                    </div>
                    <p className="text-[11px] text-slate-600">
                      q̂={report.conformalQHat} · {report.ambiguousSetCount} ambiguous dual-class set{report.ambiguousSetCount === 1 ? '' : 's'}
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-1">
                    <span className="text-[11px] font-mono text-slate-500 flex items-center gap-1">
                      <DollarSign className="w-3 h-3" />
                      <span>Cost-Matrix Net Impact</span>
                    </span>
                    <div className="text-sm font-bold text-emerald-800 font-mono tabular-nums">
                      {formatDualCurrency(report.costSummary.netSavingsInr)}
                    </div>
                    <p className="text-[11px] text-slate-600">
                      OpEx Cost {formatDualCurrency(report.costSummary.expectedOperationalCostInr)} · Opt τ*={report.costSummary.optimalThresholdTau.toFixed(2)}
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-1">
                    <span className="text-[11px] font-mono text-slate-500">5-Method Consensus &amp; P95</span>
                    <div className="text-base font-bold text-slate-900 font-mono tabular-nums">
                      {report.meanConsensusPct}% Agree · {report.batchP95LatencyMs} ms
                    </div>
                    <p className="text-[11px] text-slate-600">
                      {report.usedInBrowserFittedWeights ? 'Using Real Fitted Weights · ' : ''}
                      {report.slaWithinBudget ? 'Within <45ms SLA' : 'Exceeds 45ms SLA'}
                    </p>
                  </div>
                </div>

                {/* Record Deep Dive + 5-Method Matrix */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
                  <div className="lg:col-span-6 rounded-xl border border-slate-300 bg-white p-4 space-y-4">
                    <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-200">
                      <div>
                        <div className="text-[11px] font-mono text-slate-500">
                          Row #{activeRecord.rowIndex + 1} · {activeRecord.observedLabel !== undefined ? `Observed Truth = ${activeRecord.observedLabel}` : 'Unlabeled Inference'}
                        </div>
                        <h4 className="text-sm font-bold text-slate-900 font-mono">
                          {activeRecord.recordId}
                        </h4>
                      </div>

                      <div className="flex items-center gap-2 text-xs font-semibold">
                        {activeRecord.decisionTier === 'alert_escalate' ? (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-rose-50 text-rose-900 border border-rose-300">
                            <ShieldAlert className="w-3.5 h-3.5 text-rose-700" />
                            <span>{activeRecord.ensembleResult.predictedLabel}</span>
                          </span>
                        ) : activeRecord.decisionTier === 'review_step_up' ? (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-50 text-amber-900 border border-amber-300">
                            <AlertTriangle className="w-3.5 h-3.5 text-amber-700" />
                            <span>Step-Up Review</span>
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-50 text-emerald-900 border border-emerald-300">
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-700" />
                            <span>{activeRecord.ensembleResult.predictedLabel}</span>
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-slate-700">Calibrated Confidence &amp; Conformal Set</span>
                        <span className="font-mono tabular-nums font-bold text-slate-900">
                          {activeRecord.ensembleResult.confidencePct}%{' '}
                          <span className={activeRecord.ensembleResult.confidenceLiftPct >= 0 ? 'text-emerald-700' : 'text-amber-700'}>
                            ({activeRecord.ensembleResult.confidenceLiftPct >= 0 ? '+' : ''}
                            {activeRecord.ensembleResult.confidenceLiftPct}%)
                          </span>
                        </span>
                      </div>
                      <div className="w-full h-3 rounded-full bg-slate-200 border border-slate-300 overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            activeRecord.decisionTier === 'alert_escalate'
                              ? 'bg-rose-600'
                              : activeRecord.decisionTier === 'review_step_up'
                                ? 'bg-amber-500'
                                : 'bg-emerald-600'
                          }`}
                          style={{ width: `${Math.min(100, activeRecord.ensembleResult.confidencePct)}%` }}
                        />
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono tabular-nums text-slate-600">
                        <span>P(Target=1)={(activeRecord.ensembleResult.probability * 100).toFixed(1)}%</span>
                        <span>Conformal Set ({activeRecord.conformal.targetCoveragePct}%): <strong className="text-slate-900">{`{${activeRecord.conformal.predictionSet.join(', ')}}`}</strong></span>
                        <span>OOD {activeRecord.oodDriftZ}σ</span>
                      </div>
                    </div>

                    {activeRecord.ensembleResult.forecastValue !== undefined &&
                      activeRecord.ensembleResult.forecastInterval && (
                        <div className="rounded-lg bg-indigo-50 border border-indigo-200 px-3 py-2 text-xs flex items-center justify-between">
                          <span className="font-semibold text-indigo-950">Conformal Forecast Interval</span>
                          <span className="font-mono tabular-nums font-bold text-indigo-900">
                            ŷ={activeRecord.ensembleResult.forecastValue} · [{activeRecord.ensembleResult.forecastInterval[0]}, {activeRecord.ensembleResult.forecastInterval[1]}]
                          </span>
                        </div>
                      )}

                    <div className="rounded-xl bg-slate-50 border border-slate-200 p-3.5 space-y-1">
                      <div className="text-xs font-mono font-semibold text-slate-600">Recommended Action</div>
                      <div className="text-sm font-bold text-slate-900">
                        {activeRecord.recommendationTitle}
                      </div>
                      <p className="text-xs text-slate-700 leading-relaxed">
                        {activeRecord.recommendationDetail}
                      </p>
                    </div>

                    <div className="space-y-2.5">
                      <div className="text-xs font-bold text-slate-900">Top Feature Drivers (Log-Odds Contribution)</div>
                      <div className="space-y-2">
                        {activeRecord.topDrivers.map((d) => {
                          const pctWidth = Math.min(100, Math.round((Math.abs(d.contribution) / 2.5) * 100));
                          return (
                            <div key={d.feature} className="grid grid-cols-12 items-center gap-2 text-xs font-mono">
                              <span className="col-span-4 truncate text-slate-800 font-semibold" title={d.feature}>
                                {d.feature}
                              </span>
                              <span className="col-span-3 truncate text-slate-600 tabular-nums" title={d.valueDisplay}>
                                {d.valueDisplay}
                              </span>
                              <div className="col-span-3 h-2.5 bg-slate-200 border border-slate-300 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${d.direction === 'risk_up' ? 'bg-rose-600' : 'bg-emerald-600'}`}
                                  style={{ width: `${Math.max(8, pctWidth)}%` }}
                                />
                              </div>
                              <span
                                className={`col-span-2 text-right tabular-nums font-bold ${
                                  d.direction === 'risk_up' ? 'text-rose-800' : 'text-emerald-800'
                                }`}
                              >
                                {d.contribution >= 0 ? '+' : ''}
                                {d.contribution}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>

                  {/* 5-Method Comparison Table */}
                  <div className="lg:col-span-6 rounded-xl border border-slate-300 bg-white overflow-hidden">
                    <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <ArrowRightLeft className="w-4 h-4 text-indigo-700" />
                        <h4 className="text-sm font-bold text-slate-900">
                          5-Method Comparison on {activeRecord.recordId}
                        </h4>
                      </div>
                      <span className="text-[11px] font-mono text-slate-600">
                        {activeRecord.ensembleResult.pairAgreed ? 'Pair Aligned' : 'Pair Diverged'}
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse text-xs">
                        <thead>
                          <tr className="bg-slate-100 border-b border-slate-200 text-slate-700 font-mono text-[11px]">
                            <th className="py-2 px-3">Method &amp; Role</th>
                            <th className="py-2 px-3">Prediction</th>
                            <th className="py-2 px-3 text-right">P(Target)</th>
                            <th className="py-2 px-3 text-right">Conf</th>
                            <th className="py-2 px-3 text-right">Entropy</th>
                            <th className="py-2 px-3 text-right">P95</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-200">
                          {activeRecord.allMethodScores.map((ms) => {
                            const isPrimary = ms.methodId === report.primaryMethodId;
                            const isComp = ms.methodId === report.complementaryMethodId;
                            return (
                              <tr
                                key={ms.methodId}
                                className={
                                  isPrimary
                                    ? 'bg-indigo-50/60'
                                    : isComp
                                      ? 'bg-emerald-50/50'
                                      : 'hover:bg-slate-50'
                                }
                              >
                                <td className="py-2 px-3">
                                  <div className="font-semibold text-slate-900">{ms.shortName}</div>
                                  <div className="text-[10px] font-mono text-slate-500">
                                    {isPrimary ? 'Primary' : isComp ? 'Complementary' : 'Peer'}
                                    {ms.isTrainedInBrowser ? ' · Fitted' : ''}
                                  </div>
                                </td>
                                <td className="py-2 px-3">
                                  <span className={`font-semibold ${ms.predictedClass === 1 ? 'text-rose-800' : 'text-emerald-800'}`}>
                                    {ms.predictedLabel}
                                  </span>
                                </td>
                                <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-800">
                                  {(ms.probability * 100).toFixed(1)}%
                                </td>
                                <td className="py-2 px-3 text-right font-mono tabular-nums font-bold text-slate-900">
                                  {ms.confidencePct}%
                                </td>
                                <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-600">
                                  {ms.entropyBits}b
                                </td>
                                <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                                  {ms.latencyP95Ms}ms
                                </td>
                              </tr>
                            );
                          })}
                          <tr className="bg-slate-900 text-white">
                            <td className="py-2.5 px-3">
                              <div className="font-bold">Complementary Stacked Fusion</div>
                              <div className="text-[10px] font-mono text-slate-300">
                                Conformal q̂={activeRecord.conformal.qHatThreshold}
                              </div>
                            </td>
                            <td className="py-2.5 px-3 font-semibold">
                              {activeRecord.ensembleResult.predictedLabel}
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                              {(activeRecord.ensembleResult.probability * 100).toFixed(1)}%
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono tabular-nums font-bold text-emerald-300">
                              {activeRecord.ensembleResult.confidencePct}%
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-300">
                              {activeRecord.ensembleResult.consensusCount}/5 agree
                            </td>
                            <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-200">
                              {report.batchP95LatencyMs}ms
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>

                    <div className="p-3 bg-slate-50 border-t border-slate-200 text-[11px] text-slate-700">
                      <strong>Complementarity Synergy:</strong> {report.complementarityRationale}
                    </div>
                  </div>
                </div>

                {/* Record Ledger Table */}
                <div className="rounded-xl border border-slate-300 overflow-hidden">
                  <div className="px-4 py-2.5 bg-slate-100 border-b border-slate-200 flex flex-wrap items-center justify-between gap-2 text-xs">
                    <span className="font-bold text-slate-900">
                      Real-Data Batch Inference Ledger ({report.totalRecords} Records)
                    </span>
                    <span className="text-slate-600 font-mono text-[11px]">
                      Click any row to inspect feature drivers &amp; method comparison
                    </span>
                  </div>
                  <div className="overflow-x-auto max-h-64 bg-white">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-200 font-mono text-[11px] text-slate-700">
                          <th className="py-2 px-3">Record ID</th>
                          <th className="py-2 px-3">Ensemble Result</th>
                          <th className="py-2 px-3">Conformal Set ({Math.round((1 - report.conformalAlpha) * 100)}%)</th>
                          <th className="py-2 px-3 text-right">Conf</th>
                          <th className="py-2 px-3 text-right">Agree</th>
                          <th className="py-2 px-3">Action Recommendation</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200">
                        {report.recordResults.map((rec, idx) => (
                          <tr
                            key={`${rec.recordId}-${idx}`}
                            onClick={() => setSelectedRecordIdx(idx)}
                            className={`cursor-pointer transition-colors ${
                              idx === selectedRecordIdx ? 'bg-indigo-50/80' : 'hover:bg-slate-50'
                            }`}
                          >
                            <td className="py-2 px-3 font-mono font-semibold text-slate-900 whitespace-nowrap">
                              {rec.recordId}
                            </td>
                            <td className="py-2 px-3 whitespace-nowrap">
                              <span
                                className={`font-semibold ${
                                  rec.decisionTier === 'alert_escalate'
                                    ? 'text-rose-800'
                                    : rec.decisionTier === 'review_step_up'
                                      ? 'text-amber-800'
                                      : 'text-emerald-800'
                                }`}
                              >
                                {rec.ensembleResult.predictedLabel}
                              </span>
                            </td>
                            <td className="py-2 px-3 font-mono text-[11px] text-slate-700 whitespace-nowrap">
                              {`{${rec.conformal.predictionSet.join(', ')}}`}
                            </td>
                            <td className="py-2 px-3 text-right font-mono tabular-nums font-bold text-slate-900">
                              {rec.ensembleResult.confidencePct}%
                            </td>
                            <td className="py-2 px-3 text-right font-mono tabular-nums text-slate-700">
                              {rec.ensembleResult.consensusCount}/5
                            </td>
                            <td className="py-2 px-3 text-slate-800 max-w-xs truncate">
                              {rec.recommendationTitle}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* SUB-TAB 2: FORMAL STATISTICAL DATASET PROFILER (KS, PSI, VIF, MI, IMBALANCE) */}
            {activeSubTab === 'profiler' && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-1">
                    <span className="text-[11px] font-mono text-slate-500">Max Feature PSI Drift</span>
                    <div className="text-lg font-bold font-mono tabular-nums text-slate-900">
                      {report.datasetProfile.overallPsiMax}
                    </div>
                    <p className="text-[11px] text-slate-600">
                      {report.datasetProfile.driftedFeaturesCount} shifted feature(s) across split halves
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-1">
                    <span className="text-[11px] font-mono text-slate-500">Multicollinearity (VIF &gt; 10)</span>
                    <div className="text-lg font-bold font-mono tabular-nums text-slate-900">
                      {report.datasetProfile.multicollinearCount} Features
                    </div>
                    <p className="text-[11px] text-slate-600">
                      Computed via Gauss-Jordan correlation inverse R⁻¹
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-1">
                    <span className="text-[11px] font-mono text-slate-500">Class Imbalance Ratio</span>
                    <div className="text-lg font-bold font-mono tabular-nums text-indigo-900">
                      {report.datasetProfile.imbalance.imbalanceRatio} ({report.datasetProfile.imbalance.positiveRatePct}% Pos)
                    </div>
                    <p className="text-[11px] text-slate-600">
                      {report.datasetProfile.imbalance.severity} · Rec pos_weight={report.datasetProfile.imbalance.recommendedPosWeight}
                    </p>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3.5 space-y-1">
                    <span className="text-[11px] font-mono text-slate-500">Focal Loss &amp; Completeness</span>
                    <div className="text-lg font-bold font-mono tabular-nums text-emerald-800">
                      α={report.datasetProfile.imbalance.recommendedFocalAlpha} · {report.datasetProfile.completenessPct}% Complete
                    </div>
                    <p className="text-[11px] text-slate-600">
                      {report.datasetProfile.imbalance.hasGroundTruthTarget
                        ? `Target col: ${report.datasetProfile.imbalance.targetColumnName}`
                        : 'Unsupervised proxy target for MI ranking'}
                    </p>
                  </div>
                </div>

                <div className="rounded-xl border border-slate-300 overflow-hidden bg-white">
                  <div className="px-4 py-2.5 bg-slate-100 border-b border-slate-200 text-xs font-bold text-slate-900">
                    Per-Feature Statistical Diagnostics (Mean/Std, PSI, Kolmogorov-Smirnov p-value, VIF, Mutual Info)
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-200 font-mono text-[11px] text-slate-700">
                          <th className="py-2 px-3">Feature</th>
                          <th className="py-2 px-3 text-right">Mean ± Std</th>
                          <th className="py-2 px-3 text-right">Range [Min, Max]</th>
                          <th className="py-2 px-3 text-right">PSI Drift</th>
                          <th className="py-2 px-3 text-right">KS D (p-val)</th>
                          <th className="py-2 px-3 text-right">VIF</th>
                          <th className="py-2 px-3 text-right">Mutual Info</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
                        {report.datasetProfile.features.length === 0 ? (
                          <tr>
                            <td colSpan={7} className="py-6 text-center text-slate-500">
                              No numeric feature columns detected in current payload (text-only corpus).
                            </td>
                          </tr>
                        ) : (
                          report.datasetProfile.features.map((f) => (
                            <tr key={f.feature} className="hover:bg-slate-50">
                              <td className="py-2 px-3 font-bold text-slate-900">{f.feature}</td>
                              <td className="py-2 px-3 text-right text-slate-700">
                                {f.mean} ± {f.std}
                              </td>
                              <td className="py-2 px-3 text-right text-slate-600">
                                [{f.min}, {f.max}]
                              </td>
                              <td
                                className={`py-2 px-3 text-right font-semibold ${
                                  f.psiScore > 0.25
                                    ? 'text-rose-800'
                                    : f.psiScore >= 0.1
                                      ? 'text-amber-800'
                                      : 'text-emerald-800'
                                }`}
                              >
                                {f.psiScore}
                              </td>
                              <td className="py-2 px-3 text-right text-slate-800">
                                {f.ksStatistic} (p={f.ksPValue})
                              </td>
                              <td
                                className={`py-2 px-3 text-right font-semibold ${
                                  f.vif > 10 ? 'text-rose-800' : f.vif >= 5 ? 'text-amber-800' : 'text-slate-800'
                                }`}
                              >
                                {f.vif}
                              </td>
                              <td className="py-2 px-3 text-right font-bold text-indigo-900">
                                {f.mutualInfoBits} bits
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}

            {/* SUB-TAB 3: GROUND-TRUTH CONFUSION MATRIX, PR-AUC, RELIABILITY DIAGRAM & FITTED WEIGHTS */}
            {activeSubTab === 'evaluation' && (
              <div className="space-y-5">
                {!report.groundTruthEval ? (
                  <div className="rounded-xl border border-amber-300 bg-amber-50 p-5 space-y-2 text-xs text-amber-950">
                    <div className="font-bold text-sm">
                      No Ground-Truth Label Column Detected (`target_class`, `label`, `y`)
                    </div>
                    <p>
                      Include a binary `target_class` column (0/1) in your CSV or JSON — or pick any Labeled Cohort preset above — to compute exact Confusion Matrix, ROC-AUC, PR-AUC, Brier Score, Empirical ECE, and a 5-bin Reliability Diagram.
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
                    {/* Left: Metrics + Confusion Matrix */}
                    <div className="lg:col-span-6 space-y-4">
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                          <div className="text-[10px] font-mono text-slate-500">Accuracy</div>
                          <div className="text-lg font-bold font-mono tabular-nums text-slate-900">
                            {report.groundTruthEval.accuracyPct}%
                          </div>
                        </div>
                        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                          <div className="text-[10px] font-mono text-slate-500">ROC-AUC / PR-AUC</div>
                          <div className="text-lg font-bold font-mono tabular-nums text-indigo-900">
                            {report.groundTruthEval.rocAuc} / {report.groundTruthEval.prAuc}
                          </div>
                        </div>
                        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                          <div className="text-[10px] font-mono text-slate-500">F1 (P / R)</div>
                          <div className="text-base font-bold font-mono tabular-nums text-emerald-800">
                            {report.groundTruthEval.f1Pct}% ({report.groundTruthEval.precisionPct}/{report.groundTruthEval.recallPct})
                          </div>
                        </div>
                        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
                          <div className="text-[10px] font-mono text-slate-500">Brier / Emp ECE</div>
                          <div className="text-base font-bold font-mono tabular-nums text-slate-900">
                            {report.groundTruthEval.brierScore} / {report.groundTruthEval.empiricalEcePct}%
                          </div>
                        </div>
                      </div>

                      {/* 2x2 Confusion Matrix */}
                      <div className="rounded-xl border border-slate-300 bg-white p-4 space-y-3">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-bold text-slate-900">
                            Empirical Confusion Matrix ({report.groundTruthEval.labeledCount} rows @ τ={report.costSummary.decisionThreshold.toFixed(2)})
                          </span>
                          <span className="font-mono text-[11px] text-slate-500">
                            Target: {report.groundTruthEval.targetColumnName}
                          </span>
                        </div>
                        <div className="grid grid-cols-2 gap-2.5 text-center font-mono tabular-nums">
                          <div className="rounded-lg bg-emerald-50 border border-emerald-300 p-3">
                            <div className="text-[11px] text-emerald-800">True Positive (TP)</div>
                            <div className="text-2xl font-bold text-emerald-950">{report.groundTruthEval.tp}</div>
                          </div>
                          <div className="rounded-lg bg-amber-50 border border-amber-300 p-3">
                            <div className="text-[11px] text-amber-800">False Positive (FP)</div>
                            <div className="text-2xl font-bold text-amber-950">{report.groundTruthEval.fp}</div>
                          </div>
                          <div className="rounded-lg bg-rose-50 border border-rose-300 p-3">
                            <div className="text-[11px] text-rose-800">False Negative (FN)</div>
                            <div className="text-2xl font-bold text-rose-950">{report.groundTruthEval.fn}</div>
                          </div>
                          <div className="rounded-lg bg-slate-50 border border-slate-300 p-3">
                            <div className="text-[11px] text-slate-700">True Negative (TN)</div>
                            <div className="text-2xl font-bold text-slate-950">{report.groundTruthEval.tn}</div>
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Right: SVG Reliability Diagram (Confidence Calibration Curve) */}
                    <div className="lg:col-span-6 rounded-xl border border-slate-300 bg-white p-4 space-y-3">
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                        <div className="flex items-center gap-2">
                          <BarChart3 className="w-4 h-4 text-indigo-700" />
                          <span className="font-bold text-slate-900 text-sm">
                            Empirical Reliability Diagram (Calibration vs Ideal y=x)
                          </span>
                        </div>
                        <span className="font-mono text-xs font-bold text-indigo-900 tabular-nums">
                          ECE {report.groundTruthEval.empiricalEcePct}%
                        </span>
                      </div>

                      <svg viewBox="0 0 480 245" className="w-full h-64 bg-slate-50 rounded-lg border border-slate-200 overflow-visible">
                        {/* Grid lines */}
                        {[0, 25, 50, 75, 100].map((v) => {
                          const y = 196 - (v / 100) * 160;
                          return (
                            <g key={v}>
                              <line
                                x1="58"
                                y1={y}
                                x2="452"
                                y2={y}
                                stroke={v === 0 ? CHART_COLORS.stageStroke : CHART_COLORS.gridLine}
                                strokeDasharray={v === 0 ? undefined : '4 4'}
                              />
                              <text
                                x="50"
                                y={y + 4}
                                textAnchor="end"
                                fontSize="11.5"
                                fontFamily={CHART_FONTS.mono}
                                fontWeight="600"
                                fill={CHART_COLORS.tickText}
                              >
                                {v}%
                              </text>
                            </g>
                          );
                        })}
                        {/* Ideal Calibration Diagonal */}
                        <line x1="58" y1="196" x2="452" y2="36" stroke="#64748b" strokeWidth="1.8" strokeDasharray="6 4" />
                        <text x="448" y="28" textAnchor="end" fontSize="11" fontFamily={CHART_FONTS.mono} fontWeight="700" fill={CHART_COLORS.tickText}>
                          Ideal y = x
                        </text>
                        <text x="64" y="20" fontSize="11" fontFamily={CHART_FONTS.sans} fontWeight="700" fill={CHART_COLORS.axisYLabel}>
                          +Y: Empirical Accuracy (%)
                        </text>

                        {report.groundTruthEval.reliabilityBins.map((bin, bIdx, arr) => {
                          const slotW = 382 / Math.max(1, arr.length);
                          const x = 66 + bIdx * slotW + 8;
                          const barW = slotW - 18;
                          const accH = bin.count > 0 ? Math.max(6, (bin.empiricalAccPct / 100) * 160) : 0;
                          const confH = bin.count > 0 ? Math.max(6, (bin.avgConfidencePct / 100) * 160) : 0;
                          return (
                            <g key={bin.binLabel}>
                              {bin.count > 0 && (
                                <>
                                  <rect
                                    x={x}
                                    y={196 - confH}
                                    width={barW * 0.45}
                                    height={confH}
                                    fill="#94a3b8"
                                    rx="2.5"
                                  />
                                  <rect
                                    x={x + barW * 0.49}
                                    y={196 - accH}
                                    width={barW * 0.45}
                                    height={accH}
                                    fill="#4f46e5"
                                    rx="2.5"
                                  />
                                  <text
                                    x={x + barW / 2}
                                    y={Math.max(24, 188 - Math.max(accH, confH))}
                                    textAnchor="middle"
                                    fontSize="11"
                                    fontFamily={CHART_FONTS.mono}
                                    fontWeight="700"
                                    fill="#1e1b4b"
                                    style={{ paintOrder: 'stroke', stroke: '#f8fafc', strokeWidth: '3px' }}
                                  >
                                    n={bin.count}
                                  </text>
                                </>
                              )}
                              <text
                                x={x + barW / 2}
                                y="220"
                                textAnchor="middle"
                                fontSize="11"
                                fontFamily={CHART_FONTS.mono}
                                fontWeight="600"
                                fill={CHART_COLORS.tickText}
                              >
                                {bin.binLabel}
                              </text>
                            </g>
                          );
                        })}
                        <text x="255" y="239" textAnchor="middle" fontSize="11" fontFamily={CHART_FONTS.sans} fontWeight="600" fill={CHART_COLORS.mutedText}>
                          Predicted Confidence Probability Bin
                        </text>
                      </svg>
                      <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-slate-700">
                        <div className="flex items-center gap-3">
                          <span className="inline-flex items-center gap-1.5">
                            <span className="w-2.5 h-2.5 rounded-xs bg-slate-400 inline-block" /> Predicted Bin
                          </span>
                          <span className="inline-flex items-center gap-1.5 font-semibold text-indigo-900">
                            <span className="w-2.5 h-2.5 rounded-xs bg-indigo-600 inline-block" /> Observed Accuracy
                          </span>
                        </div>
                        <span className="font-bold tabular-nums">Brier: {report.groundTruthEval.brierScore}</span>
                      </div>
                    </div>
                  </div>
                )}

                {/* In-Browser Fitted Neural + GBDT Epoch Convergence Readout */}
                {fittedModels && (
                  <div className="rounded-xl border border-slate-300 bg-white p-4 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                      <span className="font-bold text-slate-900 flex items-center gap-1.5">
                        <Activity className="w-4 h-4 text-emerald-700" />
                        <span>Real In-Browser Fitted Weights (AdamW Neural Logistic + {fittedModels.gbdtStumps.length} GBDT Residual Stumps)</span>
                      </span>
                      <span className="font-mono text-[11px] text-slate-600">
                        Target: {fittedModels.targetColumnName} ({fittedModels.hasObservedGroundTruth ? 'Observed Truth' : 'Proxy Z-Supervision'})
                      </span>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                      <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 space-y-1.5">
                        <div className="font-semibold text-slate-800">Learned AdamW Feature Weights (Standardized Z)</div>
                        <div className="space-y-1 font-mono text-[11px]">
                          {fittedModels.featureNames.map((fn, idx) => (
                            <div key={fn} className="flex items-center justify-between">
                              <span className="text-slate-700 truncate">{fn}</span>
                              <span className="font-bold text-slate-900 tabular-nums">
                                w = {fittedModels.neuralWeights[idx]}
                              </span>
                            </div>
                          ))}
                          <div className="flex items-center justify-between pt-1 border-t border-slate-200 text-slate-600">
                            <span>Intercept Bias b</span>
                            <span className="tabular-nums">{fittedModels.neuralBias}</span>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 space-y-1.5">
                        <div className="font-semibold text-slate-800">Top GBDT Residual Decision Stumps</div>
                        <div className="space-y-1 font-mono text-[11px]">
                          {fittedModels.gbdtStumps.slice(0, 5).map((st, idx) => (
                            <div key={idx} className="flex items-center justify-between">
                              <span className="text-slate-700 truncate">
                                #{idx + 1}: {st.featureName} ≤ {st.threshold}
                              </span>
                              <span className="text-indigo-900 tabular-nums">
                                L={st.leftVal} / R={st.rightVal}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* SUB-TAB 4: EXPERIMENT PERSISTENCE LEDGER & A/B RUN DIFFING */}
            {activeSubTab === 'experiments' && (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <div className="font-mono text-slate-600">
                    Latest Dataset SHA-256: <strong className="text-slate-900">{lastDatasetHash}</strong> · Config SHA-256: <strong className="text-slate-900">{lastConfigHash}</strong>
                  </div>
                  {savedRuns.length > 0 && (
                    <button
                      type="button"
                      onClick={() => setSavedRuns(clearAllExperimentRuns())}
                      className="px-2.5 py-1 rounded border border-rose-300 bg-rose-50 hover:bg-rose-100 text-rose-900 font-medium text-[11px]"
                    >
                      Clear Saved Runs
                    </button>
                  )}
                </div>

                {/* A/B Diff Comparator */}
                {savedRuns.length >= 2 && (
                  <div className="rounded-xl border border-slate-300 bg-white overflow-hidden">
                    <div className="px-4 py-3 bg-slate-100 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-2 text-xs font-bold text-slate-900">
                        <GitCompare className="w-4 h-4 text-indigo-700" />
                        <span>A/B Experiment Diff Comparator</span>
                      </div>

                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        <select
                          value={runAId}
                          onChange={(e) => setRunAId(e.target.value)}
                          className="rounded border border-slate-300 bg-white px-2 py-1 font-mono text-[11px] text-slate-900"
                        >
                          {savedRuns.map((r) => (
                            <option key={r.runId} value={r.runId}>
                              Run A: {r.createdAt} · {ML_TECHNIQUE_META[r.primaryMethod].shortName} (τ={r.decisionThreshold})
                            </option>
                          ))}
                        </select>
                        <span className="font-mono text-slate-500">vs</span>
                        <select
                          value={runBId}
                          onChange={(e) => setRunBId(e.target.value)}
                          className="rounded border border-slate-300 bg-white px-2 py-1 font-mono text-[11px] text-slate-900"
                        >
                          {savedRuns.map((r) => (
                            <option key={r.runId} value={r.runId}>
                              Run B: {r.createdAt} · {ML_TECHNIQUE_META[r.primaryMethod].shortName} (τ={r.decisionThreshold})
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse text-xs">
                        <thead>
                          <tr className="bg-slate-50 border-b border-slate-200 font-mono text-[11px] text-slate-700">
                            <th className="py-2 px-3">Metric</th>
                            <th className="py-2 px-3 text-right">Run A</th>
                            <th className="py-2 px-3 text-right">Run B</th>
                            <th className="py-2 px-3 text-right">Delta (B - A)</th>
                            <th className="py-2 px-3 text-right">Winner</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
                          {abDiffRows.map((row) => (
                            <tr key={row.label} className="hover:bg-slate-50">
                              <td className="py-2 px-3 font-sans font-semibold text-slate-900">{row.label}</td>
                              <td className="py-2 px-3 text-right text-slate-700">{row.runAVal}</td>
                              <td className="py-2 px-3 text-right text-slate-900 font-bold">{row.runBVal}</td>
                              <td className="py-2 px-3 text-right text-indigo-900">{row.deltaDisplay}</td>
                              <td className="py-2 px-3 text-right font-bold">
                                <span
                                  className={
                                    row.winner === 'B'
                                      ? 'text-emerald-800'
                                      : row.winner === 'A'
                                        ? 'text-amber-800'
                                        : 'text-slate-500'
                                  }
                                >
                                  {row.winner === 'Tie' ? 'Tie' : `Run ${row.winner}`}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Persisted Run History */}
                <div className="rounded-xl border border-slate-300 bg-white overflow-hidden">
                  <div className="px-4 py-2.5 bg-slate-100 border-b border-slate-200 text-xs font-bold text-slate-900">
                    Persisted Experiment Snapshots (localStorage · {savedRuns.length} Saved)
                  </div>
                  <div className="overflow-x-auto max-h-60">
                    <table className="w-full text-left border-collapse text-xs">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-200 font-mono text-[11px] text-slate-700">
                          <th className="py-2 px-3">Time &amp; SHA-256</th>
                          <th className="py-2 px-3">Methods</th>
                          <th className="py-2 px-3 text-right">Conf / Agree</th>
                          <th className="py-2 px-3 text-right">PR-AUC</th>
                          <th className="py-2 px-3 text-right">Net ($ / Rs)</th>
                          <th className="py-2 px-3 text-right">Remove</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 font-mono tabular-nums text-[11px]">
                        {savedRuns.length === 0 ? (
                          <tr>
                            <td colSpan={6} className="py-6 text-center text-slate-500 font-sans">
                              Run inference or click Save Snapshot to persist reproducible SHA-256 experiment runs.
                            </td>
                          </tr>
                        ) : (
                          savedRuns.map((run) => (
                            <tr key={run.runId} className="hover:bg-slate-50">
                              <td className="py-2 px-3">
                                <div className="font-bold text-slate-900">{run.createdAt}</div>
                                <div className="text-[10px] text-slate-500">
                                  data:{run.datasetSha256.slice(0, 8)} · cfg:{run.configSha256.slice(0, 8)}
                                </div>
                              </td>
                              <td className="py-2 px-3 font-sans">
                                <div className="font-semibold text-slate-900">
                                  {ML_TECHNIQUE_META[run.primaryMethod].shortName} + {ML_TECHNIQUE_META[run.complementaryMethod].shortName}
                                </div>
                                <div className="text-[10px] font-mono text-slate-500">
                                  {run.inBrowserFitted ? 'In-Browser Fitted' : 'Baseline'} · τ={run.decisionThreshold}
                                </div>
                              </td>
                              <td className="py-2 px-3 text-right">
                                {run.meanConfidencePct}% / {run.meanConsensusPct}%
                              </td>
                              <td className="py-2 px-3 text-right">
                                {run.prAucPct !== undefined ? `${run.prAucPct}%` : '—'}
                              </td>
                              <td className="py-2 px-3 text-right font-bold text-emerald-800 whitespace-nowrap">
                                {formatDualCurrency(run.estimatedNetValueInr)}
                              </td>
                              <td className="py-2 px-3 text-right">
                                <button
                                  type="button"
                                  onClick={() => setSavedRuns(deleteExperimentRun(run.runId))}
                                  title="Delete experiment snapshot"
                                  className="p-1 rounded hover:bg-rose-100 text-slate-500 hover:text-rose-900"
                                >
                                  <Trash2 className="w-3.5 h-3.5" />
                                </button>
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
