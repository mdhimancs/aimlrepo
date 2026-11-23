import { SyntheticDataType } from '../types';
import { MlTechniqueId } from './syntheticDataset';

export interface ExperimentRunSnapshot {
  runId: string;
  createdAt: string;
  projectId: string;
  projectTitle: string;
  dataType: SyntheticDataType;
  primaryMethod: MlTechniqueId;
  complementaryMethod: MlTechniqueId;
  datasetSha256: string;
  configSha256: string;
  inBrowserFitted: boolean;
  recordCount: number;
  trainedValAccPct: number;
  meanConfidencePct: number;
  meanConsensusPct: number;
  conformalCoverageTargetPct: number;
  meanConformalSetSize: number;
  decisionThreshold: number;
  fpCostInr: number;
  fnCostInr: number;
  estimatedNetValueInr: number;
  batchP95LatencyMs: number;
  maxPsiDrift: number;
  brierScore?: number;
  prAucPct?: number;
  empiricalEcePct?: number;
  alertCount: number;
  reviewCount: number;
  clearCount: number;
}

const STORAGE_KEY = 'scholarml_experiment_runs_v1';
const MAX_STORED_RUNS = 24;

function fallbackHashHex(text: string): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x9e3779b9;
  for (let i = 0; i < text.length; i++) {
    const ch = text.charCodeAt(i);
    h1 ^= ch;
    h1 = Math.imul(h1, 0x01000193);
    h2 ^= ch + i;
    h2 = Math.imul(h2, 0x85ebca6b);
  }
  const hex1 = (h1 >>> 0).toString(16).padStart(8, '0');
  const hex2 = (h2 >>> 0).toString(16).padStart(8, '0');
  return `${hex1}${hex2}`;
}

export async function computeSha256Short(input: string): Promise<string> {
  try {
    if (typeof window !== 'undefined' && window.crypto?.subtle) {
      const encoded = new TextEncoder().encode(input);
      const digest = await window.crypto.subtle.digest('SHA-256', encoded);
      const bytes = Array.from(new Uint8Array(digest));
      return bytes
        .slice(0, 8)
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
    }
  } catch {
    // Fallback below
  }
  return fallbackHashHex(input);
}

export function loadExperimentRuns(): ExperimentRunSnapshot[] {
  try {
    if (typeof window === 'undefined' || !window.localStorage) return [];
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as ExperimentRunSnapshot[]) : [];
  } catch {
    return [];
  }
}

export function saveExperimentRun(run: ExperimentRunSnapshot): ExperimentRunSnapshot[] {
  const existing = loadExperimentRuns().filter((r) => r.runId !== run.runId);
  const updated = [run, ...existing].slice(0, MAX_STORED_RUNS);
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    }
  } catch {
    // Ignore quota errors cleanly
  }
  return updated;
}

export function deleteExperimentRun(runId: string): ExperimentRunSnapshot[] {
  const updated = loadExperimentRuns().filter((r) => r.runId !== runId);
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    }
  } catch {
    // Ignore
  }
  return updated;
}

export function clearAllExperimentRuns(): ExperimentRunSnapshot[] {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // Ignore
  }
  return [];
}

export interface ExperimentDiffMetric {
  label: string;
  runAVal: string;
  runBVal: string;
  deltaDisplay: string;
  directionBetter: 'higher' | 'lower' | 'neutral';
  winner: 'A' | 'B' | 'Tie';
}

export function diffExperimentRuns(
  runA: ExperimentRunSnapshot,
  runB: ExperimentRunSnapshot
): ExperimentDiffMetric[] {
  const mkNumDiff = (
    label: string,
    a: number | undefined,
    b: number | undefined,
    unit: string,
    directionBetter: 'higher' | 'lower' | 'neutral',
    decimals: number = 1
  ): ExperimentDiffMetric => {
    if (a === undefined || b === undefined) {
      return {
        label,
        runAVal: a !== undefined ? `${a.toFixed(decimals)}${unit}` : '—',
        runBVal: b !== undefined ? `${b.toFixed(decimals)}${unit}` : '—',
        deltaDisplay: 'N/A',
        directionBetter,
        winner: 'Tie'
      };
    }
    const delta = b - a;
    const absDelta = Math.abs(delta);
    const sign = delta > 0 ? '+' : '';
    let winner: 'A' | 'B' | 'Tie' = 'Tie';
    if (absDelta > 1e-4 && directionBetter !== 'neutral') {
      if (directionBetter === 'higher') winner = b > a ? 'B' : 'A';
      else winner = b < a ? 'B' : 'A';
    }
    return {
      label,
      runAVal: `${a.toFixed(decimals)}${unit}`,
      runBVal: `${b.toFixed(decimals)}${unit}`,
      deltaDisplay: `${sign}${delta.toFixed(decimals)}${unit}`,
      directionBetter,
      winner
    };
  };

  return [
    mkNumDiff('Validation Accuracy', runA.trainedValAccPct, runB.trainedValAccPct, '%', 'higher', 2),
    mkNumDiff('Calibrated Confidence', runA.meanConfidencePct, runB.meanConfidencePct, '%', 'higher', 1),
    mkNumDiff('5-Method Consensus', runA.meanConsensusPct, runB.meanConsensusPct, '%', 'higher', 0),
    mkNumDiff('Conformal Avg Set Size', runA.meanConformalSetSize, runB.meanConformalSetSize, '', 'lower', 2),
    mkNumDiff('PR-AUC (Observed Target)', runA.prAucPct, runB.prAucPct, '%', 'higher', 1),
    mkNumDiff('Brier Calibration Loss', runA.brierScore, runB.brierScore, '', 'lower', 4),
    mkNumDiff('Max PSI Feature Drift', runA.maxPsiDrift, runB.maxPsiDrift, '', 'lower', 3),
    mkNumDiff('Dual-Model P95 Latency', runA.batchP95LatencyMs, runB.batchP95LatencyMs, 'ms', 'lower', 2),
    mkNumDiff('Net Value Impact ($ USD)', runA.estimatedNetValueInr, runB.estimatedNetValueInr, ' USD', 'higher', 0)
  ];
}
