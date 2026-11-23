import { ParsedRealRecord, RealDatasetParseResult } from './realDataInference';

export interface FeatureProfileReport {
  feature: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  missingCount: number;
  psiScore: number;
  psiStatus: 'Stable (<0.10)' | 'Moderate Shift (0.10–0.25)' | 'Drift Alert (>0.25)';
  ksStatistic: number;
  ksPValue: number;
  ksDriftFlag: boolean;
  vif: number;
  vifStatus: 'Low (<5)' | 'Moderate (5–10)' | 'Multicollinear (>10)';
  mutualInfoBits: number;
}

export interface ImbalanceDiagnostics {
  hasGroundTruthTarget: boolean;
  targetColumnName: string | null;
  positiveCount: number;
  negativeCount: number;
  positiveRatePct: number;
  imbalanceRatio: string;
  recommendedPosWeight: number;
  recommendedFocalAlpha: number;
  severity: 'Balanced' | 'Moderate Skew' | 'Severe Imbalance';
}

export interface DatasetStatisticalProfile {
  totalRows: number;
  totalColumns: number;
  completenessPct: number;
  overallPsiMax: number;
  driftedFeaturesCount: number;
  multicollinearCount: number;
  imbalance: ImbalanceDiagnostics;
  features: FeatureProfileReport[];
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/**
 * Asymptotic Kolmogorov-Smirnov two-sample p-value via alternating series:
 * P(D_{n,m} >= d) ≈ 2 * sum_{k=1..inf} (-1)^{k-1} * exp(-2 * k^2 * lambda^2)
 */
function computeKsTwoSample(sampleA: number[], sampleB: number[]): { dStat: number; pValue: number } {
  const n = sampleA.length;
  const m = sampleB.length;
  if (n < 2 || m < 2) {
    return { dStat: 0, pValue: 1 };
  }

  const a = [...sampleA].sort((x, y) => x - y);
  const b = [...sampleB].sort((x, y) => x - y);

  let i = 0;
  let j = 0;
  let dMax = 0;

  while (i < n && j < m) {
    const val = Math.min(a[i], b[j]);
    while (i < n && a[i] <= val) i++;
    while (j < m && b[j] <= val) j++;
    const diff = Math.abs(i / n - j / m);
    if (diff > dMax) dMax = diff;
  }

  const ne = (n * m) / (n + m);
  const lambda = (Math.sqrt(ne) + 0.12 + 0.11 / Math.sqrt(ne)) * dMax;
  if (lambda <= 0.01) return { dStat: Number(dMax.toFixed(4)), pValue: 1 };

  let sum = 0;
  for (let k = 1; k <= 30; k++) {
    const term = Math.exp(-2 * k * k * lambda * lambda);
    sum += (k % 2 === 1 ? 1 : -1) * term;
    if (term < 1e-9) break;
  }
  const pVal = clamp(2 * sum, 0.0001, 1);
  return {
    dStat: Number(dMax.toFixed(4)),
    pValue: Number(pVal.toFixed(4))
  };
}

/**
 * Population Stability Index (PSI) across quantile buckets between Reference (A) & Target (B) splits
 */
function computePsi(refValues: number[], curValues: number[], bins: number = 5): number {
  if (refValues.length < 3 || curValues.length < 2) return 0;
  const sorted = [...refValues].sort((a, b) => a - b);
  const edges: number[] = [-Infinity];
  for (let b = 1; b < bins; b++) {
    const idx = Math.floor((b / bins) * sorted.length);
    edges.push(sorted[clamp(idx, 0, sorted.length - 1)]);
  }
  edges.push(Infinity);

  let psi = 0;
  const eps = 0.005;
  for (let b = 0; b < bins; b++) {
    const lo = edges[b];
    const hi = edges[b + 1];
    const refCnt = refValues.filter((v) => (b === bins - 1 ? v >= lo && v <= hi : v >= lo && v < hi)).length;
    const curCnt = curValues.filter((v) => (b === bins - 1 ? v >= lo && v <= hi : v >= lo && v < hi)).length;
    const refPct = Math.max(eps, refCnt / refValues.length);
    const curPct = Math.max(eps, curCnt / curValues.length);
    psi += (curPct - refPct) * Math.log(curPct / refPct);
  }
  return Number(Math.max(0, psi).toFixed(4));
}

/**
 * Invert correlation matrix via Gauss-Jordan elimination to compute exact VIF = diag(R^-1)
 */
function computeVifMatrix(rows: ParsedRealRecord[], featureCols: string[]): Record<string, number> {
  const out: Record<string, number> = {};
  const p = featureCols.length;
  featureCols.forEach((c) => {
    out[c] = 1.0;
  });
  if (p < 2 || rows.length < p + 2) return out;

  const colsLimited = featureCols.slice(0, 12);
  const k = colsLimited.length;

  // Build standardized matrix columns
  const colMeans = colsLimited.map((c) => {
    const vals = rows.map((r) => r.numericFeatures[c]).filter((v) => Number.isFinite(v));
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  });
  const colStds = colsLimited.map((c, j) => {
    const vals = rows.map((r) => r.numericFeatures[c]).filter((v) => Number.isFinite(v));
    if (vals.length < 2) return 1;
    const v = vals.reduce((acc, x) => acc + Math.pow(x - colMeans[j], 2), 0) / vals.length;
    return Math.max(1e-6, Math.sqrt(v));
  });

  // Correlation matrix R with tiny ridge regularization (1e-4) for numerical stability
  const aug: number[][] = Array.from({ length: k }, (_, i) => {
    const row = new Array<number>(2 * k).fill(0);
    row[k + i] = 1;
    return row;
  });

  for (let i = 0; i < k; i++) {
    for (let j = i; j < k; j++) {
      if (i === j) {
        aug[i][j] = 1 + 1e-4;
      } else {
        let sum = 0;
        let cnt = 0;
        for (const r of rows) {
          const xi = r.numericFeatures[colsLimited[i]];
          const xj = r.numericFeatures[colsLimited[j]];
          if (Number.isFinite(xi) && Number.isFinite(xj)) {
            sum += ((xi - colMeans[i]) / colStds[i]) * ((xj - colMeans[j]) / colStds[j]);
            cnt++;
          }
        }
        const corr = cnt > 1 ? clamp(sum / cnt, -0.999, 0.999) : 0;
        aug[i][j] = corr;
        aug[j][i] = corr;
      }
    }
  }

  // Gauss-Jordan
  for (let i = 0; i < k; i++) {
    let pivot = i;
    for (let r = i + 1; r < k; r++) {
      if (Math.abs(aug[r][i]) > Math.abs(aug[pivot][i])) pivot = r;
    }
    if (pivot !== i) {
      const tmp = aug[i];
      aug[i] = aug[pivot];
      aug[pivot] = tmp;
    }
    const diag = aug[i][i];
    if (Math.abs(diag) < 1e-8) continue;
    for (let j = 0; j < 2 * k; j++) aug[i][j] /= diag;
    for (let r = 0; r < k; r++) {
      if (r === i) continue;
      const factor = aug[r][i];
      for (let j = 0; j < 2 * k; j++) {
        aug[r][j] -= factor * aug[i][j];
      }
    }
  }

  colsLimited.forEach((c, i) => {
    const vifVal = aug[i][k + i];
    out[c] = Number(clamp(Number.isFinite(vifVal) && vifVal >= 1 ? vifVal : 1, 1, 99.9).toFixed(2));
  });

  return out;
}

/**
 * Discrete Mutual Information I(X_binned ; Y_binary) in bits
 */
function computeMutualInfoBits(values: number[], targets: number[], bins: number = 4): number {
  const n = Math.min(values.length, targets.length);
  if (n < 4) return 0;

  const minV = Math.min(...values);
  const maxV = Math.max(...values);
  const span = Math.max(1e-6, maxV - minV);

  const joint = Array.from({ length: bins }, () => [0, 0]);
  const xMarg = new Array<number>(bins).fill(0);
  const yMarg = [0, 0];

  for (let i = 0; i < n; i++) {
    const b = clamp(Math.floor(((values[i] - minV) / span) * (bins - 0.001)), 0, bins - 1);
    const y = targets[i] >= 0.5 ? 1 : 0;
    joint[b][y] += 1;
    xMarg[b] += 1;
    yMarg[y] += 1;
  }

  let mi = 0;
  for (let b = 0; b < bins; b++) {
    for (let y = 0; y <= 1; y++) {
      const pxy = joint[b][y] / n;
      const px = xMarg[b] / n;
      const py = yMarg[y] / n;
      if (pxy > 0 && px > 0 && py > 0) {
        mi += pxy * Math.log2(pxy / (px * py));
      }
    }
  }
  return Number(Math.max(0, mi).toFixed(4));
}

export function detectTargetColumn(parseResult: RealDatasetParseResult): {
  targetCol: string | null;
  binaryTargets: number[] | null;
} {
  const candidate = parseResult.numericColumns.find((c) =>
    /^(target|target_y|target_class|target_label|label|class|y|is_fraud|fraud|fraud_ring_flag|settlement_allow|default|malicious|anomaly)$/i.test(
      c.trim()
    )
  );
  if (!candidate) return { targetCol: null, binaryTargets: null };

  const vals = parseResult.records.map((r) => r.numericFeatures[candidate]);
  if (vals.some((v) => !Number.isFinite(v))) {
    return { targetCol: null, binaryTargets: null };
  }
  const unique = new Set(vals);
  if (unique.size <= 2) {
    const sortedUniq = Array.from(unique).sort((a, b) => a - b);
    const posVal = sortedUniq[sortedUniq.length - 1];
    return {
      targetCol: candidate,
      binaryTargets: vals.map((v) => (v === posVal && sortedUniq.length > 1 ? 1 : v > 0 ? 1 : 0))
    };
  }
  // Median binarization if continuous target column
  const sorted = [...vals].sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  return {
    targetCol: candidate,
    binaryTargets: vals.map((v) => (v > med ? 1 : 0))
  };
}

export function profileParsedDataset(parseResult: RealDatasetParseResult): DatasetStatisticalProfile {
  const { records, numericColumns, columns, missingCells } = parseResult;
  const totalRows = records.length;
  const totalCells = Math.max(1, totalRows * Math.max(1, columns.length));
  const completenessPct = Number((((totalCells - missingCells) / totalCells) * 100).toFixed(1));

  const { targetCol, binaryTargets } = detectTargetColumn(parseResult);
  const featureCols = numericColumns.filter((c) => c !== targetCol);

  // Use first 50% vs second 50% split to diagnose dataset stationarity / covariate shift
  const splitIdx = Math.max(1, Math.floor(totalRows / 2));
  const refRows = records.slice(0, splitIdx);
  const curRows = totalRows > 1 ? records.slice(splitIdx) : records;

  const vifMap = computeVifMatrix(records, featureCols);

  // Proxy binary vector if dataset has no explicit ground-truth label column (for Mutual Info ranking)
  const proxyTargets =
    binaryTargets ||
    records.map((r) => {
      let score = 0;
      featureCols.forEach((c) => {
        const stats = parseResult.columnStats[c];
        const v = r.numericFeatures[c];
        if (stats && Number.isFinite(v)) {
          score += Math.abs((v - stats.mean) / Math.max(1e-4, stats.std));
        }
      });
      return score / Math.max(1, featureCols.length) > 1.05 ? 1 : 0;
    });

  const features: FeatureProfileReport[] = featureCols.map((col) => {
    const allVals: number[] = [];
    const alignedTargets: number[] = [];
    let missingCount = 0;

    records.forEach((r, idx) => {
      const v = r.numericFeatures[col];
      if (Number.isFinite(v)) {
        allVals.push(v);
        alignedTargets.push(proxyTargets[idx]);
      } else {
        missingCount++;
      }
    });

    const refVals = refRows.map((r) => r.numericFeatures[col]).filter((v) => Number.isFinite(v));
    const curVals = curRows.map((r) => r.numericFeatures[col]).filter((v) => Number.isFinite(v));

    const stats = parseResult.columnStats[col] || { mean: 0, std: 1, min: 0, max: 0 };
    const psiScore = computePsi(refVals, curVals, Math.min(5, Math.max(2, Math.floor(refVals.length / 2))));
    const psiStatus: FeatureProfileReport['psiStatus'] =
      psiScore > 0.25
        ? 'Drift Alert (>0.25)'
        : psiScore >= 0.1
          ? 'Moderate Shift (0.10–0.25)'
          : 'Stable (<0.10)';

    const ks = computeKsTwoSample(refVals, curVals);
    const vif = vifMap[col] ?? 1;
    const vifStatus: FeatureProfileReport['vifStatus'] =
      vif > 10 ? 'Multicollinear (>10)' : vif >= 5 ? 'Moderate (5–10)' : 'Low (<5)';
    const mutualInfoBits = computeMutualInfoBits(allVals, alignedTargets);

    return {
      feature: col,
      mean: Number(stats.mean.toFixed(3)),
      std: Number(stats.std.toFixed(3)),
      min: Number(stats.min.toFixed(3)),
      max: Number(stats.max.toFixed(3)),
      missingCount,
      psiScore,
      psiStatus,
      ksStatistic: ks.dStat,
      ksPValue: ks.pValue,
      ksDriftFlag: ks.pValue < 0.05 && ks.dStat > 0.25,
      vif,
      vifStatus,
      mutualInfoBits
    };
  });

  const posCount = proxyTargets.filter((y) => y === 1).length;
  const negCount = Math.max(0, totalRows - posCount);
  const posRate = totalRows > 0 ? posCount / totalRows : 0.5;
  const minSide = Math.max(1, Math.min(posCount, negCount));
  const maxSide = Math.max(posCount, negCount);
  const ratioVal = maxSide / minSide;
  const recommendedPosWeight = posCount > 0 ? Number((negCount / posCount).toFixed(2)) : 1;
  const recommendedFocalAlpha = Number(clamp(1 - posRate, 0.25, 0.85).toFixed(2));
  const severity: ImbalanceDiagnostics['severity'] =
    ratioVal >= 5 ? 'Severe Imbalance' : ratioVal >= 2.2 ? 'Moderate Skew' : 'Balanced';

  return {
    totalRows,
    totalColumns: columns.length,
    completenessPct,
    overallPsiMax: features.length > 0 ? Math.max(...features.map((f) => f.psiScore)) : 0,
    driftedFeaturesCount: features.filter((f) => f.psiScore >= 0.1 || f.ksDriftFlag).length,
    multicollinearCount: features.filter((f) => f.vif > 10).length,
    imbalance: {
      hasGroundTruthTarget: Boolean(targetCol),
      targetColumnName: targetCol,
      positiveCount: posCount,
      negativeCount: negCount,
      positiveRatePct: Number((posRate * 100).toFixed(1)),
      imbalanceRatio: `${ratioVal.toFixed(1)}:1`,
      recommendedPosWeight,
      recommendedFocalAlpha,
      severity
    },
    features
  };
}
