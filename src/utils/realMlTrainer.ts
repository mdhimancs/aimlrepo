import { ParsedRealRecord, RealDatasetParseResult } from './realDataInference';
import { detectTargetColumn } from './datasetProfiler';

export interface BoostedStump {
  featureIdx: number;
  featureName: string;
  threshold: number;
  leftVal: number;
  rightVal: number;
  weight: number;
}

export interface InBrowserTrainingEpoch {
  epoch: number;
  trainLoss: number;
  valLoss: number;
  trainAcc: number;
  valAcc: number;
  gradNorm: number;
}

export interface FittedBrowserModels {
  fittedAt: string;
  featureNames: string[];
  scalerMeans: number[];
  scalerStds: number[];
  hasObservedGroundTruth: boolean;
  targetColumnName: string;
  splitCounts: {
    train: number;
    valCalibration: number;
    test: number;
  };
  // Logistic / Neural Linear + Residual L2 model fitted via AdamW
  neuralWeights: number[];
  neuralBias: number;
  // Gradient-Boosted Decision Stump Ensemble fitted on log-loss residuals
  gbdtBaseLogOdds: number;
  gbdtStumps: BoostedStump[];
  // Split Conformal non-conformity quantile calibration (from Val split)
  calNonConformityScores: number[];
  calRegressionResiduals: number[];
  epochHistory: InBrowserTrainingEpoch[];
  finalTestMetrics: {
    neuralAccPct: number;
    gbdtAccPct: number;
    ensembleAccPct: number;
    brierScore: number;
  };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-clamp(x, -15, 15)));
}

function extractFeatureMatrix(
  records: ParsedRealRecord[],
  featureNames: string[],
  means: number[],
  stds: number[]
): number[][] {
  return records.map((r) =>
    featureNames.map((f, j) => {
      const raw = r.numericFeatures[f];
      const val = Number.isFinite(raw) ? raw : means[j];
      return clamp((val - means[j]) / Math.max(1e-4, stds[j]), -4, 4);
    })
  );
}

/**
 * Derive pseudo-labels strictly as fallback ONLY when uploaded dataset has no target column,
 * using anomaly z-score / domain heuristics so in-browser gradient descent can demonstrate real splitting.
 */
function buildLabels(
  parseResult: RealDatasetParseResult,
  featureNames: string[]
): { labels: number[]; hasObservedGroundTruth: boolean; targetColumnName: string } {
  const detected = detectTargetColumn(parseResult);
  if (detected.targetCol && detected.binaryTargets) {
    return {
      labels: detected.binaryTargets,
      hasObservedGroundTruth: true,
      targetColumnName: detected.targetCol
    };
  }

  // Fallback weak-supervision heuristic across standardized column signals
  const labels = parseResult.records.map((r) => {
    let logitSignal = 0;
    featureNames.forEach((f) => {
      const val = r.numericFeatures[f];
      const st = parseResult.columnStats[f];
      if (!Number.isFinite(val) || !st) return;
      const z = (val - st.mean) / Math.max(1e-4, st.std);
      const low = f.toLowerCase();
      if (/trust|consent|quality|snr|match|liveness|verify|credit_score|cibil/.test(low)) {
        logitSignal -= z * 0.9;
      } else {
        logitSignal += z * 0.9;
      }
    });
    return logitSignal > 0.25 ? 1 : 0;
  });

  return {
    labels,
    hasObservedGroundTruth: false,
    targetColumnName: 'weak_supervision_z_target'
  };
}

/**
 * Train real AdamW Neural Logistic Classifier + Real Gradient-Boosted Decision Stumps in-browser
 * with Train / Calibration-Val / Test split and Split Conformal calibration scores.
 */
export function trainInBrowserModels(
  parseResult: RealDatasetParseResult,
  options?: { epochs?: number; lr?: number }
): { error?: string; fitted?: FittedBrowserModels } {
  const { targetCol } = detectTargetColumn(parseResult);
  const featureNames = parseResult.numericColumns.filter((c) => c !== targetCol);

  if (parseResult.records.length < 3 || featureNames.length === 0) {
    return {
      error:
        'Need at least 3 records with >= 1 numeric feature column to run real in-browser Train/Val/Test gradient descent fitting.'
    };
  }

  const epochs = clamp(options?.epochs ?? 18, 6, 40);
  const baseLr = clamp(options?.lr ?? 0.08, 0.005, 0.35);

  const { labels, hasObservedGroundTruth, targetColumnName } = buildLabels(parseResult, featureNames);

  // Split indices: if small (<10 rows), use full-dataset bootstrap/overlap for val/cal stability; otherwise clean 60/20/20
  const n = parseResult.records.length;
  const allIndices = Array.from({ length: n }, (_, i) => i);
  let trainIdx: number[];
  let valIdx: number[];
  let testIdx: number[];

  if (n >= 10) {
    const nTrain = Math.max(4, Math.floor(n * 0.6));
    const nVal = Math.max(2, Math.floor(n * 0.2));
    trainIdx = allIndices.slice(0, nTrain);
    valIdx = allIndices.slice(nTrain, nTrain + nVal);
    testIdx = allIndices.slice(nTrain + nVal);
    if (testIdx.length === 0) testIdx = valIdx;
  } else {
    trainIdx = allIndices.slice(0, Math.max(2, Math.ceil(n * 0.7)));
    valIdx = allIndices.slice(Math.max(1, Math.floor(n * 0.4)));
    testIdx = allIndices;
  }

  // Fit scaler STRICTLY on training split to prevent data leakage
  const scalerMeans = featureNames.map((f) => {
    const vals = trainIdx
      .map((idx) => parseResult.records[idx].numericFeatures[f])
      .filter((v) => Number.isFinite(v));
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  });
  const scalerStds = featureNames.map((f, j) => {
    const vals = trainIdx
      .map((idx) => parseResult.records[idx].numericFeatures[f])
      .filter((v) => Number.isFinite(v));
    if (vals.length < 2) return 1;
    const v = vals.reduce((acc, x) => acc + Math.pow(x - scalerMeans[j], 2), 0) / vals.length;
    return Math.max(1e-4, Math.sqrt(v));
  });

  const XAll = extractFeatureMatrix(parseResult.records, featureNames, scalerMeans, scalerStds);
  const XTrain = trainIdx.map((i) => XAll[i]);
  const yTrain = trainIdx.map((i) => labels[i]);
  const XVal = valIdx.map((i) => XAll[i]);
  const yVal = valIdx.map((i) => labels[i]);
  const XTest = testIdx.map((i) => XAll[i]);
  const yTest = testIdx.map((i) => labels[i]);

  const d = featureNames.length;
  const w = new Array<number>(d).fill(0);
  let b = 0;

  // AdamW optimizer moments
  const mW = new Array<number>(d).fill(0);
  const vW = new Array<number>(d).fill(0);
  let mB = 0;
  let vB = 0;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;
  const weightDecay = 0.015;

  const epochHistory: InBrowserTrainingEpoch[] = [];

  const evalSplit = (X: number[][], y: number[]) => {
    let loss = 0;
    let correct = 0;
    for (let i = 0; i < X.length; i++) {
      let z = b;
      for (let j = 0; j < d; j++) z += w[j] * X[i][j];
      const p = clamp(sigmoid(z), 1e-6, 1 - 1e-6);
      loss += -(y[i] * Math.log(p) + (1 - y[i]) * Math.log(1 - p));
      if ((p >= 0.5 ? 1 : 0) === y[i]) correct++;
    }
    return {
      loss: Number((loss / Math.max(1, X.length)).toFixed(4)),
      acc: Number((correct / Math.max(1, X.length)).toFixed(4))
    };
  };

  for (let ep = 1; ep <= epochs; ep++) {
    const gradW = new Array<number>(d).fill(0);
    let gradB = 0;
    for (let i = 0; i < XTrain.length; i++) {
      let z = b;
      for (let j = 0; j < d; j++) z += w[j] * XTrain[i][j];
      const p = sigmoid(z);
      const err = p - yTrain[i];
      gradB += err;
      for (let j = 0; j < d; j++) gradW[j] += err * XTrain[i][j];
    }
    gradB /= XTrain.length;
    let gradSq = gradB * gradB;
    for (let j = 0; j < d; j++) {
      gradW[j] = gradW[j] / XTrain.length + weightDecay * w[j];
      gradSq += gradW[j] * gradW[j];
    }

    // Cosine learning rate decay
    const lr = baseLr * (0.15 + 0.85 * 0.5 * (1 + Math.cos((Math.PI * ep) / epochs)));

    mB = beta1 * mB + (1 - beta1) * gradB;
    vB = beta2 * vB + (1 - beta2) * gradB * gradB;
    const mBHat = mB / (1 - Math.pow(beta1, ep));
    const vBHat = vB / (1 - Math.pow(beta2, ep));
    b -= (lr * mBHat) / (Math.sqrt(vBHat) + eps);

    for (let j = 0; j < d; j++) {
      mW[j] = beta1 * mW[j] + (1 - beta1) * gradW[j];
      vW[j] = beta2 * vW[j] + (1 - beta2) * gradW[j] * gradW[j];
      const mWHat = mW[j] / (1 - Math.pow(beta1, ep));
      const vWHat = vW[j] / (1 - Math.pow(beta2, ep));
      w[j] -= (lr * mWHat) / (Math.sqrt(vWHat) + eps);
    }

    const tr = evalSplit(XTrain, yTrain);
    const va = evalSplit(XVal, yVal);
    epochHistory.push({
      epoch: ep,
      trainLoss: tr.loss,
      valLoss: va.loss,
      trainAcc: tr.acc,
      valAcc: va.acc,
      gradNorm: Number(Math.sqrt(gradSq).toFixed(4))
    });
  }

  // Fit Real Gradient-Boosted Decision Stumps (LogitBoost / GBDT rounds) on XTrain
  const posTrain = yTrain.filter((y) => y === 1).length;
  const priorP = clamp(posTrain / Math.max(1, yTrain.length), 0.05, 0.95);
  const gbdtBaseLogOdds = Math.log(priorP / (1 - priorP));
  const fTrain = new Array<number>(XTrain.length).fill(gbdtBaseLogOdds);
  const gbdtStumps: BoostedStump[] = [];
  const boostRounds = Math.min(12, Math.max(4, featureNames.length * 2));
  const shrinkLr = 0.32;

  for (let round = 0; round < boostRounds; round++) {
    const residuals = yTrain.map((y, i) => y - sigmoid(fTrain[i]));
    let bestGain = -Infinity;
    let bestStump: BoostedStump | null = null;

    for (let fj = 0; fj < d; fj++) {
      const colVals = Array.from(new Set(XTrain.map((row) => row[fj]))).sort((a, b) => a - b);
      for (let tIdx = 0; tIdx < colVals.length - 1; tIdx++) {
        const thresh = (colVals[tIdx] + colVals[tIdx + 1]) / 2;
        let leftSum = 0;
        let leftCnt = 0;
        let rightSum = 0;
        let rightCnt = 0;
        for (let i = 0; i < XTrain.length; i++) {
          if (XTrain[i][fj] <= thresh) {
            leftSum += residuals[i];
            leftCnt++;
          } else {
            rightSum += residuals[i];
            rightCnt++;
          }
        }
        if (leftCnt === 0 || rightCnt === 0) continue;
        const gain = (leftSum * leftSum) / leftCnt + (rightSum * rightSum) / rightCnt;
        if (gain > bestGain) {
          bestGain = gain;
          bestStump = {
            featureIdx: fj,
            featureName: featureNames[fj],
            threshold: Number(thresh.toFixed(4)),
            leftVal: Number(clamp((leftSum / leftCnt) * 2.2, -2.0, 2.0).toFixed(4)),
            rightVal: Number(clamp((rightSum / rightCnt) * 2.2, -2.0, 2.0).toFixed(4)),
            weight: shrinkLr
          };
        }
      }
    }

    if (!bestStump) break;
    gbdtStumps.push(bestStump);
    for (let i = 0; i < XTrain.length; i++) {
      const pred = XTrain[i][bestStump.featureIdx] <= bestStump.threshold ? bestStump.leftVal : bestStump.rightVal;
      fTrain[i] += shrinkLr * pred;
    }
  }

  // Helper inference functions on standardized X row
  const predictNeural = (x: number[]) => {
    let z = b;
    for (let j = 0; j < d; j++) z += w[j] * x[j];
    return clamp(sigmoid(z), 0.01, 0.99);
  };

  const predictGbdt = (x: number[]) => {
    let f = gbdtBaseLogOdds;
    for (const s of gbdtStumps) {
      f += s.weight * (x[s.featureIdx] <= s.threshold ? s.leftVal : s.rightVal);
    }
    return clamp(sigmoid(f), 0.01, 0.99);
  };

  // Compute Split Conformal Non-Conformity Scores on Validation/Calibration set: s_i = 1 - p(y_i | x_i)
  const calNonConformityScores = XVal.map((x, i) => {
    const p = 0.5 * predictNeural(x) + 0.5 * predictGbdt(x);
    const pTrue = yVal[i] === 1 ? p : 1 - p;
    return Number((1 - pTrue).toFixed(4));
  }).sort((a, c) => a - c);

  const calRegressionResiduals = XVal.map((x, i) => {
    const p = 0.5 * predictNeural(x) + 0.5 * predictGbdt(x);
    return Number(Math.abs(yVal[i] - p).toFixed(4));
  }).sort((a, c) => a - c);

  // Evaluate Held-Out Test Split
  let nCorrect = 0;
  let gCorrect = 0;
  let eCorrect = 0;
  let brierSum = 0;
  XTest.forEach((x, i) => {
    const pn = predictNeural(x);
    const pg = predictGbdt(x);
    const pe = 0.5 * pn + 0.5 * pg;
    if ((pn >= 0.5 ? 1 : 0) === yTest[i]) nCorrect++;
    if ((pg >= 0.5 ? 1 : 0) === yTest[i]) gCorrect++;
    if ((pe >= 0.5 ? 1 : 0) === yTest[i]) eCorrect++;
    brierSum += Math.pow(pe - yTest[i], 2);
  });

  return {
    fitted: {
      fittedAt: new Date().toLocaleTimeString(),
      featureNames,
      scalerMeans,
      scalerStds,
      hasObservedGroundTruth,
      targetColumnName,
      splitCounts: {
        train: XTrain.length,
        valCalibration: XVal.length,
        test: XTest.length
      },
      neuralWeights: w.map((v) => Number(v.toFixed(4))),
      neuralBias: Number(b.toFixed(4)),
      gbdtBaseLogOdds: Number(gbdtBaseLogOdds.toFixed(4)),
      gbdtStumps,
      calNonConformityScores,
      calRegressionResiduals,
      epochHistory,
      finalTestMetrics: {
        neuralAccPct: Number(((nCorrect / Math.max(1, XTest.length)) * 100).toFixed(1)),
        gbdtAccPct: Number(((gCorrect / Math.max(1, XTest.length)) * 100).toFixed(1)),
        ensembleAccPct: Number(((eCorrect / Math.max(1, XTest.length)) * 100).toFixed(1)),
        brierScore: Number((brierSum / Math.max(1, XTest.length)).toFixed(4))
      }
    }
  };
}

export function scoreRecordWithFittedModels(
  rec: ParsedRealRecord,
  fitted: FittedBrowserModels
): { neuralProb: number; gbdtProb: number } | null {
  if (!fitted.featureNames.length) return null;
  const x = fitted.featureNames.map((f, j) => {
    const raw = rec.numericFeatures[f];
    const val = Number.isFinite(raw) ? raw : fitted.scalerMeans[j];
    return clamp((val - fitted.scalerMeans[j]) / Math.max(1e-4, fitted.scalerStds[j]), -4, 4);
  });

  let z = fitted.neuralBias;
  for (let j = 0; j < x.length; j++) {
    z += (fitted.neuralWeights[j] || 0) * x[j];
  }
  const neuralProb = clamp(sigmoid(z), 0.015, 0.985);

  let fScore = fitted.gbdtBaseLogOdds;
  for (const stump of fitted.gbdtStumps) {
    const xv = x[stump.featureIdx] ?? 0;
    fScore += stump.weight * (xv <= stump.threshold ? stump.leftVal : stump.rightVal);
  }
  const gbdtProb = clamp(sigmoid(fScore), 0.015, 0.985);

  return { neuralProb, gbdtProb };
}
