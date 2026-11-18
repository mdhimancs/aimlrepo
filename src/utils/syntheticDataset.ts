import {
  FlagshipProject,
  SyntheticColumnSchema,
  SyntheticDataType,
  SyntheticDatasetProfile,
  SyntheticVariant
} from '../types';

export const SYNTHETIC_DATA_TYPE_META: Record<
  SyntheticDataType,
  {
    label: string;
    shortLabel: string;
    tensorShape: string;
    lossObjective: string;
    throughputBase: number;
  }
> = {
  tabular: {
    label: 'Tabular Structured (Float32 / Categorical)',
    shortLabel: 'Tabular Structured',
    tensorShape: '[B, 24] float32 + int64',
    lossObjective: 'Stratified Log-Loss / Huber RMSE',
    throughputBase: 6400
  },
  vision: {
    label: 'Computer Vision Tensor (2D / 3D Image Patch)',
    shortLabel: 'Vision Tensor',
    tensorShape: '[B, 3, 224, 224] float16',
    lossObjective: 'Label-Smoothed Cross-Entropy / IoU',
    throughputBase: 1450
  },
  nlp: {
    label: 'NLP Tokenized Sequence (Subword + Mask)',
    shortLabel: 'NLP Text Sequence',
    tensorShape: '[B, 512] int64 (input_ids + mask)',
    lossObjective: 'Causal / Masked Cross-Entropy',
    throughputBase: 2850
  },
  timeseries: {
    label: 'Multivariate Time-Series Sliding Window',
    shortLabel: 'Time-Series Signal',
    tensorShape: '[B, 96, 12] float32',
    lossObjective: 'Quantile Pinball + MAPE Loss',
    throughputBase: 4900
  },
  graph: {
    label: 'Graph / Knowledge Graph Sparse Adjacency',
    shortLabel: 'Graph / PDE Topology',
    tensorShape: 'X:[N, 128], edge_index:[2, E]',
    lossObjective: 'InfoNCE + Dirichlet Energy Reg',
    throughputBase: 1920
  },
  multimodal: {
    label: 'Multimodal Fusion (Vision + Text + Latent)',
    shortLabel: 'Multimodal Fusion',
    tensorShape: 'txt:[B,128], img:[B,196,768]',
    lossObjective: 'Hybrid ELBO + Contrastive Alignment',
    throughputBase: 1180
  },
  audio: {
    label: 'Audio Acoustic Log-Mel Spectrogram',
    shortLabel: 'Audio Spectrogram',
    tensorShape: '[B, 1, 128, 256] float32',
    lossObjective: 'Spectral Convergence + Focal CE',
    throughputBase: 2240
  }
};

export const SYNTHETIC_VARIANT_META: Record<
  SyntheticVariant,
  {
    label: string;
    desc: string;
    noiseMultiplier: number;
    lossPenalty: number;
    accOffset: number;
    sigmaBase: number;
  }
> = {
  balanced: {
    label: 'Domain-Native Balanced',
    desc: 'IID synthetic sampling matching clean class priors and nominal SNR.',
    noiseMultiplier: 1.0,
    lossPenalty: 1.0,
    accOffset: 0,
    sigmaBase: 0.42
  },
  imbalanced: {
    label: 'Class-Imbalanced Stress (5% Minority)',
    desc: 'Synthetic extreme skew simulating rare-event detection & hard negatives.',
    noiseMultiplier: 1.35,
    lossPenalty: 1.24,
    accOffset: -0.025,
    sigmaBase: 0.88
  },
  ood_noisy: {
    label: 'OOD Shift & Gaussian Perturbation (+σ)',
    desc: 'Covariate drift + corrupted features testing calibration & stability.',
    noiseMultiplier: 1.85,
    lossPenalty: 1.38,
    accOffset: -0.042,
    sigmaBase: 1.25
  },
  scaled_100k: {
    label: 'Scaled Benchmark Batch (100k Samples)',
    desc: 'High-volume synthetic stress benchmark testing convergence & throughput.',
    noiseMultiplier: 0.9,
    lossPenalty: 0.94,
    accOffset: 0.012,
    sigmaBase: 0.34
  }
};

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function inferProjectDataType(project: FlagshipProject): SyntheticDataType {
  const haystack = `${project.id} ${project.title} ${project.domain} ${project.techStack.join(' ')}`.toLowerCase();

  if (
    haystack.includes('multimodal') ||
    haystack.includes('diffusioncom') ||
    haystack.includes('ddpm') ||
    haystack.includes('medical imaging') ||
    haystack.includes('swin transformer')
  ) {
    return 'multimodal';
  }
  if (
    haystack.includes('graph') ||
    haystack.includes('protein') ||
    haystack.includes('se(3)') ||
    haystack.includes('opinion dynamics') ||
    haystack.includes('pde')
  ) {
    return 'graph';
  }
  if (
    haystack.includes('audio') ||
    haystack.includes('speech') ||
    haystack.includes('urbansound') ||
    haystack.includes('mfcc')
  ) {
    return 'audio';
  }
  if (
    haystack.includes('vision') ||
    haystack.includes('yolo') ||
    haystack.includes('mnist') ||
    haystack.includes('digit') ||
    haystack.includes('xray') ||
    haystack.includes('vit') ||
    haystack.includes('art')
  ) {
    return 'vision';
  }
  if (
    haystack.includes('time-series') ||
    haystack.includes('forecast') ||
    haystack.includes('stock') ||
    haystack.includes('sales') ||
    haystack.includes('energy') ||
    haystack.includes('bitcoin') ||
    haystack.includes('quant')
  ) {
    return 'timeseries';
  }
  if (
    haystack.includes('nlp') ||
    haystack.includes('rag') ||
    haystack.includes('llm') ||
    haystack.includes('kv-cache') ||
    haystack.includes('prompt-injection') ||
    haystack.includes('jailbreak') ||
    haystack.includes('fuzzyai') ||
    haystack.includes('sentiment') ||
    haystack.includes('spam') ||
    haystack.includes('fake news') ||
    haystack.includes('chatbot') ||
    haystack.includes('hr analytics')
  ) {
    return 'nlp';
  }
  return 'tabular';
}

function getColumnsForType(
  project: FlagshipProject,
  dataType: SyntheticDataType
): SyntheticColumnSchema[] {
  const domainLower = project.domain.toLowerCase();

  switch (dataType) {
    case 'multimodal':
      if (domainLower.includes('uidai') || domainLower.includes('biometric')) {
        return [
          { name: 'rd_probe_id', dtype: 'string', spec: 'SYN_VID_RD_XXXX', role: 'metadata' },
          { name: 'iris_gabor_2048b', dtype: 'uint64[32]', spec: '2048-bit masked phase code', role: 'tensor' },
          { name: 'fp_face_fused_emb', dtype: 'float16[256]', spec: 'Quality-gated L2 embedding', role: 'tensor' },
          { name: 'pad_liveness_score', dtype: 'float32', spec: '[0.00, 1.00] ISO-30107 score', role: 'feature' },
          { name: 'auth_verified', dtype: 'int64', spec: '0 (Reject/Spoof) / 1 (Live Match)', role: 'target' }
        ];
      }
      return [
        { name: 'sample_id', dtype: 'string', spec: 'UUID / entity_pair', role: 'metadata' },
        { name: 'text_tokens', dtype: 'int64[128]', spec: 'Tokenizer IDs (vocab 50k)', role: 'tensor' },
        { name: 'visual_patch_emb', dtype: 'float16[196,768]', spec: 'ViT-B/16 L2-norm patches', role: 'tensor' },
        { name: 'cross_gate_weight', dtype: 'float32', spec: 'g ∈ [0.12, 0.98]', role: 'feature' },
        { name: 'target_label', dtype: 'int64', spec: 'ground-truth class / tail_id', role: 'target' }
      ];
    case 'graph':
      return [
        { name: 'node_id', dtype: 'int64', spec: '0..4095 subgraph idx', role: 'metadata' },
        { name: 'node_features', dtype: 'float32[128]', spec: 'Standardized node state x_i', role: 'tensor' },
        { name: 'degree_k', dtype: 'int32', spec: '1..48 incident edges', role: 'feature' },
        { name: 'dirichlet_flux', dtype: 'float32', spec: '0.04..1.85 energy norm', role: 'feature' },
        { name: 'target_class', dtype: 'int64', spec: 'node / conformation label', role: 'target' }
      ];
    case 'vision':
      return [
        { name: 'frame_id', dtype: 'string', spec: 'patch_224x224', role: 'metadata' },
        { name: 'rgb_nchw_norm', dtype: 'float16[3,224,224]', spec: 'ImageNet μ/σ normalized', role: 'tensor' },
        { name: 'snr_db', dtype: 'float32', spec: '18.5..34.2 dB signal quality', role: 'feature' },
        { name: 'aug_mask_ratio', dtype: 'float32', spec: '0.00..0.40 cutout', role: 'feature' },
        { name: 'target_label', dtype: 'int64', spec: 'object / pathology class', role: 'target' }
      ];
    case 'nlp':
      return [
        { name: 'doc_id', dtype: 'string', spec: 'seq_512_tok', role: 'metadata' },
        { name: 'input_ids', dtype: 'int64[512]', spec: 'BPE / WordPiece IDs', role: 'tensor' },
        { name: 'token_entropy', dtype: 'float32', spec: '2.1..6.4 bits', role: 'feature' },
        { name: 'kv_sparsity', dtype: 'float32', spec: '0.25..0.75 retained ratio', role: 'feature' },
        { name: 'target_label', dtype: 'int64', spec: 'intent / veracity / rank', role: 'target' }
      ];
    case 'timeseries':
      return [
        { name: 'window_t', dtype: 'datetime', spec: 'Hourly / Daily UTC step', role: 'metadata' },
        { name: 'lag_signal_1d', dtype: 'float32', spec: 'Seasonal deseasonalized z', role: 'feature' },
        { name: 'rolling_vol_7d', dtype: 'float32', spec: '0.02..0.45 std window', role: 'feature' },
        { name: 'exog_promo_idx', dtype: 'float32', spec: '0.00..1.00 normalized driver', role: 'feature' },
        { name: 'target_horizon', dtype: 'float32', spec: 'Next-step forecast y_{t+1}', role: 'target' }
      ];
    case 'audio':
      return [
        { name: 'clip_id', dtype: 'string', spec: 'wav_16kHz_4s', role: 'metadata' },
        { name: 'mel_spec_128', dtype: 'float32[128,256]', spec: 'Log-Mel dB matrix', role: 'tensor' },
        { name: 'spectral_centroid', dtype: 'float32', spec: '420..3,800 Hz', role: 'feature' },
        { name: 'zero_cross_rate', dtype: 'float32', spec: '0.03..0.28 ZCR', role: 'feature' },
        { name: 'target_class', dtype: 'int64', spec: 'acoustic / emotion class', role: 'target' }
      ];
    default: {
      if (domainLower.includes('uidai') || domainLower.includes('payment')) {
        return [
          { name: 'vid_sector_token', dtype: 'string', spec: 'DPI_TOK_XXXX', role: 'metadata' },
          { name: 'upi_aeps_velocity_z', dtype: 'float32', spec: '[-1.8, +4.2] cross-rail z', role: 'feature' },
          { name: 'dbt_dedup_overlap', dtype: 'float32', spec: '[0.00, 1.00] entity score', role: 'feature' },
          { name: 'consent_trust_score', dtype: 'float32', spec: '[0.10, 1.00] DEPA artifact', role: 'feature' },
          { name: 'settlement_allow', dtype: 'int64', spec: '0 (Hold/Review) / 1 (Authorized)', role: 'target' }
        ];
      }
      if (domainLower.includes('malware') || domainLower.includes('cybersecurity')) {
        return [
          { name: 'pe_sha256_id', dtype: 'string', spec: 'SYN_BIN_XXXX', role: 'metadata' },
          { name: 'shannon_entropy_max', dtype: 'float32', spec: '[3.10, 7.98] bits/byte', role: 'feature' },
          { name: 'packed_section_ratio', dtype: 'float32', spec: '[0.00, 1.00] RWE density', role: 'feature' },
          { name: 'import_iat_anomaly', dtype: 'float32', spec: '[0.02, 0.99] API score', role: 'feature' },
          { name: 'target_class', dtype: 'int64', spec: '0 (Signed Benign) / 1 (Malware)', role: 'target' }
        ];
      }
      if (domainLower.includes('credit') || domainLower.includes('fraud') || domainLower.includes('finance')) {
        return [
          { name: 'tx_record_id', dtype: 'string', spec: 'SYN_TX_XXXX', role: 'metadata' },
          { name: 'log_amount_z', dtype: 'float32', spec: '[-2.4, +3.6] standardized', role: 'feature' },
          { name: 'emi_income_ratio', dtype: 'float32', spec: '[0.05, 0.78] debt ratio', role: 'feature' },
          { name: 'velocity_score', dtype: 'float32', spec: '[0.00, 1.00] behavioral', role: 'feature' },
          { name: 'target_class', dtype: 'int64', spec: '0 (Normal) / 1 (Alert/Approved)', role: 'target' }
        ];
      }
      return [
        { name: 'sample_id', dtype: 'string', spec: 'SYN_ROW_XXXX', role: 'metadata' },
        { name: 'feature_x1_norm', dtype: 'float32', spec: 'N(0, 1) standardized', role: 'feature' },
        { name: 'feature_x2_ratio', dtype: 'float32', spec: '[0.00, 1.00] scaled ratio', role: 'feature' },
        { name: 'interaction_term', dtype: 'float32', spec: 'x1 * log1p(|x2|)', role: 'feature' },
        { name: 'target_y', dtype: 'int64', spec: '0 / 1 stratified label', role: 'target' }
      ];
    }
  }
}

export function getSyntheticDatasetProfile(
  project: FlagshipProject,
  overrideDataType?: SyntheticDataType,
  variant: SyntheticVariant = 'balanced'
): SyntheticDatasetProfile {
  const defaultDataType = inferProjectDataType(project);
  const activeType = overrideDataType || defaultDataType;
  const meta = SYNTHETIC_DATA_TYPE_META[activeType];
  const varMeta = SYNTHETIC_VARIANT_META[variant];

  const baseCount =
    variant === 'scaled_100k'
      ? '100,000 synthetic records'
      : project.sampleDistribution[0]?.count
      ? `25,000 synthetic (${project.sampleDistribution[0].count} ref)`
      : '25,000 synthetic samples';

  const shortTitle = project.title.split(':')[0].slice(0, 38).trim();

  return {
    datasetName: `Synthetic ${shortTitle} (${varMeta.label.split(' ')[0]})`,
    defaultDataType,
    dataTypeLabel: meta.label,
    tensorShape: meta.tensorShape,
    sampleCountLabel: baseCount,
    recommendedLoss: meta.lossObjective,
    noiseSigma: `σ = ${(0.12 * varMeta.noiseMultiplier).toFixed(2)}`,
    columns: getColumnsForType(project, activeType)
  };
}

export function generateSyntheticSamples(
  project: FlagshipProject,
  dataType: SyntheticDataType,
  variant: SyntheticVariant = 'balanced',
  count: number = 6,
  seed: number = 42
): Record<string, string | number>[] {
  const hashOffset = project.id
    .split('')
    .reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
  const rand = mulberry32(seed * 97 + hashOffset + dataType.length * 13);
  const varMeta = SYNTHETIC_VARIANT_META[variant];
  const cols = getColumnsForType(project, dataType);
  const rows: Record<string, string | number>[] = [];

  for (let i = 0; i < count; i++) {
    const noise = (rand() - 0.5) * 0.4 * varMeta.noiseMultiplier;
    const isPositive =
      variant === 'imbalanced' ? rand() < 0.12 : rand() < 0.52;

    const row: Record<string, string | number> = {};
    cols.forEach((col, cIdx) => {
      if (col.role === 'metadata') {
        if (dataType === 'timeseries') {
          const day = 10 + i;
          row[col.name] = `2026-11-${String(day).padStart(2, '0')}T${String((i * 3) % 24).padStart(2, '0')}:00Z`;
        } else {
          row[col.name] = `${dataType.slice(0, 3).toUpperCase()}_${1000 + i * 7 + seed}`;
        }
      } else if (col.role === 'tensor') {
        const v1 = (rand() * 1.8 - 0.9 + noise).toFixed(2);
        const v2 = (rand() * 1.8 - 0.9 - noise).toFixed(2);
        const v3 = (rand() * 1.8 - 0.9).toFixed(2);
        if (col.dtype.startsWith('int64')) {
          row[col.name] = `[${Math.floor(rand() * 32000)}, ${Math.floor(rand() * 32000)}, ${Math.floor(rand() * 32000)}, ...]`;
        } else {
          row[col.name] = `[${v1}, ${v2}, ${v3}, ...]`;
        }
      } else if (col.role === 'target') {
        if (dataType === 'timeseries') {
          row[col.name] = Number((108.5 + i * 2.4 + rand() * 14 * varMeta.noiseMultiplier).toFixed(2));
        } else {
          const labels = project.sampleDistribution.map((s) => s.label.split('(')[0].trim());
          const labelIdx = isPositive ? 0 : Math.min(1, labels.length - 1);
          const tag = labels[labelIdx] ? ` (${labels[labelIdx].slice(0, 16)})` : '';
          row[col.name] = `${isPositive ? 1 : 0}${tag}`;
        }
      } else {
        const val =
          cIdx === 1
            ? (isPositive ? 0.65 : -0.45) + rand() * 0.8 + noise
            : rand() * (cIdx === 2 ? 0.95 : 2.5) + Math.abs(noise);
        row[col.name] = Number(val.toFixed(3));
      }
    });
    rows.push(row);
  }
  return rows;
}
