import {
  FlagshipProject,
  SyntheticColumnSchema,
  SyntheticDataType,
  SyntheticDatasetProfile,
  SyntheticVariant
} from '../types';

export type ProgrammingArchId =
  | 'rust_tokio_libtorch'
  | 'cpp20_tensorrt_onnx'
  | 'rust_burn_edge'
  | 'arrow_flight_polars'
  | 'python_fastapi_pytorch';

export interface LearningRatePreset {
  id: string;
  lrValue: string;
  numericLr: number;
  schedule: string;
  label: string;
  bestFor: string;
  recommendedBatch: string;
  convergenceSpeed: string;
  stabilityRisk: string;
  lossModifier: number;
  accBonus: number;
  sigmaMult: number;
}

export const LEARNING_RATE_PRESETS: LearningRatePreset[] = [
  {
    id: 'lr_1e5',
    lrValue: '0.00001',
    numericLr: 1e-5,
    schedule: 'Linear Warmup (10%) + Cosine Decay',
    label: '1e-5 · DP-SGD & Federated Fine-Tuning',
    bestFor: 'Privacy-preserving federated biometrics, ArcFace margin preservation, zero-forgetting 20-yr continual recalibration',
    recommendedBatch: '32–64',
    convergenceSpeed: 'Slow / Ultra-Smooth',
    stabilityRisk: 'Minimal (DP Noise Safe)',
    lossModifier: 1.04,
    accBonus: 0.004,
    sigmaMult: 0.72
  },
  {
    id: 'lr_3e5',
    lrValue: '0.00003',
    numericLr: 3e-5,
    schedule: 'Warmup + Cosine Annealing (η_min=1e-6)',
    label: '3e-5 · Swin Transformer / ViT / LLM Optimal',
    bestFor: 'Multimodal transformers, 3D imaging Swin blocks, BPE/NLP encoders with LayerNorm attention stability',
    recommendedBatch: '16–32',
    convergenceSpeed: 'Balanced High Precision',
    stabilityRisk: 'Very Low',
    lossModifier: 0.94,
    accBonus: 0.009,
    sigmaMult: 0.82
  },
  {
    id: 'lr_1e4',
    lrValue: '0.0001',
    numericLr: 1e-4,
    schedule: 'Cosine Annealing w/ Warm Restarts',
    label: '1e-4 · GraphSAGE & Multimodal Fusion Default',
    bestFor: 'Heterogeneous money-trail graphs, 1.50B stratified multimodal encoders, general AdamW baseline',
    recommendedBatch: '32–64',
    convergenceSpeed: 'Fast',
    stabilityRisk: 'Low',
    lossModifier: 0.96,
    accBonus: 0.007,
    sigmaMult: 0.9
  },
  {
    id: 'lr_3e4',
    lrValue: '0.0003',
    numericLr: 3e-4,
    schedule: 'OneCycleLR / WSD (Warmup-Stable-Decay)',
    label: '3e-4 · Contrastive InfoNCE & Diffusion (Karpathy LR)',
    bestFor: 'Contrastive InfoNCE telemetry, discrete denoising diffusion, rapid exploration on normalized embeddings',
    recommendedBatch: '64–128',
    convergenceSpeed: 'Very Fast',
    stabilityRisk: 'Moderate (Needs Warmup)',
    lossModifier: 0.92,
    accBonus: 0.005,
    sigmaMult: 1.08
  },
  {
    id: 'lr_1e3',
    lrValue: '0.001',
    numericLr: 1e-3,
    schedule: 'Cyclic Triangular + Exponential Decay',
    label: '1e-3 · High-Frequency Streaming & Tabular GBDT',
    bestFor: 'Sub-millisecond online drift adaptation, NPCI/UPI micro-batch linear probes, tabular residual nets',
    recommendedBatch: '64–128',
    convergenceSpeed: 'Aggressive Real-Time',
    stabilityRisk: 'Elevated on Deep Transformers',
    lossModifier: 1.06,
    accBonus: -0.008,
    sigmaMult: 1.28
  },
  {
    id: 'lr_3e2',
    lrValue: '0.03',
    numericLr: 3e-2,
    schedule: 'SGD Nesterov + LARS Polynomial Decay',
    label: '3e-2 · Distributed Large-Batch SGD / LARS',
    bestFor: 'Multi-node 1.50B sharded SGD Nesterov + momentum runs across bare-metal CPU/GPU clusters',
    recommendedBatch: '128+',
    convergenceSpeed: 'Burst Parallel',
    stabilityRisk: 'High (Requires LARS Clip)',
    lossModifier: 1.15,
    accBonus: -0.016,
    sigmaMult: 1.45
  }
];

export type MlTechniqueId =
  | 'multimodal_transformer'
  | 'heterogeneous_graphsage'
  | 'gbdt_catboost_xgb'
  | 'federated_contrastive'
  | 'diffusion_focal_resnet';

export const SYNTHETIC_DATA_TYPE_META: Record<
  SyntheticDataType,
  {
    label: string;
    shortLabel: string;
    tensorShape: string;
    lossObjective: string;
    throughputBase: number;
    baseLatencyMs: number;
    vramGb: number;
    accBias: number;
    lossMult: number;
  }
> = {
  tabular: {
    label: 'Tabular Structured (Float32 / Categorical)',
    shortLabel: 'Tabular Structured',
    tensorShape: '[B, 24] float32 + int64',
    lossObjective: 'Stratified Log-Loss / Huber RMSE',
    throughputBase: 28400,
    baseLatencyMs: 2.2,
    vramGb: 0.48,
    accBias: 0.0,
    lossMult: 1.0
  },
  vision: {
    label: 'Computer Vision Tensor (2D / 3D Image Patch)',
    shortLabel: 'Vision Tensor',
    tensorShape: '[B, 3, 224, 224] float16',
    lossObjective: 'Label-Smoothed Cross-Entropy / IoU',
    throughputBase: 6450,
    baseLatencyMs: 9.8,
    vramGb: 2.15,
    accBias: 0.008,
    lossMult: 0.96
  },
  nlp: {
    label: 'NLP Tokenized Sequence (Subword + Mask)',
    shortLabel: 'NLP Text Sequence',
    tensorShape: '[B, 512] int64 (input_ids + mask)',
    lossObjective: 'Causal / Masked Cross-Entropy',
    throughputBase: 14200,
    baseLatencyMs: 5.4,
    vramGb: 1.64,
    accBias: -0.004,
    lossMult: 1.04
  },
  timeseries: {
    label: 'Multivariate Time-Series Sliding Window',
    shortLabel: 'Time-Series Signal',
    tensorShape: '[B, 96, 12] float32',
    lossObjective: 'Quantile Pinball + MAPE Loss',
    throughputBase: 21900,
    baseLatencyMs: 3.1,
    vramGb: 0.62,
    accBias: -0.006,
    lossMult: 1.08
  },
  graph: {
    label: 'Graph / Knowledge Graph Sparse Adjacency',
    shortLabel: 'Graph / PDE Topology',
    tensorShape: 'X:[N, 128], edge_index:[2, E]',
    lossObjective: 'InfoNCE + Dirichlet Energy Reg',
    throughputBase: 18500,
    baseLatencyMs: 3.6,
    vramGb: 1.42,
    accBias: 0.011,
    lossMult: 0.92
  },
  multimodal: {
    label: 'Multimodal Fusion (Vision + Text + Biometric + NavIC)',
    shortLabel: 'Multimodal Fusion',
    tensorShape: 'bio:[B,256], txt:[B,128], img:[B,196,768]',
    lossObjective: 'Hybrid ELBO + ArcFace Margin Alignment',
    throughputBase: 16200,
    baseLatencyMs: 4.2,
    vramGb: 1.84,
    accBias: 0.016,
    lossMult: 0.88
  },
  audio: {
    label: 'Audio Acoustic Log-Mel Spectrogram',
    shortLabel: 'Audio Spectrogram',
    tensorShape: '[B, 1, 128, 256] float32',
    lossObjective: 'Spectral Convergence + Focal CE',
    throughputBase: 11240,
    baseLatencyMs: 6.7,
    vramGb: 0.95,
    accBias: -0.009,
    lossMult: 1.11
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
    eceCalibration: number;
    skewProfile: string;
    stabilityVerdict: string;
  }
> = {
  balanced: {
    label: 'Domain-Native Balanced (1.50B Cohort)',
    desc: 'IID stratified synthetic sampling across 1.50 billion records matching nominal class priors.',
    noiseMultiplier: 1.0,
    lossPenalty: 1.0,
    accOffset: 0,
    sigmaBase: 0.38,
    eceCalibration: 0.014,
    skewProfile: '50% / 50% Stratified IID',
    stabilityVerdict: 'Nominal Baseline'
  },
  imbalanced: {
    label: 'Class-Imbalanced Fraud Stress (5% Minority · 1.50B)',
    desc: 'Synthetic extreme skew (75M minority alerts in 1.50B) modeling rare crime & spoof rings.',
    noiseMultiplier: 1.35,
    lossPenalty: 1.24,
    accOffset: -0.022,
    sigmaBase: 0.84,
    eceCalibration: 0.028,
    skewProfile: '95% Benign / 5% Hard Negatives',
    stabilityVerdict: 'High Precision Guard'
  },
  ood_noisy: {
    label: 'OOD 20-Yr Aging & Sensor Drift (+σ Perturbation)',
    desc: 'Simulates 20-year ridge wear, lighting drift, and rural micro-ATM telemetry noise.',
    noiseMultiplier: 1.75,
    lossPenalty: 1.36,
    accOffset: -0.036,
    sigmaBase: 1.18,
    eceCalibration: 0.039,
    skewProfile: 'Covariate Drift +1.75σ',
    stabilityVerdict: 'Stress-Calibrated'
  },
  scaled_100k: {
    label: '20-Year Growth Horizon (1.50B → 2.41B @ Δ ±2.4%/yr)',
    desc: 'Full 20-year longitudinal scaling (2026–2046) with annual ±2.4% population delta & NavIC drift.',
    noiseMultiplier: 0.92,
    lossPenalty: 0.94,
    accOffset: 0.014,
    sigmaBase: 0.31,
    eceCalibration: 0.011,
    skewProfile: '1.50B Base -> 2.41B (Yr 20)',
    stabilityVerdict: 'SLA Verified (20-Yr)'
  }
};

export const PROGRAMMING_ARCHITECTURE_META: Record<
  ProgrammingArchId,
  {
    id: ProgrammingArchId;
    name: string;
    shortName: string;
    license: string;
    platformLockIn: string;
    hftMechanism: string;
    concurrencyModel: string;
    memorySafety: string;
    throughputMultiplier: number;
    latencyMsBase: number;
    p99JitterUs: number;
    memoryFootprintGb: number;
    cpuUtilPct: number;
    costIndex: string;
    suitability: string;
  }
> = {
  rust_tokio_libtorch: {
    id: 'rust_tokio_libtorch',
    name: 'Rust (io_uring + LMAX Ring-Buffer) + Tract/ONNX',
    shortName: 'Rust io_uring + Tract',
    license: 'Apache-2.0 / MIT (100% FOSS)',
    platformLockIn: 'Zero Lock-In · x86_64 / ARM64 / Bare-Metal Linux',
    hftMechanism: 'Kernel-Bypass io_uring + Lock-Free SPSC Ring Buffer + Core Pinning',
    concurrencyModel: 'Thread-Per-Core Shared-Nothing (No GC, No GIL)',
    memorySafety: 'Compile-Time Borrow Checker (Zero UAF)',
    throughputMultiplier: 1.72,
    latencyMsBase: 0.78,
    p99JitterUs: 140,
    memoryFootprintGb: 0.44,
    cpuUtilPct: 96,
    costIndex: '0.78x (Bare-Metal Optimal)',
    suitability: 'HFT Biometric Match, UPI Fraud Guard (<1ms SLA)'
  },
  cpp20_tensorrt_onnx: {
    id: 'cpp20_tensorrt_onnx',
    name: 'C++20 (Seastar / DPDK Bypass) + Apache TVM / OpenVINO',
    shortName: 'C++20 Seastar + TVM',
    license: 'Apache-2.0 / BSD (100% FOSS)',
    platformLockIn: 'Zero Lock-In · CPU AVX-512 / AMD ROCm / Vulkan',
    hftMechanism: 'DPDK User-Space NIC Poll-Mode + Seastar Futures + TVM Fused Kernels',
    concurrencyModel: 'Lock-Free Shard-Per-Core Async Futures',
    memorySafety: 'Modern C++20 RAII + Static Arena Allocators',
    throughputMultiplier: 1.64,
    latencyMsBase: 0.96,
    p99JitterUs: 195,
    memoryFootprintGb: 0.56,
    cpuUtilPct: 94,
    costIndex: '0.86x (HFT Cluster)',
    suitability: 'Ultra-Low-Latency Multimodal & Vision Inference'
  },
  rust_burn_edge: {
    id: 'rust_burn_edge',
    name: 'Rust Burn + Candle + WasmEdge Zero-Copy',
    shortName: 'Rust Burn / Candle',
    license: 'MIT / Apache-2.0 (Pure Rust FOSS)',
    platformLockIn: 'Universal · ARM Edge, Linux, WebGPU, WasmEdge',
    hftMechanism: 'Zero-Allocation Arena Tensors + Portable SIMD Vectorization',
    concurrencyModel: 'Rayon Work-Stealing + Pre-Allocated Arena Pool',
    memorySafety: '100% Safe Rust (Zero C/C++ FFI)',
    throughputMultiplier: 1.44,
    latencyMsBase: 1.55,
    p99JitterUs: 290,
    memoryFootprintGb: 0.34,
    cpuUtilPct: 95,
    costIndex: '0.72x (Edge Native)',
    suitability: 'RD Terminals, Micro-ATMs & Offline NavIC Edge'
  },
  arrow_flight_polars: {
    id: 'arrow_flight_polars',
    name: 'Apache Arrow Flight (gRPC) + Polars / DuckDB Mesh',
    shortName: 'Arrow Flight + Polars',
    license: 'Apache-2.0 / MIT (100% FOSS)',
    platformLockIn: 'Cloud & Hardware Free · Zero Proprietary DB',
    hftMechanism: 'Zero-Deserialization Columnar IPC + Vectorized SIMD Scans',
    concurrencyModel: 'Morsel-Driven Parallel Vector Execution',
    memorySafety: 'Rust Core (Polars) + Safe C++ Columnar Buffers',
    throughputMultiplier: 1.38,
    latencyMsBase: 2.15,
    p99JitterUs: 420,
    memoryFootprintGb: 0.72,
    cpuUtilPct: 92,
    costIndex: '0.84x (Zero Warehouse Fee)',
    suitability: '1.50B Multi-Hop Money Trails & Graph Feature Joins'
  },
  python_fastapi_pytorch: {
    id: 'python_fastapi_pytorch',
    name: 'Python 3.12 (Granian Rust ASGI) + PyTorch 2 OpenXLA',
    shortName: 'Granian + PyTorch XLA',
    license: 'BSD-3 / Apache-2.0 (100% FOSS)',
    platformLockIn: 'Portable · CPU / ROCm / OpenXLA / Standard Linux',
    hftMechanism: 'Rust HTTP/2 Worker Frontend (Granian) + torch.compile Inductor Graph',
    concurrencyModel: 'Multiprocess Rust-Routed Event Loop + C++ Graph Dispatch',
    memorySafety: 'Managed Runtime + Rust Request Ingress',
    throughputMultiplier: 0.92,
    latencyMsBase: 6.4,
    p99JitterUs: 1650,
    memoryFootprintGb: 1.42,
    cpuUtilPct: 71,
    costIndex: '1.45x',
    suitability: 'Rapid ML Research Orchestration & Model Serving'
  }
};

export const ML_TECHNIQUE_META: Record<
  MlTechniqueId,
  {
    id: MlTechniqueId;
    name: string;
    shortName: string;
    objectiveFormula: string;
    accModifier: number;
    lossModifier: number;
    latencyMultiplier: number;
    eceScore: number;
    trainHours1B5: number;
    strengths: string;
  }
> = {
  multimodal_transformer: {
    id: 'multimodal_transformer',
    name: 'Multimodal ArcFace + Swin Transformer Fusion',
    shortName: 'Transformer Fusion',
    objectiveFormula: 'ArcFace Margin + Quality-Gated CE',
    accModifier: 0.018,
    lossModifier: 0.86,
    latencyMultiplier: 1.05,
    eceScore: 0.012,
    trainHours1B5: 78.4,
    strengths: 'Highest TAR@FAR=1e-4 across noisy ridges & iris aging'
  },
  heterogeneous_graphsage: {
    id: 'heterogeneous_graphsage',
    name: 'Heterogeneous GraphSAGE + Temporal Attention',
    shortName: 'GraphSAGE + Attn',
    objectiveFormula: 'InfoNCE + Focal Collusion Loss',
    accModifier: 0.015,
    lossModifier: 0.91,
    latencyMultiplier: 0.94,
    eceScore: 0.015,
    trainHours1B5: 54.2,
    strengths: 'Multi-hop black money trails & NavIC impossible-travel'
  },
  federated_contrastive: {
    id: 'federated_contrastive',
    name: 'Federated Contrastive Meta-Calibration',
    shortName: 'Federated Contrastive',
    objectiveFormula: 'Temperature-Scaled Contrastive BCE',
    accModifier: 0.011,
    lossModifier: 0.94,
    latencyMultiplier: 0.78,
    eceScore: 0.011,
    trainHours1B5: 36.8,
    strengths: 'Cross-rail NPCI/UPI/DBT scoring with zero raw PII sharing'
  },
  gbdt_catboost_xgb: {
    id: 'gbdt_catboost_xgb',
    name: 'Stratified GBDT Ensemble (CatBoost + XGBoost)',
    shortName: 'GBDT Ensemble',
    objectiveFormula: 'Class-Weighted Stratified Log-Loss',
    accModifier: -0.014,
    lossModifier: 1.15,
    latencyMultiplier: 0.52,
    eceScore: 0.024,
    trainHours1B5: 19.5,
    strengths: 'Ultra-fast tabular baseline; weaker on raw biometric tensors'
  },
  diffusion_focal_resnet: {
    id: 'diffusion_focal_resnet',
    name: 'Discrete Denoising Diffusion + Focal ResNet',
    shortName: 'Diffusion / ResNet',
    objectiveFormula: 'Hybrid ELBO + Cosine Noise Reg',
    accModifier: 0.008,
    lossModifier: 0.98,
    latencyMultiplier: 1.24,
    eceScore: 0.017,
    trainHours1B5: 108.0,
    strengths: 'Strong missing-modality imputation & knowledge graph denoising'
  }
};

export type EncryptionProfileId =
  | 'pqc_hybrid_vault'
  | 'aes_gcm_mtls13'
  | 'chacha20_wireguard'
  | 'fpe_confidential_tee';

export const ENCRYPTION_MECHANISM_META: Record<
  EncryptionProfileId,
  {
    id: EncryptionProfileId;
    name: string;
    shortName: string;
    atRestCipher: string;
    inTransitProtocol: string;
    keyManagement: string;
    quantumSafe20Yr: string;
    latencyOverheadMs: number;
    p99JitterExtraUs: number;
    throughputMultiplier: number;
    suitability: string;
  }
> = {
  pqc_hybrid_vault: {
    id: 'pqc_hybrid_vault',
    name: 'Post-Quantum Hybrid (ML-KEM-1024 + AES-256-XTS + mTLS 1.3)',
    shortName: 'PQC ML-KEM-1024 + AES-256-XTS',
    atRestCipher: 'AES-256-XTS (NVMe Hot) + ML-KEM-1024 Wrapped DEKs + Shamir HSM WORM Cold',
    inTransitProtocol: 'Hybrid X25519+ML-KEM-1024 mTLS 1.3 + ML-DSA-65 Signed Envelopes',
    keyManagement: 'FIPS 140-3 L3 HSM + 24h Ephemeral DEK Rotation + Quorum Split',
    quantumSafe20Yr: 'Yes (NIST FIPS 203/204 · HNDL Proof)',
    latencyOverheadMs: 0.28,
    p99JitterExtraUs: 55,
    throughputMultiplier: 0.95,
    suitability: 'Sovereign 1.50B–2.41B 20-Year DPI & Biometric Archives'
  },
  aes_gcm_mtls13: {
    id: 'aes_gcm_mtls13',
    name: 'AES-256-GCM-SIV + Hardware mTLS 1.3 (AES-NI / AVX-512)',
    shortName: 'AES-256-GCM-SIV + mTLS 1.3',
    atRestCipher: 'AES-256-GCM-SIV Envelope + Argon2id VID Vault + LUKS2 Block Encryption',
    inTransitProtocol: 'Mutual TLS 1.3 (ECDHE-X25519 + AES-256-GCM) + SPIFFE/SPIRE Certs',
    keyManagement: 'OpenBao / HashiCorp Vault KMIP Auto-Rotation (7-day)',
    quantumSafe20Yr: 'Classical 256-bit (Grover-safe symmetric)',
    latencyOverheadMs: 0.16,
    p99JitterExtraUs: 32,
    throughputMultiplier: 0.98,
    suitability: 'High-Frequency NPCI/UPI Payments & Enterprise Risk Gateways'
  },
  chacha20_wireguard: {
    id: 'chacha20_wireguard',
    name: 'XChaCha20-Poly1305 + WireGuard / QUIC 0-RTT (ARM Edge)',
    shortName: 'XChaCha20 + WireGuard/QUIC',
    atRestCipher: 'XChaCha20-Poly1305 Authenticated Parquet Shards + BLAKE3 Merkle Tree',
    inTransitProtocol: 'WireGuard Noise_IK UDP Kernel Tunnel + QUIC 0-RTT AEAD Frames',
    keyManagement: 'Ed25519 Hardware Root Key + Ephemeral Curve25519 Session Keys',
    quantumSafe20Yr: 'Symmetric-safe + WireGuard PSK Extension',
    latencyOverheadMs: 0.12,
    p99JitterExtraUs: 24,
    throughputMultiplier: 0.99,
    suitability: 'RD Micro-ATMs, Smart Grid Edge & NavIC Field Terminals'
  },
  fpe_confidential_tee: {
    id: 'fpe_confidential_tee',
    name: 'FF3-1 Format-Preserving + Confidential Compute TEE (SEV-SNP)',
    shortName: 'FF3-1 FPE + SEV-SNP RA-TLS',
    atRestCipher: 'NIST FF3-1 Tokenization + AMD SEV-SNP Encrypted RAM + AES-256-GCM Cold',
    inTransitProtocol: 'RA-TLS (Remote Attested TLS 1.3) Terminated Inside Hardware Enclave',
    keyManagement: 'Hardware Root-of-Trust Attestation + Zero-Operator Key Access',
    quantumSafe20Yr: 'Enclave Isolated + ML-KEM Upgradeable',
    latencyOverheadMs: 0.44,
    p99JitterExtraUs: 95,
    throughputMultiplier: 0.91,
    suitability: 'Zero-Trust Cross-Bank Entity Resolution & Healthcare Clinics'
  }
};

export interface HorizonProjectionPoint {
  yearOffset: number;
  calendarYear: number;
  populationUpperM: number;
  populationNominalM: number;
  populationLowerM: number;
  populationUpperB: number;
  populationNominalB: number;
  populationLowerB: number;
  dailyAuthCallsB: number;
  peakQps: number;
  hotStorageTb: number;
  warmColdStorageTb: number;
  expectedAccPct: number;
}

export function computeHorizonProjections(
  baseMillions: number = 1500,
  annualDeltaPct: number = 2.4,
  years: number = 20,
  baseAccuracy: number = 0.985
): HorizonProjectionPoint[] {
  const rate = annualDeltaPct / 100;
  const points: HorizonProjectionPoint[] = [];
  for (let y = 0; y <= years; y++) {
    const upperM = baseMillions * Math.pow(1 + rate, y);
    const lowerM = baseMillions * Math.pow(1 - rate * 0.45, y);
    const nominalM = baseMillions * Math.pow(1 + rate * 0.65, y);
    const dailyAuthB = (nominalM * 0.62) / 1000;
    const peakQps = Math.round(((dailyAuthB * 1_000_000_000) / 86400) * 3.2);
    const hotTb = Number((nominalM * 0.14).toFixed(1));
    const warmColdTb = Number((nominalM * 1.85 + y * 420).toFixed(1));
    // Periodic 3-year continual recalibration maintains >98% accuracy over 20 yrs
    const driftDecay = (y % 3) * 0.0018;
    const expectedAccPct = Number(((baseAccuracy - driftDecay + Math.min(0.006, y * 0.0003)) * 100).toFixed(2));
    points.push({
      yearOffset: y,
      calendarYear: 2026 + y,
      populationUpperM: Number(upperM.toFixed(1)),
      populationNominalM: Number(nominalM.toFixed(1)),
      populationLowerM: Number(lowerM.toFixed(1)),
      populationUpperB: Number((upperM / 1000).toFixed(2)),
      populationNominalB: Number((nominalM / 1000).toFixed(2)),
      populationLowerB: Number((lowerM / 1000).toFixed(2)),
      dailyAuthCallsB: Number(dailyAuthB.toFixed(2)),
      peakQps,
      hotStorageTb: hotTb,
      warmColdStorageTb: warmColdTb,
      expectedAccPct
    });
  }
  return points;
}

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
    haystack.includes('smartgrid') ||
    haystack.includes('vpp') ||
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
    haystack.includes('multimodal') ||
    haystack.includes('diffusioncom') ||
    haystack.includes('ddpm') ||
    haystack.includes('medical imaging') ||
    haystack.includes('swin transformer') ||
    haystack.includes('agri') ||
    haystack.includes('biologics') ||
    haystack.includes('coldchain')
  ) {
    return 'multimodal';
  }
  if (
    haystack.includes('graph') ||
    haystack.includes('protein') ||
    haystack.includes('se(3)') ||
    haystack.includes('opinion dynamics') ||
    haystack.includes('pde') ||
    haystack.includes('financial crime') ||
    haystack.includes('money trail') ||
    haystack.includes('cbdc') ||
    haystack.includes('iso-20022')
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

export function inferBestProgrammingArch(project: FlagshipProject): ProgrammingArchId {
  const hay = `${project.id} ${project.techStack.join(' ')} ${project.performanceIndicators.hardwareTarget}`.toLowerCase();
  if (hay.includes('burn') || hay.includes('micro-atm')) {
    return 'rust_burn_edge';
  }
  if (hay.includes('io_uring') || hay.includes('lmax') || hay.includes('uidai') || hay.includes('cidr')) {
    return 'rust_tokio_libtorch';
  }
  if (hay.includes('seastar') || hay.includes('openvino') || hay.includes('tvm') || hay.includes('yolo')) {
    return 'cpp20_tensorrt_onnx';
  }
  if (hay.includes('financial crime') || hay.includes('money trail') || hay.includes('arrow') || hay.includes('polars')) {
    return 'arrow_flight_polars';
  }
  if (hay.includes('edge') || hay.includes('jetson')) {
    return 'rust_burn_edge';
  }
  return 'rust_tokio_libtorch';
}

export function inferBestMlTechnique(project: FlagshipProject): MlTechniqueId {
  const dt = inferProjectDataType(project);
  const hay = `${project.id} ${project.title} ${project.techStack.join(' ')}`.toLowerCase();
  if (hay.includes('federated') || hay.includes('payment') || hay.includes('upi')) {
    return 'federated_contrastive';
  }
  if (hay.includes('diffusion')) {
    return 'diffusion_focal_resnet';
  }
  if (dt === 'graph') return 'heterogeneous_graphsage';
  if (dt === 'multimodal' || dt === 'vision' || dt === 'nlp') return 'multimodal_transformer';
  return 'gbdt_catboost_xgb';
}

export function inferBestLearningRate(project: FlagshipProject): LearningRatePreset {
  const tech = inferBestMlTechnique(project);
  const dt = inferProjectDataType(project);
  if (tech === 'federated_contrastive') return LEARNING_RATE_PRESETS[0]; // 1e-5 DP-SGD safe
  if (tech === 'multimodal_transformer' && (dt === 'vision' || dt === 'multimodal' || dt === 'nlp')) {
    return LEARNING_RATE_PRESETS[1]; // 3e-5 Transformer optimal
  }
  if (tech === 'diffusion_focal_resnet') return LEARNING_RATE_PRESETS[3]; // 3e-4 Diffusion / InfoNCE
  if (tech === 'gbdt_catboost_xgb' || dt === 'tabular' || dt === 'timeseries') {
    return LEARNING_RATE_PRESETS[4]; // 1e-3 Fast tabular / streaming
  }
  return LEARNING_RATE_PRESETS[2]; // 1e-4 GraphSAGE / Fusion baseline
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
          { name: 'rd_probe_id', dtype: 'string', spec: 'SYN_VID_RD_1_5B', role: 'metadata' },
          { name: 'iris_gabor_2048b', dtype: 'uint64[32]', spec: '2048-bit masked phase code', role: 'tensor' },
          { name: 'fp_face_fused_emb', dtype: 'float16[256]', spec: 'Quality-gated L2 embedding', role: 'tensor' },
          { name: 'navic_geo_fix', dtype: 'float32[2]', spec: 'NavIC lat/lon geofence', role: 'feature' },
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
      if (domainLower.includes('uidai') || domainLower.includes('financial crime') || domainLower.includes('fraud')) {
        return [
          { name: 'vid_entity_node', dtype: 'string', spec: 'VID_1_5B_NODE', role: 'metadata' },
          { name: 'minutiae_graph_vec', dtype: 'float32[128]', spec: 'Fingerprint + Telecom embedding', role: 'tensor' },
          { name: 'navic_hop_kmh', dtype: 'float32', spec: 'NavIC velocity km/h', role: 'feature' },
          { name: 'upi_layering_depth', dtype: 'int32', spec: '1..12 financial hops', role: 'feature' },
          { name: 'fraud_ring_flag', dtype: 'int64', spec: '0 (Verified) / 1 (Illicit Trail)', role: 'target' }
        ];
      }
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
        { name: 'window_t', dtype: 'datetime', spec: '2026-2046 horizon UTC', role: 'metadata' },
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
          { name: 'vid_sector_token', dtype: 'string', spec: 'DPI_TOK_1_5B', role: 'metadata' },
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
        { name: 'sample_id', dtype: 'string', spec: 'SYN_1_5B_ROW', role: 'metadata' },
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
      ? '1.50B -> 2.41B Synthetic Cohort (20-Yr Horizon)'
      : '1,500,000,000 Synthetic Cohort (Δ ±2.4%/yr)';

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

    const row: Record<string, string | number>[] = [];
    const rowObj: Record<string, string | number> = {};
    cols.forEach((col, cIdx) => {
      if (col.role === 'metadata') {
        if (dataType === 'timeseries') {
          const yr = 2026 + (i % 20);
          rowObj[col.name] = `${yr}-11-${String(10 + i).padStart(2, '0')}T${String((i * 3) % 24).padStart(2, '0')}:00Z`;
        } else {
          rowObj[col.name] = `${dataType.slice(0, 3).toUpperCase()}_1_5B_${1000 + i * 7 + seed}`;
        }
      } else if (col.role === 'tensor') {
        const v1 = (rand() * 1.8 - 0.9 + noise).toFixed(2);
        const v2 = (rand() * 1.8 - 0.9 - noise).toFixed(2);
        const v3 = (rand() * 1.8 - 0.9).toFixed(2);
        if (col.dtype.startsWith('int64') || col.dtype.startsWith('uint64')) {
          rowObj[col.name] = `[${Math.floor(rand() * 32000)}, ${Math.floor(rand() * 32000)}, ${Math.floor(rand() * 32000)}, ...]`;
        } else {
          rowObj[col.name] = `[${v1}, ${v2}, ${v3}, ...]`;
        }
      } else if (col.role === 'target') {
        if (dataType === 'timeseries') {
          rowObj[col.name] = Number((108.5 + i * 2.4 + rand() * 14 * varMeta.noiseMultiplier).toFixed(2));
        } else {
          const labels = project.sampleDistribution.map((s) => s.label.split('(')[0].trim());
          const labelIdx = isPositive ? 0 : Math.min(1, labels.length - 1);
          const tag = labels[labelIdx] ? ` (${labels[labelIdx].slice(0, 16)})` : '';
          rowObj[col.name] = `${isPositive ? 1 : 0}${tag}`;
        }
      } else if (col.name.includes('navic')) {
        const lat = (12.9 + rand() * 16.0).toFixed(3);
        const lon = (72.8 + rand() * 14.0).toFixed(3);
        rowObj[col.name] = col.dtype.includes('[2]') ? `[${lat}, ${lon}]` : Number((12.0 + rand() * 980).toFixed(1));
      } else {
        const val =
          cIdx === 1
            ? (isPositive ? 0.65 : -0.45) + rand() * 0.8 + noise
            : rand() * (cIdx === 2 ? 0.95 : 2.5) + Math.abs(noise);
        rowObj[col.name] = Number(val.toFixed(3));
      }
    });
    void row;
    rows.push(rowObj);
  }
  return rows;
}

export function computeLatencyP95P99(
  progArch: ProgrammingArchId,
  dataType: SyntheticDataType,
  mlTech: MlTechniqueId,
  encProfile: EncryptionProfileId = 'pqc_hybrid_vault'
): { p95Ms: number; p99Ms: number; p99JitterUs: number } {
  const progMeta = PROGRAMMING_ARCHITECTURE_META[progArch];
  const typeMeta = SYNTHETIC_DATA_TYPE_META[dataType];
  const mlMeta = ML_TECHNIQUE_META[mlTech];
  const encMeta = ENCRYPTION_MECHANISM_META[encProfile];
  const p95Ms = Number(
    (
      (progMeta.latencyMsBase + typeMeta.baseLatencyMs * 0.35) * mlMeta.latencyMultiplier +
      encMeta.latencyOverheadMs
    ).toFixed(2)
  );
  const p99Ms = Number(
    (
      p95Ms +
      ((progMeta.p99JitterUs + encMeta.p99JitterExtraUs) / 1000) *
        mlMeta.latencyMultiplier *
        (1 + typeMeta.baseLatencyMs * 0.08)
    ).toFixed(2)
  );
  return { p95Ms, p99Ms, p99JitterUs: progMeta.p99JitterUs + encMeta.p99JitterExtraUs };
}

