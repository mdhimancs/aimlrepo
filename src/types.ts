export interface ProjectBlueprint {
  overview: string;
  phases: {
    stepNumber: number;
    title: string;
    description: string;
    pythonSnippet: string;
  }[];
  recommendedStack: string[];
  estimatedTimeWeeks: number;
  whitePaper?: {
    problem: string;
    breakingGround: string;
    scope: string;
    functionality: string;
    abstract: string;
    methodology: string;
    architecture: string;
    benchmarks: string;
    productionConsiderations: string;
  };
}

export type OpExCostKey =
  | 'infra'
  | 'aiOps'
  | 'mlOps'
  | 'energy'
  | 'manpower'
  | 'maintenance';

export interface CapExBreakdownItem {
  label: string;
  amountUsd: number;
  sharePct: number;
  notes: string;
}

export interface OpExBreakdownItem {
  key: OpExCostKey;
  label: string;
  shortLabel: string;
  description: string;
  year1Usd: number;
  year20Usd: number;
  total20YrUsd: number;
  sharePct: number;
}

export interface YearlyOpExRow {
  year: number;
  calendarYear: number;
  scaleFactor: number;
  infra: number;
  aiOps: number;
  mlOps: number;
  energy: number;
  manpower: number;
  maintenance: number;
  totalOpEx: number;
  yoyDeltaPct: number;
  cumulativeOpEx: number;
  cumulativeTco: number;
}

export interface ProjectEconomics {
  deploymentTier:
    | 'Sovereign / World-Scale (1.50B–2.41B)'
    | 'Enterprise / Research Cluster'
    | 'Production Multi-Node'
    | 'Lean Edge / Serverless';
  totalCapExUsd: number;
  capExBreakdown: CapExBreakdownItem[];
  opExYear1Usd: number;
  opExYear10Usd: number;
  opExYear20Usd: number;
  opExYoYGrowthPct: number;
  totalOpEx20YrUsd: number;
  totalTco20YrUsd: number;
  estimatedPowerKw: number;
  annualEnergyMwhY1: number;
  fteHeadcount: number;
  opExCategories: OpExBreakdownItem[];
  yearlySchedule: YearlyOpExRow[];
  unitCostSummary: string;
}

export interface FlagshipProject {
  id: string;
  title: string;
  domain: string;
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced' | 'Research Grade';
  sourceName?: string;
  sourceUrl?: string;
  description: string;
  objective: string;
  metrics: string;
  trainingLoss: string;
  trainingAccuracy: string;
  problem: string;
  breakingGround: string;
  epochHistory: { epoch: number; loss: number; accuracy: number; valAccuracy: number }[];
  sampleDistribution: { label: string; percentage: number; count: string }[];
  performanceIndicators: {
    latency: string;
    throughput: string;
    memoryFootprint: string;
    hardwareTarget: string;
  };
  techStack: string[];
  architectureFlow: string[];
  pipeline: {
    title: string;
    desc: string;
    code: string;
    language?: 'python' | 'rust';
  }[];
  whitePaper: {
    problem: string;
    breakingGround: string;
    scope: string;
    functionality: string;
    abstract: string;
    methodology: string;
    architecture: string;
    benchmarks: string;
    productionConsiderations: string;
  };
  economics?: ProjectEconomics;
}

export interface BlogArticle {
  id: string;
  title: string;
  category: string;
  readTime: string;
  date: string;
  authors: string;
  summary: string;
  tags: string[];
  sections: {
    heading: string;
    colorClass: 'text-rose-900' | 'text-blue-900' | 'text-emerald-900' | 'text-slate-900';
    dotClass: 'bg-rose-900' | 'bg-blue-900' | 'bg-emerald-900' | 'bg-slate-900';
    body: string;
    codeSnippet?: {
      filename: string;
      language: string;
      code: string;
    };
  }[];
}

export type RagProcessingStatus = 'indexed' | 'processing' | 'degraded' | 'failed';
export type RagSearchMode = 'hybrid' | 'dense' | 'bm25';

export interface RagDocumentTokenStats {
  sourceTokens: number;
  totalChunkTokens: number;
  overlapTokens: number;
  avgChunkTokens: number;
  minChunkTokens: number;
  maxChunkTokens: number;
}

export interface RagDocumentQualityMetrics {
  overallScore: number;
  cleanlinessPct: number;
  chunkUniformityPct: number;
  overlapRatioPct: number;
  orphanChunks: number;
  uniqueTerms: number;
  avgChunkChars: number;
  verdict: 'Optimal' | 'Good' | 'Needs Tuning';
}

export interface RagDocumentSummary {
  id: string;
  title: string;
  sourceType: 'raw_text' | 'pdf_url';
  sourceUrl?: string;
  pageCount?: number;
  charCount: number;
  chunkCount: number;
  embeddingDim: number;
  indexedAt: string;
  preview: string;
  status: RagProcessingStatus;
  statusNote?: string;
  stageStatus: {
    extract: 'ok' | 'failed';
    split: 'ok' | 'failed';
    denseEmbed: 'ok' | 'skipped' | 'failed';
    bm25Index: 'ok' | 'failed';
  };
  tokenStats: RagDocumentTokenStats;
  qualityMetrics: RagDocumentQualityMetrics;
  latencyMs: {
    extractMs: number;
    splitMs: number;
    embedMs: number;
    bm25Ms: number;
    indexMs: number;
    totalMs: number;
  };
}

export interface RagAuditEvent {
  id: string;
  docTitle: string;
  sourceType: 'raw_text' | 'pdf_url';
  status: RagProcessingStatus;
  timestamp: string;
  totalMs: number;
  tokensProcessed: number;
  qualityScore: number;
  message: string;
}

export interface RagCitation {
  docId: string;
  docTitle: string;
  source: string;
  chunkIndex: number;
  score: number;
  denseScore: number;
  bm25Score: number;
  denseRank?: number;
  bm25Rank?: number;
  matchedTerms: string[];
  tokenCount: number;
  retrievalMode: RagSearchMode;
  snippet: string;
}

export interface RagSearchResponse {
  query: string;
  mode: RagSearchMode;
  alpha: number;
  latencyMs: {
    denseMs: number;
    bm25Ms: number;
    fusionMs: number;
    totalMs: number;
  };
  queryTokens: string[];
  results: RagCitation[];
}

export interface RagStatusResponse {
  indexType: string;
  embeddingModel: string;
  splitter: string;
  totalDocuments: number;
  totalVectors: number;
  embeddingDim: number;
  documents: RagDocumentSummary[];
  bm25Stats: {
    totalUniqueTerms: number;
    avgDocLengthTokens: number;
    k1: number;
    b: number;
  };
  tokenUsageTotals: {
    totalSourceTokens: number;
    totalChunkTokens: number;
    totalOverlapTokens: number;
    totalIngestTokens: number;
    queryEmbeddingTokens: number;
  };
  auditSummary: {
    avgQualityScore: number;
    indexedDocs: number;
    degradedDocs: number;
    failedDocs: number;
    totalOrphanChunks: number;
    totalQueriesExecuted: number;
    avgQueryLatencyMs: number;
  };
  recentAuditEvents: RagAuditEvent[];
}

export interface RagIngestResponse {
  document: RagDocumentSummary;
  chunksAdded: number;
  totalVectors: number;
  workflowMetrics: {
    extractMs: number;
    splitMs: number;
    embedMs: number;
    bm25Ms: number;
    indexMs: number;
    embeddingDim: number;
  };
}

export type SyntheticDataType =
  | 'tabular'
  | 'vision'
  | 'nlp'
  | 'timeseries'
  | 'graph'
  | 'multimodal'
  | 'audio';

export type SyntheticVariant = 'balanced' | 'imbalanced' | 'ood_noisy' | 'scaled_100k';

export interface SyntheticColumnSchema {
  name: string;
  dtype: string;
  spec: string;
  role: 'feature' | 'target' | 'tensor' | 'metadata';
}

export interface SyntheticDatasetProfile {
  datasetName: string;
  defaultDataType: SyntheticDataType;
  dataTypeLabel: string;
  tensorShape: string;
  sampleCountLabel: string;
  recommendedLoss: string;
  noiseSigma: string;
  columns: SyntheticColumnSchema[];
}


