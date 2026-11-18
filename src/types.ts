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
}

export interface RagCitation {
  docId: string;
  docTitle: string;
  source: string;
  chunkIndex: number;
  score: number;
  snippet: string;
}

export interface RagStatusResponse {
  indexType: string;
  embeddingModel: string;
  splitter: string;
  totalDocuments: number;
  totalVectors: number;
  embeddingDim: number;
  documents: RagDocumentSummary[];
}

export interface RagIngestResponse {
  document: RagDocumentSummary;
  chunksAdded: number;
  totalVectors: number;
  workflowMetrics: {
    extractMs: number;
    splitMs: number;
    embedMs: number;
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


