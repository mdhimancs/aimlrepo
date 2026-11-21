import React, { useState } from 'react';
import {
  Database,
  FileText,
  Link2,
  Sparkles,
  Trash2,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Layers,
  Cpu,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Search,
  BarChart3,
  SlidersHorizontal
} from 'lucide-react';
import {
  RagIngestResponse,
  RagSearchMode,
  RagSearchResponse,
  RagStatusResponse
} from '../types';
import RepoRagAuditDashboard from './RepoRagAuditDashboard';

interface RepoRagIngestionPanelProps {
  ragStatus: RagStatusResponse | null;
  onRefreshStatus: () => Promise<void>;
  onOpenCopilot: () => void;
}

type PanelSubTab = 'ingest' | 'hybrid_search' | 'audit';

const SAMPLE_PDF_PRESETS = [
  {
    label: 'DiffusionCom (arXiv:2504.06543)',
    title: 'DiffusionCom: Multimodal Knowledge Graph Completion',
    url: 'https://arxiv.org/pdf/2504.06543'
  },
  {
    label: 'Opinion Dynamics Graph PDE (arXiv:2508.11249)',
    title: 'Graph Neural Diffusion via Generalized Opinion Dynamics',
    url: 'https://arxiv.org/pdf/2508.11249'
  },
  {
    label: 'Attention Is All You Need (arXiv:1706.03762)',
    title: 'Attention Is All You Need (Vaswani et al.)',
    url: 'https://arxiv.org/pdf/1706.03762'
  }
];

const SAMPLE_RAW_TEXT = `Abstract — Continuous-Time Graph Neural Diffusion, PQC Vaults & HFT Rust Inference
Deep discrete Graph Neural Networks (GCN, GAT) suffer from Dirichlet energy collapse—commonly known as oversmoothing—when stacking more than 4 to 6 propagation layers, particularly on heterophilic benchmarks. We formulate node feature evolution as a continuous reaction-diffusion PDE solved via 4th-order Runge-Kutta (RK4) integration. By incorporating bounded-confidence opinion coupling, edges with high cosine alignment induce smoothing attraction whereas divergent heterophilic neighbors apply bounded repulsion, maintaining non-zero Dirichlet energy across 32 propagation steps and improving heterophilic classification accuracy by +8.7% over GATv2 baselines.

Deployment Stack: Kernel-bypass Rust io_uring + LMAX SPSC ring buffers achieve sub-1.5ms P99 tail latency with Tract/ONNX INT8 kernels. At-rest identity shards are vaulted using AES-256-XTS envelope keys wrapped with NIST FIPS 203 ML-KEM-1024 post-quantum key encapsulation over Hybrid mTLS 1.3 transport.`;

const SAMPLE_TECH_QUERIES = [
  'RK4 Dirichlet energy oversmoothing GATv2',
  'ML-KEM-1024 AES-256-XTS mTLS 1.3',
  'Rust io_uring P99 latency LMAX ring'
];

export default function RepoRagIngestionPanel({
  ragStatus,
  onRefreshStatus,
  onOpenCopilot
}: RepoRagIngestionPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [activeSubTab, setActiveSubTab] = useState<PanelSubTab>('ingest');
  const [mode, setMode] = useState<'pdf_url' | 'raw_text'>('pdf_url');
  const [title, setTitle] = useState('');
  const [pdfUrlsInput, setPdfUrlsInput] = useState('');
  const [rawText, setRawText] = useState('');
  const [chunkSize, setChunkSize] = useState(800);
  const [chunkOverlap, setChunkOverlap] = useState(120);

  const [isIndexing, setIsIndexing] = useState(false);
  const [activeStage, setActiveStage] = useState<number>(0);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<RagIngestResponse | null>(null);

  // Hybrid Search Evaluator State
  const [searchQuery, setSearchQuery] = useState('');
  const [searchMode, setSearchMode] = useState<RagSearchMode>('hybrid');
  const [alphaWeight, setAlphaWeight] = useState<number>(0.55);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchResponse, setSearchResponse] = useState<RagSearchResponse | null>(null);

  const handleIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setStatusMessage(null);
    setIsIndexing(true);
    setActiveStage(1);

    try {
      if (mode === 'pdf_url') {
        const urls = pdfUrlsInput
          .split(/\r?\n/)
          .map((u) => u.trim())
          .filter(Boolean);

        if (urls.length === 0) {
          throw new Error('Please enter at least one PDF or document URL.');
        }

        let latestResponse: RagIngestResponse | null = null;
        let totalChunksAdded = 0;

        for (let i = 0; i < urls.length; i++) {
          const url = urls[i];
          setActiveStage(1);
          setStatusMessage(
            urls.length > 1
              ? `Processing URL ${i + 1}/${urls.length}: extract -> LangChain split -> Dense + BM25 hybrid index...`
              : 'Extracting document -> LangChain RecursiveSplitter -> FAISS Dense + BM25 Okapi Sparse...'
          );

          const stageTimer1 = setTimeout(() => setActiveStage(2), 400);
          const stageTimer2 = setTimeout(() => setActiveStage(3), 850);

          try {
            const res = await fetch('/api/rag/ingest', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                mode: 'pdf_url',
                title: urls.length === 1 ? title : undefined,
                pdfUrl: url,
                chunkSize,
                chunkOverlap
              })
            });
            const data = await res.json();
            if (!res.ok) {
              throw new Error(data.error || `Failed to ingest ${url}`);
            }
            setActiveStage(4);
            latestResponse = data as RagIngestResponse;
            totalChunksAdded += latestResponse.chunksAdded;
          } finally {
            clearTimeout(stageTimer1);
            clearTimeout(stageTimer2);
          }
        }

        if (latestResponse) {
          setLastResult(latestResponse);
          const dimLabel =
            latestResponse.workflowMetrics.embeddingDim > 0
              ? `${latestResponse.workflowMetrics.embeddingDim}d Dense + BM25 Sparse`
              : 'BM25 Sparse Store';
          setStatusMessage(
            `Indexed ${urls.length} document${urls.length > 1 ? 's' : ''} (${totalChunksAdded} chunks · ${dimLabel}). Quality: ${latestResponse.document.qualityMetrics?.overallScore ?? 0}/100.`
          );
        }
        setPdfUrlsInput('');
        setTitle('');
      } else {
        if (rawText.trim().length < 20) {
          throw new Error('Please paste at least 20 characters of raw text.');
        }
        setStatusMessage('Splitting text with LangChain -> building Dense FAISS & BM25 Okapi Sparse indices...');
        const stageTimer1 = setTimeout(() => setActiveStage(2), 250);
        const stageTimer2 = setTimeout(() => setActiveStage(3), 600);

        try {
          const res = await fetch('/api/rag/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              mode: 'raw_text',
              title,
              rawText,
              chunkSize,
              chunkOverlap
            })
          });
          const data = await res.json();
          if (!res.ok) {
            throw new Error(data.error || 'Failed to ingest raw text');
          }
          setActiveStage(4);
          const typed = data as RagIngestResponse;
          setLastResult(typed);
          const dimLabel =
            typed.workflowMetrics.embeddingDim > 0
              ? `${typed.workflowMetrics.embeddingDim}d Dense + BM25`
              : 'BM25 Sparse';
          setStatusMessage(
            `Indexed "${typed.document.title}" (${typed.chunksAdded} chunks · ${dimLabel} · ${typed.document.tokenStats?.totalChunkTokens ?? 0} tokens · Quality ${typed.document.qualityMetrics?.overallScore ?? 0}/100).`
          );
          setRawText('');
          setTitle('');
        } finally {
          clearTimeout(stageTimer1);
          clearTimeout(stageTimer2);
        }
      }

      await onRefreshStatus();
    } catch (err: any) {
      setErrorMsg(err.message || 'RAG indexing workflow failed.');
      setActiveStage(0);
      await onRefreshStatus();
    } finally {
      setIsIndexing(false);
    }
  };

  const handleHybridSearch = async (e?: React.FormEvent, queryOverride?: string) => {
    if (e) e.preventDefault();
    const q = (queryOverride ?? searchQuery).trim();
    if (!q) {
      setSearchError('Enter a technical query (e.g., RK4 Dirichlet, ML-KEM-1024, io_uring).');
      return;
    }
    setSearchError(null);
    setIsSearching(true);
    try {
      const res = await fetch('/api/rag/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          topK: 5,
          mode: searchMode,
          alpha: alphaWeight
        })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Hybrid search failed');
      }
      setSearchResponse(data as RagSearchResponse);
      await onRefreshStatus();
    } catch (err: any) {
      setSearchError(err.message || 'Failed to execute hybrid search.');
    } finally {
      setIsSearching(false);
    }
  };

  const handleDeleteDoc = async (docId: string) => {
    setErrorMsg(null);
    try {
      const res = await fetch(`/api/rag/documents/${docId}`, { method: 'DELETE' });
      if (!res.ok) throw new Error('Failed to remove document from index');
      await onRefreshStatus();
    } catch (err: any) {
      setErrorMsg(err.message);
    }
  };

  const handleClearAll = async () => {
    setErrorMsg(null);
    try {
      const res = await fetch('/api/rag/clear', { method: 'POST' });
      if (!res.ok) throw new Error('Failed to clear RAG store');
      setLastResult(null);
      setSearchResponse(null);
      setStatusMessage('Hybrid FAISS + BM25 vector store cleared.');
      await onRefreshStatus();
    } catch (err: any) {
      setErrorMsg(err.message);
    }
  };

  const totalDocs = ragStatus?.totalDocuments || 0;
  const totalVectors = ragStatus?.totalVectors || 0;
  const embeddingDim = ragStatus?.embeddingDim || 0;
  const totalTokens = ragStatus?.tokenUsageTotals?.totalChunkTokens || 0;
  const avgQuality = ragStatus?.auditSummary?.avgQualityScore || 0;

  return (
    <div className="bg-slate-100 rounded-2xl border border-slate-300 shadow-xs transition-all duration-200 overflow-hidden">
      {/* Collapsible Header */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="px-6 py-4 bg-slate-200/80 border-b border-slate-300 flex flex-col lg:flex-row lg:items-center justify-between gap-3 cursor-pointer"
      >
        <div className="flex items-start md:items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-rose-900 text-white flex items-center justify-center shrink-0 shadow-xs">
            <Database className="w-4 h-4" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
              <span className="bg-rose-100 text-rose-900 px-2 py-0.5 rounded font-bold border border-rose-200">
                [Hybrid RAG: Dense + BM25 RRF]
              </span>
              <span className="text-blue-900 font-semibold">
                {totalDocs} Docs · {totalVectors} Chunks {embeddingDim > 0 ? `(${embeddingDim}d)` : ''} ·{' '}
                {totalTokens.toLocaleString()} Tokens
              </span>
              {totalDocs > 0 && (
                <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 border border-emerald-200 font-bold">
                  Quality {avgQuality}/100
                </span>
              )}
            </div>
            <h3 className="text-base font-bold text-slate-900 mt-0.5">
              Repository Hybrid RAG Knowledge Base, Search Evaluator &amp; Audit Console
            </h3>
          </div>
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
          {totalVectors > 0 && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onOpenCopilot();
              }}
              className="px-3.5 py-1.5 rounded-xl bg-emerald-900 hover:bg-emerald-800 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs"
            >
              <span>Query RAG Copilot</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
          <span className="p-1.5 rounded-lg bg-white border border-slate-300 text-slate-700">
            {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </span>
        </div>
      </div>

      {isExpanded && (
        <div className="p-6 space-y-5">
          {/* Sub-Navigation Bar: Ingest | Hybrid Search | Audit Dashboard */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-300 pb-3">
            <div className="inline-flex flex-wrap rounded-xl bg-slate-200 p-1 border border-slate-300 gap-1">
              <button
                type="button"
                onClick={() => setActiveSubTab('ingest')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                  activeSubTab === 'ingest'
                    ? 'bg-blue-900 text-white shadow-xs'
                    : 'text-slate-700 hover:text-slate-900'
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <span>1. Document Ingest &amp; Index</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveSubTab('hybrid_search')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                  activeSubTab === 'hybrid_search'
                    ? 'bg-indigo-700 text-white shadow-xs'
                    : 'text-slate-700 hover:text-slate-900'
                }`}
              >
                <Search className="w-3.5 h-3.5" />
                <span>2. Hybrid Search (Dense + BM25)</span>
              </button>
              <button
                type="button"
                onClick={() => setActiveSubTab('audit')}
                className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                  activeSubTab === 'audit'
                    ? 'bg-rose-900 text-white shadow-xs'
                    : 'text-slate-700 hover:text-slate-900'
                }`}
              >
                <BarChart3 className="w-3.5 h-3.5" />
                <span>3. Audit Dashboard ({totalDocs} Docs)</span>
              </button>
            </div>

            <div className="text-[11px] font-mono text-slate-600 flex items-center gap-2">
              <span>BM25 Terms: <strong className="text-slate-900">{ragStatus?.bm25Stats?.totalUniqueTerms || 0}</strong></span>
              <span>·</span>
              <span>Queries: <strong className="text-slate-900">{ragStatus?.auditSummary?.totalQueriesExecuted || 0}</strong></span>
            </div>
          </div>

          {/* SUB-TAB 1: INGEST & INVENTORY */}
          {activeSubTab === 'ingest' && (
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* LEFT: Ingestion Form (7 cols) */}
              <form onSubmit={handleIngest} className="lg:col-span-7 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="inline-flex rounded-xl bg-slate-200 p-1 border border-slate-300">
                    <button
                      type="button"
                      onClick={() => setMode('pdf_url')}
                      className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                        mode === 'pdf_url'
                          ? 'bg-blue-900 text-white shadow-xs'
                          : 'text-slate-700 hover:text-slate-900'
                      }`}
                    >
                      <Link2 className="w-3.5 h-3.5" />
                      <span>PDF / Document URLs</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setMode('raw_text')}
                      className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all ${
                        mode === 'raw_text'
                          ? 'bg-blue-900 text-white shadow-xs'
                          : 'text-slate-700 hover:text-slate-900'
                      }`}
                    >
                      <FileText className="w-3.5 h-3.5" />
                      <span>Paste Raw Text</span>
                    </button>
                  </div>

                  {/* LangChain Splitter Params */}
                  <div className="flex items-center gap-2 text-xs font-mono">
                    <label className="text-slate-700 font-semibold flex items-center gap-1">
                      <span>Chunk:</span>
                      <select
                        value={chunkSize}
                        onChange={(e) => setChunkSize(Number(e.target.value))}
                        className="bg-white border border-slate-300 rounded-lg px-2 py-1 text-slate-900 focus:outline-none focus:border-blue-900"
                      >
                        <option value={400}>400 chars</option>
                        <option value={800}>800 chars</option>
                        <option value={1200}>1200 chars</option>
                        <option value={1600}>1600 chars</option>
                      </select>
                    </label>
                    <label className="text-slate-700 font-semibold flex items-center gap-1">
                      <span>Overlap:</span>
                      <select
                        value={chunkOverlap}
                        onChange={(e) => setChunkOverlap(Number(e.target.value))}
                        className="bg-white border border-slate-300 rounded-lg px-2 py-1 text-slate-900 focus:outline-none focus:border-blue-900"
                      >
                        <option value={80}>80</option>
                        <option value={120}>120</option>
                        <option value={200}>200</option>
                        <option value={250}>250</option>
                      </select>
                    </label>
                  </div>
                </div>

                {/* Optional Title */}
                <div>
                  <label className="block text-xs font-semibold text-slate-800 mb-1">
                    Document Title / Reference Label (Optional)
                  </label>
                  <input
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder={
                      mode === 'pdf_url'
                        ? 'e.g., DiffusionCom Paper (auto-detected from URL if blank)'
                        : 'e.g., Graph Neural Diffusion & PQC Vault Spec'
                    }
                    className="w-full bg-white border border-slate-300 rounded-xl px-3.5 py-2 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-900 shadow-xs"
                  />
                </div>

                {/* Input Field by Mode */}
                {mode === 'pdf_url' ? (
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <label className="text-xs font-semibold text-slate-800">
                        PDF or Web Document URL(s) (one URL per line)
                      </label>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="text-[10px] font-mono text-slate-500">Quick-load arXiv PDF:</span>
                        {SAMPLE_PDF_PRESETS.map((preset) => (
                          <button
                            key={preset.url}
                            type="button"
                            onClick={() => {
                              setPdfUrlsInput(preset.url);
                              setTitle(preset.title);
                            }}
                            className="px-2 py-0.5 rounded-md bg-slate-200 hover:bg-slate-300 text-[10px] font-mono text-blue-900 font-semibold border border-slate-300 transition-colors"
                          >
                            + {preset.label.split(' ')[0]}
                          </button>
                        ))}
                      </div>
                    </div>
                    <textarea
                      rows={3}
                      value={pdfUrlsInput}
                      onChange={(e) => setPdfUrlsInput(e.target.value)}
                      placeholder={'https://arxiv.org/pdf/2504.06543\nhttps://arxiv.org/pdf/2508.11249'}
                      className="w-full bg-white border border-slate-300 rounded-xl px-3.5 py-2.5 text-xs font-mono text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-900 shadow-xs"
                    />
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <label className="text-xs font-semibold text-slate-800">
                        Paste Raw Research Text, White Paper Excerpt, or Code Notes
                      </label>
                      <button
                        type="button"
                        onClick={() => {
                          setRawText(SAMPLE_RAW_TEXT);
                          if (!title) setTitle('Graph Neural Diffusion & PQC Vault Spec');
                        }}
                        className="px-2 py-0.5 rounded-md bg-slate-200 hover:bg-slate-300 text-[10px] font-mono text-blue-900 font-semibold border border-slate-300 transition-colors"
                      >
                        + Fill Sample Research Abstract
                      </button>
                    </div>
                    <textarea
                      rows={4}
                      value={rawText}
                      onChange={(e) => setRawText(e.target.value)}
                      placeholder="Paste full paper sections, experiment logs, dataset documentation, or methodology notes..."
                      className="w-full bg-white border border-slate-300 rounded-xl px-3.5 py-2.5 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-blue-900 shadow-xs leading-relaxed"
                    />
                  </div>
                )}

                {/* Live Workflow Pipeline Indicator */}
                <div className="grid grid-cols-4 gap-2 text-[10px] font-mono">
                  {[
                    { step: 1, label: '01. PDF/Text Extract' },
                    { step: 2, label: '02. LangChain Split' },
                    { step: 3, label: '03. Dense Vector' },
                    { step: 4, label: '04. BM25 + FAISS Upsert' }
                  ].map((st) => {
                    const isDone = activeStage > st.step || (!isIndexing && activeStage === 4);
                    const isRunning = isIndexing && activeStage === st.step;
                    return (
                      <div
                        key={st.step}
                        className={`px-2 py-1.5 rounded-lg border text-center font-semibold transition-all ${
                          isDone
                            ? 'bg-emerald-100 text-emerald-900 border-emerald-300'
                            : isRunning
                              ? 'bg-blue-900 text-white border-blue-900 animate-pulse'
                              : 'bg-white text-slate-600 border-slate-300'
                        }`}
                      >
                        {st.label}
                      </div>
                    );
                  })}
                </div>

                {/* Alerts */}
                {errorMsg && (
                  <div className="p-3 rounded-xl bg-rose-50 border border-rose-300 text-rose-900 text-xs flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{errorMsg}</span>
                  </div>
                )}
                {statusMessage && !errorMsg && (
                  <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-300 text-emerald-900 text-xs flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{statusMessage}</span>
                  </div>
                )}

                {/* Action Row */}
                <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
                  <button
                    type="submit"
                    disabled={isIndexing}
                    className="px-5 py-2.5 bg-blue-900 hover:bg-blue-800 disabled:opacity-60 text-white text-xs font-semibold rounded-xl flex items-center gap-2 transition-colors shadow-xs"
                  >
                    {isIndexing ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        <span>Running Hybrid Index Pipeline...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5" />
                        <span>Chunk &amp; Index (Dense + BM25)</span>
                      </>
                    )}
                  </button>

                  {lastResult && (
                    <div className="text-[11px] font-mono text-slate-700 bg-white px-3 py-1.5 rounded-lg border border-slate-300">
                      Ext {lastResult.workflowMetrics.extractMs}ms · Split {lastResult.workflowMetrics.splitMs}ms ·
                      Emb {lastResult.workflowMetrics.embedMs}ms · BM25 {lastResult.workflowMetrics.bm25Ms}ms
                    </div>
                  )}
                </div>
              </form>

              {/* RIGHT: Indexed Hybrid Knowledge Store Inventory (5 cols) */}
              <div className="lg:col-span-5 bg-white rounded-xl border border-slate-300 p-4 flex flex-col justify-between space-y-3 shadow-xs">
                <div className="flex items-center justify-between border-b border-slate-200 pb-2.5">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-blue-900" />
                    <span className="text-xs font-bold text-slate-900">Hybrid Indexed Knowledge Store</span>
                  </div>
                  {totalDocs > 0 && (
                    <button
                      type="button"
                      onClick={handleClearAll}
                      className="text-[11px] font-mono text-rose-900 hover:underline font-semibold"
                    >
                      Clear All
                    </button>
                  )}
                </div>

                {/* Spec Summary */}
                <div className="grid grid-cols-4 gap-2 text-center bg-slate-100 p-2.5 rounded-xl border border-slate-200 text-xs font-mono">
                  <div>
                    <span className="text-[10px] text-slate-500 block">Docs</span>
                    <span className="font-bold text-slate-900">{totalDocs}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Chunks</span>
                    <span className="font-bold text-blue-900">{totalVectors}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Tokens</span>
                    <span className="font-bold text-indigo-900">{totalTokens.toLocaleString()}</span>
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 block">Quality</span>
                    <span className="font-bold text-emerald-900">{totalDocs > 0 ? `${avgQuality}%` : '—'}</span>
                  </div>
                </div>

                {/* Document List */}
                <div className="flex-1 max-h-56 overflow-y-auto space-y-2 pr-1">
                  {totalDocs === 0 ? (
                    <div className="py-8 text-center text-xs text-slate-500 space-y-1.5">
                      <Cpu className="w-6 h-6 text-slate-400 mx-auto" />
                      <p className="font-semibold text-slate-700">No documents indexed yet.</p>
                      <p>Paste raw text or load an arXiv PDF URL to populate Dense + BM25 Hybrid Search &amp; Audit stats.</p>
                    </div>
                  ) : (
                    ragStatus?.documents.map((doc) => (
                      <div
                        key={doc.id}
                        className="p-2.5 rounded-lg bg-slate-100 hover:bg-slate-200/80 border border-slate-300 transition-colors space-y-1"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span
                                className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold border ${
                                  doc.sourceType === 'pdf_url'
                                    ? 'bg-rose-100 text-rose-900 border-rose-200'
                                    : 'bg-blue-100 text-blue-900 border-blue-200'
                                }`}
                              >
                                {doc.sourceType === 'pdf_url' ? 'PDF/URL' : 'TEXT'}
                              </span>
                              <span
                                className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold border ${
                                  doc.status === 'indexed'
                                    ? 'bg-emerald-100 text-emerald-900 border-emerald-200'
                                    : 'bg-amber-100 text-amber-900 border-amber-200'
                                }`}
                              >
                                {doc.status === 'indexed' ? 'INDEXED' : 'BM25'}
                              </span>
                              <span className="text-xs font-bold text-slate-900 truncate">{doc.title}</span>
                            </div>
                            <div className="text-[10px] font-mono text-slate-600 mt-0.5 flex flex-wrap items-center gap-2">
                              <span className="text-emerald-900 font-semibold">{doc.chunkCount} chunks</span>
                              <span>·</span>
                              <span className="text-blue-900 font-semibold">
                                {(doc.tokenStats?.totalChunkTokens || 0).toLocaleString()} tok
                              </span>
                              <span>·</span>
                              <span className="text-indigo-900 font-semibold">
                                Q:{doc.qualityMetrics?.overallScore ?? 0}/100
                              </span>
                              {doc.sourceUrl && (
                                <a
                                  href={doc.sourceUrl}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-900 hover:underline inline-flex items-center gap-0.5"
                                >
                                  <span>Source</span>
                                  <ExternalLink className="w-2.5 h-2.5" />
                                </a>
                              )}
                            </div>
                          </div>
                          <button
                            type="button"
                            onClick={() => handleDeleteDoc(doc.id)}
                            title="Remove document from RAG index"
                            className="p-1 rounded hover:bg-rose-100 text-slate-500 hover:text-rose-900 transition-colors shrink-0"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                        <p className="text-[11px] text-slate-600 line-clamp-2 leading-snug">{doc.preview}...</p>
                      </div>
                    ))
                  )}
                </div>

                <div className="pt-2 border-t border-slate-200 flex items-center justify-between text-[11px] font-mono text-slate-600">
                  <button
                    type="button"
                    onClick={() => setActiveSubTab('audit')}
                    className="text-rose-900 font-bold hover:underline"
                  >
                    Inspect Full Audit Metrics →
                  </button>
                  <button
                    type="button"
                    onClick={onOpenCopilot}
                    className="text-blue-900 font-bold hover:underline flex items-center gap-1"
                  >
                    <span>Open Copilot</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* SUB-TAB 2: HYBRID SEARCH EVALUATOR (DENSE + BM25 RRF) */}
          {activeSubTab === 'hybrid_search' && (
            <div className="space-y-4">
              <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-4 shadow-2xs">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <h4 className="text-sm font-bold text-slate-900">
                      Hybrid Retrieval Lab (Dense Gemini Embedding + BM25 Okapi Sparse RRF)
                    </h4>
                    <p className="text-xs text-slate-600">
                      Test exact technical acronyms, cryptographic suites, and semantic concepts across {totalVectors} indexed chunks.
                    </p>
                  </div>

                  {/* Mode Toggle */}
                  <div className="inline-flex rounded-xl bg-slate-100 p-1 border border-slate-300 text-xs font-mono">
                    {(
                      [
                        { id: 'hybrid', label: 'Hybrid (RRF)' },
                        { id: 'dense', label: 'Dense Vector' },
                        { id: 'bm25', label: 'BM25 Sparse' }
                      ] as const
                    ).map((m) => (
                      <button
                        key={m.id}
                        type="button"
                        onClick={() => setSearchMode(m.id)}
                        className={`px-3 py-1 rounded-lg font-semibold transition-all ${
                          searchMode === m.id
                            ? 'bg-indigo-700 text-white shadow-2xs'
                            : 'text-slate-700 hover:text-slate-900'
                        }`}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Query Input + Weight Slider */}
                <form onSubmit={(e) => handleHybridSearch(e)} className="space-y-3">
                  <div className="flex flex-col md:flex-row gap-3">
                    <div className="relative flex-1">
                      <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-1/2 -translate-y-1/2" />
                      <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Enter technical query (e.g. 'ML-KEM-1024', 'RK4 Dirichlet energy', 'io_uring P99')..."
                        className="w-full bg-slate-50 border border-slate-300 rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-indigo-700"
                      />
                    </div>
                    <button
                      type="submit"
                      disabled={isSearching || totalVectors === 0}
                      className="px-5 py-2.5 rounded-xl bg-indigo-700 hover:bg-indigo-600 disabled:opacity-50 text-white text-xs font-semibold flex items-center justify-center gap-2 transition-colors shadow-xs"
                    >
                      {isSearching ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          <span>Searching...</span>
                        </>
                      ) : (
                        <>
                          <Search className="w-3.5 h-3.5" />
                          <span>Execute Hybrid Search</span>
                        </>
                      )}
                    </button>
                  </div>

                  <div className="flex flex-wrap items-center justify-between gap-3 text-[11px] font-mono">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="text-slate-500">Quick Queries:</span>
                      {SAMPLE_TECH_QUERIES.map((sq) => (
                        <button
                          key={sq}
                          type="button"
                          onClick={() => {
                            setSearchQuery(sq);
                            if (totalVectors > 0) {
                              handleHybridSearch(undefined, sq);
                            }
                          }}
                          className="px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 border border-slate-300 text-blue-900 font-semibold"
                        >
                          {sq}
                        </button>
                      ))}
                    </div>

                    {searchMode === 'hybrid' && (
                      <label className="flex items-center gap-2 text-slate-700 bg-slate-50 px-3 py-1 rounded-lg border border-slate-200">
                        <SlidersHorizontal className="w-3.5 h-3.5 text-indigo-700" />
                        <span>
                          α Dense {Math.round(alphaWeight * 100)}% / BM25 {Math.round((1 - alphaWeight) * 100)}%
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={alphaWeight}
                          onChange={(e) => setAlphaWeight(Number(e.target.value))}
                          className="w-24 accent-indigo-700"
                        />
                      </label>
                    )}
                  </div>
                </form>

                {searchError && (
                  <div className="p-3 rounded-xl bg-rose-50 border border-rose-300 text-rose-900 text-xs flex items-center gap-2">
                    <AlertCircle className="w-4 h-4 shrink-0" />
                    <span>{searchError}</span>
                  </div>
                )}

                {totalVectors === 0 && (
                  <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-center justify-between">
                    <span>Index is currently empty. Add a sample PDF URL or raw technical text in Tab 1 to test Hybrid Retrieval.</span>
                    <button
                      type="button"
                      onClick={() => setActiveSubTab('ingest')}
                      className="px-3 py-1 rounded-lg bg-amber-900 text-white font-semibold text-[11px]"
                    >
                      Go to Ingest
                    </button>
                  </div>
                )}
              </div>

              {/* Search Results */}
              {searchResponse && (
                <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-3 shadow-2xs">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-2 text-xs font-mono">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-bold text-slate-900">
                        Top-{searchResponse.results.length} Results ({searchResponse.mode.toUpperCase()})
                      </span>
                      <span className="text-slate-500">·</span>
                      <span className="text-indigo-800">
                        Query Terms: {searchResponse.queryTokens.slice(0, 8).join(', ') || '—'}
                      </span>
                    </div>
                    <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
                      <span>Dense: {searchResponse.latencyMs.denseMs}ms</span>
                      <span>·</span>
                      <span>BM25: {searchResponse.latencyMs.bm25Ms}ms</span>
                      <span>·</span>
                      <span>Fusion: {searchResponse.latencyMs.fusionMs}ms</span>
                      <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 font-bold">
                        Total {searchResponse.latencyMs.totalMs}ms
                      </span>
                    </div>
                  </div>

                  {searchResponse.results.length === 0 ? (
                    <p className="text-xs text-slate-500 py-4 text-center">No matching chunks found for this query.</p>
                  ) : (
                    <div className="space-y-2.5">
                      {searchResponse.results.map((cit, i) => (
                        <div
                          key={`${cit.docId}-${cit.chunkIndex}`}
                          className="p-3 rounded-xl bg-slate-50 border border-slate-200 space-y-1.5"
                        >
                          <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-[11px]">
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="px-2 py-0.5 rounded bg-indigo-700 text-white font-bold">
                                #{i + 1}
                              </span>
                              <span className="font-bold text-slate-900 truncate">{cit.docTitle}</span>
                              <span className="text-slate-500">Chunk #{cit.chunkIndex} ({cit.tokenCount} tok)</span>
                            </div>

                            <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
                              <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 border border-emerald-200 font-bold">
                                Score: {cit.score}
                              </span>
                              <span className="px-2 py-0.5 rounded bg-blue-100 text-blue-900 border border-blue-200">
                                Dense Cosine: {cit.denseScore} {cit.denseRank ? `(R${cit.denseRank})` : ''}
                              </span>
                              <span className="px-2 py-0.5 rounded bg-rose-100 text-rose-900 border border-rose-200">
                                BM25: {cit.bm25Score} {cit.bm25Rank ? `(R${cit.bm25Rank})` : ''}
                              </span>
                            </div>
                          </div>

                          <p className="text-xs text-slate-700 leading-relaxed">{cit.snippet}</p>

                          {cit.matchedTerms.length > 0 && (
                            <div className="flex flex-wrap items-center gap-1 pt-1">
                              <span className="text-[10px] font-mono text-slate-500 mr-1">Matched BM25 Tokens:</span>
                              {cit.matchedTerms.map((term) => (
                                <span
                                  key={term}
                                  className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-100 text-amber-900 border border-amber-200 font-semibold"
                                >
                                  {term}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* SUB-TAB 3: RAG INGESTION & TOKEN AUDIT DASHBOARD */}
          {activeSubTab === 'audit' && (
            <RepoRagAuditDashboard
              ragStatus={ragStatus}
              onDeleteDoc={handleDeleteDoc}
              onSwitchToIngest={() => setActiveSubTab('ingest')}
            />
          )}
        </div>
      )}
    </div>
  );
}
