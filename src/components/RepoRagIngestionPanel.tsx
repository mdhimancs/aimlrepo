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
  ExternalLink
} from 'lucide-react';
import { RagIngestResponse, RagStatusResponse } from '../types';

interface RepoRagIngestionPanelProps {
  ragStatus: RagStatusResponse | null;
  onRefreshStatus: () => Promise<void>;
  onOpenCopilot: () => void;
}

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

const SAMPLE_RAW_TEXT = `Abstract — Continuous-Time Graph Neural Diffusion and Opinion Dynamics
Deep discrete Graph Neural Networks (GCN, GAT) suffer from Dirichlet energy collapse—commonly known as oversmoothing—when stacking more than 4 to 6 propagation layers, particularly on heterophilic benchmarks. We formulate node feature evolution as a continuous reaction-diffusion PDE solved via 4th-order Runge-Kutta (RK4) integration. By incorporating bounded-confidence opinion coupling, edges with high cosine alignment induce smoothing attraction whereas divergent heterophilic neighbors apply bounded repulsion, maintaining non-zero Dirichlet energy across 32 propagation steps and improving heterophilic classification accuracy by +8.7% over GATv2 baselines.`;

export default function RepoRagIngestionPanel({
  ragStatus,
  onRefreshStatus,
  onOpenCopilot
}: RepoRagIngestionPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);
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
              ? `Processing URL ${i + 1}/${urls.length}: fetching & parsing PDF -> LangChain split -> Gemini embedding -> FAISS upsert...`
              : 'Fetching document & extracting text via unpdf -> LangChain RecursiveSplitter -> FAISS IndexFlatIP...'
          );

          // Visual stage progression cues while server executes extract->split->embed->upsert
          const stageTimer1 = setTimeout(() => setActiveStage(2), 450);
          const stageTimer2 = setTimeout(() => setActiveStage(3), 900);

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
          setStatusMessage(
            `Indexed ${urls.length} document${urls.length > 1 ? 's' : ''} (${totalChunksAdded} LangChain chunks) into FAISS IndexFlatIP (${latestResponse.workflowMetrics.embeddingDim}d).`
          );
        }
        setPdfUrlsInput('');
        setTitle('');
      } else {
        if (rawText.trim().length < 20) {
          throw new Error('Please paste at least 20 characters of raw text.');
        }
        setStatusMessage('Splitting text with LangChain RecursiveCharacterTextSplitter & indexing into FAISS...');
        const stageTimer1 = setTimeout(() => setActiveStage(2), 300);
        const stageTimer2 = setTimeout(() => setActiveStage(3), 650);

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
          setStatusMessage(
            `Indexed "${typed.document.title}" (${typed.chunksAdded} chunks · ${typed.workflowMetrics.embeddingDim}d) into FAISS vector store.`
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
    } finally {
      setIsIndexing(false);
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
      if (!res.ok) throw new Error('Failed to clear FAISS store');
      setLastResult(null);
      setStatusMessage('FAISS IndexFlatIP vector store cleared.');
      await onRefreshStatus();
    } catch (err: any) {
      setErrorMsg(err.message);
    }
  };

  const totalDocs = ragStatus?.totalDocuments || 0;
  const totalVectors = ragStatus?.totalVectors || 0;
  const embeddingDim = ragStatus?.embeddingDim || 0;

  return (
    <div className="bg-slate-100 hover:bg-slate-200/60 rounded-2xl border border-slate-300 shadow-xs transition-all duration-200 overflow-hidden">
      {/* Collapsible Header */}
      <div
        onClick={() => setIsExpanded(!isExpanded)}
        className="px-6 py-4 bg-slate-200/70 border-b border-slate-300 flex flex-col md:flex-row md:items-center justify-between gap-3 cursor-pointer"
      >
        <div className="flex items-start md:items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-rose-900 text-white flex items-center justify-center shrink-0 shadow-xs">
            <Database className="w-4 h-4" />
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
              <span className="bg-rose-100 text-rose-900 px-2 py-0.5 rounded font-bold border border-rose-200">
                [LangChain + FAISS RAG Ingest]
              </span>
              <span className="text-blue-900 font-semibold">
                {totalDocs} Docs · {totalVectors} Vector Chunks {embeddingDim > 0 ? `(${embeddingDim}d IndexFlatIP)` : ''}
              </span>
            </div>
            <h3 className="text-base font-bold text-slate-900 mt-0.5">
              Ingest Research PDFs &amp; Raw Text into AI Copilot RAG Knowledge Base
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
        <div className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* LEFT: Ingestion Form (7 cols) */}
          <form onSubmit={handleIngest} className="lg:col-span-7 space-y-4">
            {/* Mode Switcher */}
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
                    : 'e.g., Custom Diffusion Hyperparameter Notes'
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
                      if (!title) setTitle('Graph Neural Diffusion & Opinion Dynamics Notes');
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
                { step: 3, label: '03. Dense Embed' },
                { step: 4, label: '04. FAISS IndexFlatIP' }
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
                    <span>Running LangChain + FAISS Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Chunk &amp; Index into FAISS RAG Store</span>
                  </>
                )}
              </button>

              {lastResult && (
                <div className="text-[11px] font-mono text-slate-700 bg-white px-3 py-1.5 rounded-lg border border-slate-300">
                  Last Run: Extract {lastResult.workflowMetrics.extractMs}ms · Split{' '}
                  {lastResult.workflowMetrics.splitMs}ms · Embed {lastResult.workflowMetrics.embedMs}ms · Index{' '}
                  {lastResult.workflowMetrics.indexMs}ms
                </div>
              )}
            </div>
          </form>

          {/* RIGHT: Indexed FAISS Vector Store Inventory (5 cols) */}
          <div className="lg:col-span-5 bg-white rounded-xl border border-slate-300 p-4 flex flex-col justify-between space-y-3 shadow-xs">
            <div className="flex items-center justify-between border-b border-slate-200 pb-2.5">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-blue-900" />
                <span className="text-xs font-bold text-slate-900">FAISS Indexed Knowledge Store</span>
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
            <div className="grid grid-cols-3 gap-2 text-center bg-slate-100 p-2.5 rounded-xl border border-slate-200 text-xs font-mono">
              <div>
                <span className="text-[10px] text-slate-500 block">Docs</span>
                <span className="font-bold text-slate-900">{totalDocs}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">Vectors</span>
                <span className="font-bold text-blue-900">{totalVectors}</span>
              </div>
              <div>
                <span className="text-[10px] text-slate-500 block">Dim</span>
                <span className="font-bold text-emerald-900">{embeddingDim > 0 ? `${embeddingDim}d` : '—'}</span>
              </div>
            </div>

            {/* Document List */}
            <div className="flex-1 max-h-56 overflow-y-auto space-y-2 pr-1">
              {totalDocs === 0 ? (
                <div className="py-8 text-center text-xs text-slate-500 space-y-1.5">
                  <Cpu className="w-6 h-6 text-slate-400 mx-auto" />
                  <p className="font-semibold text-slate-700">No documents indexed in FAISS yet.</p>
                  <p>Paste raw text or load an arXiv PDF URL on the left to populate the RAG assistant.</p>
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
                          <span className="text-xs font-bold text-slate-900 truncate">{doc.title}</span>
                        </div>
                        <div className="text-[10px] font-mono text-slate-600 mt-0.5 flex flex-wrap items-center gap-2">
                          <span className="text-emerald-900 font-semibold">{doc.chunkCount} chunks</span>
                          <span>·</span>
                          <span>{doc.charCount.toLocaleString()} chars</span>
                          {doc.pageCount && (
                            <>
                              <span>·</span>
                              <span>{doc.pageCount} pages</span>
                            </>
                          )}
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
                        title="Remove document from FAISS index"
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
              <span>{ragStatus?.indexType || 'FAISS IndexFlatIP (Cosine)'}</span>
              <button
                type="button"
                onClick={onOpenCopilot}
                className="text-blue-900 font-bold hover:underline flex items-center gap-1"
              >
                <span>Open RAG Assistant</span>
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
