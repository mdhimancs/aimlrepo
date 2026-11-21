import React from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Cpu,
  Database,
  FileCheck2,
  Gauge,
  Layers,
  ShieldCheck,
  Trash2,
  XCircle
} from 'lucide-react';
import { RagStatusResponse } from '../types';

interface RepoRagAuditDashboardProps {
  ragStatus: RagStatusResponse | null;
  onDeleteDoc: (docId: string) => Promise<void>;
  onSwitchToIngest: () => void;
}

export default function RepoRagAuditDashboard({
  ragStatus,
  onDeleteDoc,
  onSwitchToIngest
}: RepoRagAuditDashboardProps) {
  const docs = ragStatus?.documents || [];
  const auditSummary = ragStatus?.auditSummary || {
    avgQualityScore: 0,
    indexedDocs: 0,
    degradedDocs: 0,
    failedDocs: 0,
    totalOrphanChunks: 0,
    totalQueriesExecuted: 0,
    avgQueryLatencyMs: 0
  };
  const tokenTotals = ragStatus?.tokenUsageTotals || {
    totalSourceTokens: 0,
    totalChunkTokens: 0,
    totalOverlapTokens: 0,
    totalIngestTokens: 0,
    queryEmbeddingTokens: 0
  };
  const bm25Stats = ragStatus?.bm25Stats || {
    totalUniqueTerms: 0,
    avgDocLengthTokens: 0,
    k1: 1.5,
    b: 0.75
  };
  const auditEvents = ragStatus?.recentAuditEvents || [];

  const overlapOverheadPct =
    tokenTotals.totalSourceTokens > 0
      ? ((tokenTotals.totalOverlapTokens / tokenTotals.totalSourceTokens) * 100).toFixed(1)
      : '0.0';

  const maxDocTokens = Math.max(1, ...docs.map((d) => d.tokenStats?.totalChunkTokens || 1));

  return (
    <div className="space-y-5">
      {/* Top KPI Audit Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
        {/* Card 1: Processing Status */}
        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-2 shadow-2xs">
          <div className="flex items-center justify-between text-xs font-mono text-slate-600">
            <span className="font-bold text-slate-900 flex items-center gap-1.5">
              <FileCheck2 className="w-4 h-4 text-emerald-700" />
              <span>Processing Status</span>
            </span>
            <span className="px-1.5 py-0.5 rounded bg-slate-100 border border-slate-200 text-[10px]">
              {ragStatus?.totalDocuments || 0} Docs
            </span>
          </div>
          <div className="flex items-baseline gap-2 font-mono">
            <span className="text-2xl font-bold text-emerald-800">{auditSummary.indexedDocs}</span>
            <span className="text-xs text-slate-600">Indexed</span>
            <span className="text-sm font-bold text-amber-700 ml-2">{auditSummary.degradedDocs}</span>
            <span className="text-[11px] text-slate-500">Degraded</span>
            <span className="text-sm font-bold text-rose-700 ml-2">{auditSummary.failedDocs}</span>
            <span className="text-[11px] text-slate-500">Failed</span>
          </div>
          <div className="text-[11px] font-mono text-slate-500 pt-1 border-t border-slate-200 flex justify-between">
            <span>Vectors: {ragStatus?.totalVectors || 0}</span>
            <span>Dim: {ragStatus?.embeddingDim ? `${ragStatus.embeddingDim}d` : 'BM25-Only'}</span>
          </div>
        </div>

        {/* Card 2: Token Usage Statistics */}
        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-2 shadow-2xs">
          <div className="flex items-center justify-between text-xs font-mono text-slate-600">
            <span className="font-bold text-slate-900 flex items-center gap-1.5">
              <Cpu className="w-4 h-4 text-blue-900" />
              <span>Token Usage Stats</span>
            </span>
            <span className="px-1.5 py-0.5 rounded bg-blue-50 border border-blue-200 text-blue-900 text-[10px]">
              +{overlapOverheadPct}% overlap
            </span>
          </div>
          <div className="flex items-baseline gap-2 font-mono">
            <span className="text-2xl font-bold text-blue-900">
              {tokenTotals.totalChunkTokens.toLocaleString()}
            </span>
            <span className="text-xs text-slate-600">chunk tokens</span>
          </div>
          <div className="text-[11px] font-mono text-slate-500 pt-1 border-t border-slate-200 flex justify-between">
            <span>Src: {tokenTotals.totalSourceTokens.toLocaleString()}</span>
            <span>Query Embed: {tokenTotals.queryEmbeddingTokens.toLocaleString()}</span>
          </div>
        </div>

        {/* Card 3: Ingestion Quality Score */}
        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-2 shadow-2xs">
          <div className="flex items-center justify-between text-xs font-mono text-slate-600">
            <span className="font-bold text-slate-900 flex items-center gap-1.5">
              <Gauge className="w-4 h-4 text-indigo-700" />
              <span>Ingestion Quality</span>
            </span>
            <span
              className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${
                auditSummary.avgQualityScore >= 85
                  ? 'bg-emerald-50 text-emerald-900 border-emerald-300'
                  : auditSummary.avgQualityScore >= 70
                    ? 'bg-blue-50 text-blue-900 border-blue-300'
                    : 'bg-amber-50 text-amber-900 border-amber-300'
              }`}
            >
              {docs.length === 0
                ? 'N/A'
                : auditSummary.avgQualityScore >= 85
                  ? 'Optimal'
                  : auditSummary.avgQualityScore >= 70
                    ? 'Good'
                    : 'Tune Splitter'}
            </span>
          </div>
          <div className="flex items-baseline gap-2 font-mono">
            <span className="text-2xl font-bold text-indigo-900">
              {docs.length > 0 ? `${auditSummary.avgQualityScore}/100` : '—'}
            </span>
            <span className="text-xs text-slate-600">corpus health</span>
          </div>
          <div className="text-[11px] font-mono text-slate-500 pt-1 border-t border-slate-200 flex justify-between">
            <span>Orphans (&lt;120c): {auditSummary.totalOrphanChunks}</span>
            <span>Vocab: {bm25Stats.totalUniqueTerms.toLocaleString()} terms</span>
          </div>
        </div>

        {/* Card 4: Hybrid BM25 + Dense Telemetry */}
        <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-2 shadow-2xs">
          <div className="flex items-center justify-between text-xs font-mono text-slate-600">
            <span className="font-bold text-slate-900 flex items-center gap-1.5">
              <Activity className="w-4 h-4 text-rose-800" />
              <span>Hybrid Search Telemetry</span>
            </span>
            <span className="px-1.5 py-0.5 rounded bg-rose-50 border border-rose-200 text-rose-900 text-[10px]">
              RRF k=60
            </span>
          </div>
          <div className="flex items-baseline gap-2 font-mono">
            <span className="text-2xl font-bold text-rose-900">
              {auditSummary.totalQueriesExecuted}
            </span>
            <span className="text-xs text-slate-600">queries audited</span>
          </div>
          <div className="text-[11px] font-mono text-slate-500 pt-1 border-t border-slate-200 flex justify-between">
            <span>Avg Query: {auditSummary.avgQueryLatencyMs}ms</span>
            <span>BM25 k1={bm25Stats.k1} b={bm25Stats.b}</span>
          </div>
        </div>
      </div>

      {docs.length === 0 && auditEvents.length === 0 ? (
        <div className="bg-white border border-slate-300 rounded-xl p-8 text-center space-y-3">
          <Database className="w-8 h-8 text-slate-400 mx-auto" />
          <div className="space-y-1">
            <h4 className="text-sm font-bold text-slate-900">
              RAG Audit Telemetry Waiting for First Document Ingestion
            </h4>
            <p className="text-xs text-slate-600 max-w-lg mx-auto">
              Ingest a research PDF URL or raw paper excerpt in the Ingest tab to inspect real-time processing status, token consumption breakdown, and ingestion quality scoring.
            </p>
          </div>
          <button
            type="button"
            onClick={onSwitchToIngest}
            className="px-4 py-2 rounded-xl bg-blue-900 hover:bg-blue-800 text-white text-xs font-semibold transition-colors"
          >
            Go to Document Ingestion
          </button>
        </div>
      ) : (
        <>
          {/* Per-Document Processing Status, Token Stats & Quality Table */}
          {docs.length > 0 && (
            <div className="bg-white border border-slate-300 rounded-xl overflow-hidden shadow-2xs">
              <div className="px-4 py-3 bg-slate-100 border-b border-slate-300 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-xs font-bold text-slate-900">
                  <ShieldCheck className="w-4 h-4 text-blue-900" />
                  <span>Indexed RAG Document Audit Matrix ({docs.length} Documents)</span>
                </div>
                <span className="text-[11px] font-mono text-slate-600">
                  Stages: Extract → Split → Dense Vector → BM25 Sparse
                </span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 font-mono text-[11px] text-slate-700">
                      <th className="py-2.5 px-3">Document &amp; Pipeline Stages</th>
                      <th className="py-2.5 px-3">Status</th>
                      <th className="py-2.5 px-3">Token Usage (Src / Chunks / Overlap)</th>
                      <th className="py-2.5 px-3">Ingestion Quality</th>
                      <th className="py-2.5 px-3">Chunk &amp; Cleanliness</th>
                      <th className="py-2.5 px-3">Latency</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {docs.map((doc) => {
                      const q = doc.qualityMetrics;
                      const t = doc.tokenStats;
                      const st = doc.stageStatus;
                      return (
                        <tr key={doc.id} className="hover:bg-slate-50/90 align-top">
                          <td className="py-3 px-3 space-y-1.5 max-w-[260px]">
                            <div className="font-bold text-slate-900 truncate" title={doc.title}>
                              {doc.title}
                            </div>
                            <div className="flex flex-wrap items-center gap-1 font-mono text-[10px]">
                              <span
                                className={`px-1.5 py-0.5 rounded border ${
                                  st?.extract === 'ok'
                                    ? 'bg-emerald-50 text-emerald-900 border-emerald-200'
                                    : 'bg-rose-50 text-rose-900 border-rose-200'
                                }`}
                              >
                                Extract:{st?.extract || 'ok'}
                              </span>
                              <span
                                className={`px-1.5 py-0.5 rounded border ${
                                  st?.split === 'ok'
                                    ? 'bg-emerald-50 text-emerald-900 border-emerald-200'
                                    : 'bg-rose-50 text-rose-900 border-rose-200'
                                }`}
                              >
                                Split:{st?.split || 'ok'}
                              </span>
                              <span
                                className={`px-1.5 py-0.5 rounded border ${
                                  st?.denseEmbed === 'ok'
                                    ? 'bg-emerald-50 text-emerald-900 border-emerald-200'
                                    : 'bg-amber-50 text-amber-900 border-amber-200'
                                }`}
                              >
                                Dense:{st?.denseEmbed || 'ok'}
                              </span>
                              <span className="px-1.5 py-0.5 rounded border bg-emerald-50 text-emerald-900 border-emerald-200">
                                BM25:{st?.bm25Index || 'ok'}
                              </span>
                            </div>
                            {doc.statusNote && (
                              <div className="text-[10px] text-amber-800 font-mono leading-tight">
                                {doc.statusNote}
                              </div>
                            )}
                          </td>

                          <td className="py-3 px-3 font-mono">
                            {doc.status === 'indexed' ? (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-100 text-emerald-900 border border-emerald-300 text-[10px] font-bold">
                                <CheckCircle2 className="w-3 h-3" />
                                <span>INDEXED</span>
                              </span>
                            ) : doc.status === 'degraded' ? (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-amber-100 text-amber-900 border border-amber-300 text-[10px] font-bold">
                                <AlertTriangle className="w-3 h-3" />
                                <span>BM25-ONLY</span>
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-rose-100 text-rose-900 border border-rose-300 text-[10px] font-bold">
                                <XCircle className="w-3 h-3" />
                                <span>FAILED</span>
                              </span>
                            )}
                            <div className="text-[10px] text-slate-500 mt-1">
                              {doc.chunkCount} chunks · {doc.embeddingDim > 0 ? `${doc.embeddingDim}d` : 'Sparse'}
                            </div>
                          </td>

                          <td className="py-3 px-3 font-mono">
                            <div className="font-bold text-blue-900">
                              {(t?.totalChunkTokens || 0).toLocaleString()} total
                            </div>
                            <div className="text-[10px] text-slate-600">
                              Src: {(t?.sourceTokens || 0).toLocaleString()} · +{(t?.overlapTokens || 0).toLocaleString()} overlap
                            </div>
                            <div className="text-[10px] text-slate-500">
                              Avg {t?.avgChunkTokens || 0} tok/chunk ({t?.minChunkTokens || 0}–{t?.maxChunkTokens || 0})
                            </div>
                          </td>

                          <td className="py-3 px-3 font-mono min-w-[140px]">
                            <div className="flex items-center justify-between text-[11px] font-bold">
                              <span
                                className={
                                  (q?.overallScore || 0) >= 85
                                    ? 'text-emerald-800'
                                    : (q?.overallScore || 0) >= 70
                                      ? 'text-blue-900'
                                      : 'text-amber-800'
                                }
                              >
                                {q?.overallScore || 0}/100
                              </span>
                              <span className="text-[10px] text-slate-600">{q?.verdict}</span>
                            </div>
                            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden mt-1">
                              <div
                                className={`h-full rounded-full ${
                                  (q?.overallScore || 0) >= 85
                                    ? 'bg-emerald-600'
                                    : (q?.overallScore || 0) >= 70
                                      ? 'bg-blue-700'
                                      : 'bg-amber-600'
                                }`}
                                style={{ width: `${q?.overallScore || 0}%` }}
                              />
                            </div>
                            <div className="text-[10px] text-slate-500 mt-1">
                              {q?.uniqueTerms || 0} BM25 terms
                            </div>
                          </td>

                          <td className="py-3 px-3 font-mono text-[10px] text-slate-700 space-y-0.5">
                            <div>Cleanliness: <strong className="text-slate-900">{q?.cleanlinessPct ?? 100}%</strong></div>
                            <div>Uniformity: <strong className="text-slate-900">{q?.chunkUniformityPct ?? 100}%</strong></div>
                            <div>
                              Orphans: <strong className={(q?.orphanChunks || 0) > 0 ? 'text-amber-800' : 'text-emerald-800'}>{q?.orphanChunks ?? 0}</strong> · Overlap {q?.overlapRatioPct ?? 0}%
                            </div>
                          </td>

                          <td className="py-3 px-3 font-mono text-[10px] text-slate-700 space-y-0.5">
                            <div className="font-bold text-slate-900">{doc.latencyMs?.totalMs || 0}ms</div>
                            <div>Ext {doc.latencyMs?.extractMs || 0}ms</div>
                            <div>Emb {doc.latencyMs?.embedMs || 0}ms · BM25 {doc.latencyMs?.bm25Ms || 0}ms</div>
                          </td>

                          <td className="py-3 px-3 text-right">
                            <button
                              type="button"
                              onClick={() => onDeleteDoc(doc.id)}
                              title="Remove document from RAG audit index"
                              className="p-1.5 rounded-lg bg-slate-100 hover:bg-rose-100 text-slate-600 hover:text-rose-900 border border-slate-200 transition-colors"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Bottom Split: Token Distribution Bar Chart + Chronological Audit Trail */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            {/* Token Usage Breakdown Bar Chart */}
            <div className="lg:col-span-6 bg-white border border-slate-300 rounded-xl p-4 space-y-3 shadow-2xs">
              <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                  <Layers className="w-4 h-4 text-blue-900" />
                  <span>Token Allocation: Unique Source vs Overlap Overhead</span>
                </span>
                <div className="flex items-center gap-3 text-[10px] font-mono">
                  <span className="text-blue-900 font-semibold">■ Source Tokens</span>
                  <span className="text-amber-700 font-semibold">■ Overlap Tokens</span>
                </div>
              </div>

              {docs.length === 0 ? (
                <div className="py-8 text-center text-xs text-slate-500">No indexed documents to chart yet.</div>
              ) : (
                <div className="space-y-3 max-h-56 overflow-y-auto pr-1">
                  {docs.map((doc) => {
                    const src = doc.tokenStats?.sourceTokens || 0;
                    const ov = doc.tokenStats?.overlapTokens || 0;
                    const total = Math.max(1, doc.tokenStats?.totalChunkTokens || src + ov);
                    const srcW = (src / maxDocTokens) * 100;
                    const ovW = (ov / maxDocTokens) * 100;
                    return (
                      <div key={doc.id} className="space-y-1">
                        <div className="flex items-center justify-between text-[11px] font-mono">
                          <span className="font-semibold text-slate-800 truncate max-w-[230px]" title={doc.title}>
                            {doc.title}
                          </span>
                          <span className="text-slate-600 shrink-0">
                            {src.toLocaleString()} + <span className="text-amber-800">{ov.toLocaleString()}</span> ={' '}
                            <strong className="text-blue-900">{total.toLocaleString()}</strong> tok
                          </span>
                        </div>
                        <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden border border-slate-200 flex">
                          <div
                            className="h-full bg-blue-800"
                            style={{ width: `${Math.min(100, srcW)}%` }}
                            title={`Source Tokens: ${src}`}
                          />
                          <div
                            className="h-full bg-amber-500"
                            style={{ width: `${Math.max(0, Math.min(100 - srcW, ovW))}%` }}
                            title={`Chunk Overlap Tokens: ${ov}`}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Ingestion Processing Audit Trail */}
            <div className="lg:col-span-6 bg-white border border-slate-300 rounded-xl p-4 space-y-3 shadow-2xs">
              <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                  <Clock className="w-4 h-4 text-emerald-800" />
                  <span>Recent Ingestion Processing Audit Log</span>
                </span>
                <span className="text-[10px] font-mono text-slate-500">
                  {auditEvents.length} events logged
                </span>
              </div>

              {auditEvents.length === 0 ? (
                <div className="py-8 text-center text-xs text-slate-500">No ingestion events logged yet.</div>
              ) : (
                <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                  {auditEvents.map((evt) => (
                    <div
                      key={evt.id}
                      className={`p-2.5 rounded-lg border text-[11px] space-y-1 ${
                        evt.status === 'indexed'
                          ? 'bg-emerald-50/50 border-emerald-200'
                          : evt.status === 'degraded'
                            ? 'bg-amber-50/60 border-amber-200'
                            : 'bg-rose-50/60 border-rose-200'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 font-mono">
                        <span className="font-bold text-slate-900 truncate">{evt.docTitle}</span>
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-bold shrink-0 ${
                            evt.status === 'indexed'
                              ? 'bg-emerald-100 text-emerald-900'
                              : evt.status === 'degraded'
                                ? 'bg-amber-100 text-amber-900'
                                : 'bg-rose-100 text-rose-900'
                          }`}
                        >
                          {evt.status.toUpperCase()}
                        </span>
                      </div>
                      <p className="text-slate-700 leading-snug">{evt.message}</p>
                      <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                        <span>
                          {evt.tokensProcessed.toLocaleString()} tok · {evt.totalMs}ms
                        </span>
                        <span>{new Date(evt.timestamp).toLocaleTimeString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
