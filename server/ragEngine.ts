import { GoogleGenAI } from '@google/genai';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { extractText } from 'unpdf';
import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import type {
  RagCitation,
  RagDocumentSummary,
  RagIngestResponse,
  RagStatusResponse,
} from '../src/types';

interface FaissVectorRecord {
  id: string;
  docId: string;
  docTitle: string;
  source: string;
  chunkIndex: number;
  content: string;
  vector: number[];
}

interface PersistedFaissStore {
  indexType: string;
  embeddingDim: number;
  documents: RagDocumentSummary[];
  vectors: FaissVectorRecord[];
}

const STORE_PATH = path.resolve(process.cwd(), '.rag_faiss_store.json');
const MAX_CHUNKS_PER_DOC = 120;
const EMBED_BATCH_SIZE = 16;

function l2Normalize(vec: number[]): number[] {
  let normSq = 0;
  for (let i = 0; i < vec.length; i++) {
    normSq += vec[i] * vec[i];
  }
  const norm = Math.sqrt(normSq);
  if (norm === 0) return vec;
  const out = new Array<number>(vec.length);
  for (let i = 0; i < vec.length; i++) {
    out[i] = vec[i] / norm;
  }
  return out;
}

function innerProduct(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

function stripHtmlToText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<\/(p|div|section|article|li|h[1-6]|tr)>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

function deriveTitleFromUrl(urlStr: string): string {
  try {
    const parsed = new URL(urlStr);
    const segments = parsed.pathname.split('/').filter(Boolean);
    const last = segments[segments.length - 1] || parsed.hostname;
    return decodeURIComponent(last).replace(/\.pdf$/i, '');
  } catch {
    return 'Imported PDF Document';
  }
}

class FaissRagEngine {
  private store: PersistedFaissStore = {
    indexType: 'FAISS IndexFlatIP (L2-Normalized Cosine)',
    embeddingDim: 0,
    documents: [],
    vectors: [],
  };

  constructor() {
    this.loadFromDisk();
  }

  private loadFromDisk() {
    try {
      if (fs.existsSync(STORE_PATH)) {
        const raw = fs.readFileSync(STORE_PATH, 'utf-8');
        const parsed = JSON.parse(raw) as PersistedFaissStore;
        if (Array.isArray(parsed.documents) && Array.isArray(parsed.vectors)) {
          this.store = parsed;
        }
      }
    } catch (err) {
      console.warn('Could not load persisted FAISS store, starting fresh:', err);
    }
  }

  private saveToDisk() {
    try {
      fs.writeFileSync(STORE_PATH, JSON.stringify(this.store), 'utf-8');
    } catch (err) {
      console.warn('Could not persist FAISS store to disk:', err);
    }
  }

  public getStatus(): RagStatusResponse {
    return {
      indexType: this.store.indexType,
      embeddingModel: 'gemini-embedding-2',
      splitter: 'LangChain RecursiveCharacterTextSplitter',
      totalDocuments: this.store.documents.length,
      totalVectors: this.store.vectors.length,
      embeddingDim: this.store.embeddingDim,
      documents: [...this.store.documents].reverse(),
    };
  }

  public deleteDocument(docId: string): boolean {
    const before = this.store.documents.length;
    this.store.documents = this.store.documents.filter((d) => d.id !== docId);
    this.store.vectors = this.store.vectors.filter((v) => v.docId !== docId);
    if (this.store.vectors.length === 0) {
      this.store.embeddingDim = 0;
    }
    this.saveToDisk();
    return this.store.documents.length < before;
  }

  public clearAll() {
    this.store.documents = [];
    this.store.vectors = [];
    this.store.embeddingDim = 0;
    this.saveToDisk();
  }

  private async extractFromUrl(pdfUrl: string): Promise<{ text: string; pageCount?: number }> {
    let parsedUrl: URL;
    try {
      parsedUrl = new URL(pdfUrl.trim());
    } catch {
      throw new Error(`Invalid URL format: "${pdfUrl}"`);
    }

    if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
      throw new Error('Only http:// and https:// document URLs are supported.');
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000);

    let res: Response;
    try {
      res = await fetch(parsedUrl.toString(), {
        signal: controller.signal,
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; MLStudioRAGIndexer/1.0)',
          Accept: 'application/pdf, text/html, text/plain, */*',
        },
      });
    } catch (err: any) {
      throw new Error(`Failed to fetch document URL (${err.message || 'network timeout'}).`);
    } finally {
      clearTimeout(timeoutId);
    }

    if (!res.ok) {
      throw new Error(`Remote server returned HTTP ${res.status} for ${parsedUrl.toString()}`);
    }

    const contentType = (res.headers.get('content-type') || '').toLowerCase();
    const arrayBuffer = await res.arrayBuffer();
    const bytes = new Uint8Array(arrayBuffer);

    const hasPdfMagic =
      bytes.length >= 4 &&
      bytes[0] === 0x25 && // %
      bytes[1] === 0x50 && // P
      bytes[2] === 0x44 && // D
      bytes[3] === 0x46; // F

    const looksLikePdf =
      contentType.includes('application/pdf') ||
      parsedUrl.pathname.toLowerCase().endsWith('.pdf') ||
      hasPdfMagic;

    if (looksLikePdf) {
      const extracted = await extractText(bytes, { mergePages: true });
      const cleaned = (extracted.text || '').replace(/\0/g, '').trim();
      if (!cleaned) {
        throw new Error(
          'PDF fetched successfully, but no extractable text layer was found (document may be scanned images).'
        );
      }
      return { text: cleaned, pageCount: extracted.totalPages };
    }

    const decoded = new TextDecoder('utf-8', { fatal: false }).decode(bytes);
    const cleaned = contentType.includes('text/html') ? stripHtmlToText(decoded) : decoded.trim();
    if (!cleaned) {
      throw new Error('Fetched URL contained no readable text content.');
    }
    return { text: cleaned };
  }

  public async ingestDocument(
    ai: GoogleGenAI,
    params: {
      mode: 'raw_text' | 'pdf_url';
      title?: string;
      rawText?: string;
      pdfUrl?: string;
      chunkSize?: number;
      chunkOverlap?: number;
    }
  ): Promise<RagIngestResponse> {
    if (!process.env.GEMINI_API_KEY) {
      throw new Error(
        'GEMINI_API_KEY is missing. Please configure it in Settings > Secrets before running embeddings.'
      );
    }

    const chunkSize = Math.min(2000, Math.max(200, Number(params.chunkSize) || 800));
    const chunkOverlap = Math.min(Math.floor(chunkSize / 2), Math.max(0, Number(params.chunkOverlap) || 120));

    // Step 1: Extract Text
    const tExtractStart = performance.now();
    let fullText = '';
    let pageCount: number | undefined;
    let sourceLabel = '';
    let title = (params.title || '').trim();

    if (params.mode === 'pdf_url') {
      const url = (params.pdfUrl || '').trim();
      if (!url) {
        throw new Error('Please provide a valid PDF or document URL.');
      }
      const extracted = await this.extractFromUrl(url);
      fullText = extracted.text;
      pageCount = extracted.pageCount;
      sourceLabel = url;
      if (!title) {
        title = deriveTitleFromUrl(url);
      }
    } else {
      fullText = (params.rawText || '').trim();
      if (fullText.length < 20) {
        throw new Error('Please paste at least 20 characters of text to chunk and index.');
      }
      sourceLabel = 'Raw Text Paste';
      if (!title) {
        title = `Pasted Research Notes (${new Date().toLocaleTimeString()})`;
      }
    }
    const extractMs = Math.round(performance.now() - tExtractStart);

    // Step 2: LangChain RecursiveCharacterTextSplitter
    const tSplitStart = performance.now();
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize,
      chunkOverlap,
      separators: ['\n\n', '\n', '. ', ' ', ''],
    });
    const lcDocs = await splitter.createDocuments([fullText], [{ source: sourceLabel, title }]);
    const boundedDocs = lcDocs.slice(0, MAX_CHUNKS_PER_DOC);
    if (boundedDocs.length === 0) {
      throw new Error('LangChain text splitter produced zero chunks from the supplied document.');
    }
    const splitMs = Math.round(performance.now() - tSplitStart);

    // Step 3: Dense Embeddings via Gemini (@google/genai)
    const tEmbedStart = performance.now();
    const chunkEmbeddings: number[][] = [];
    for (let i = 0; i < boundedDocs.length; i += EMBED_BATCH_SIZE) {
      const batch = boundedDocs.slice(i, i + EMBED_BATCH_SIZE).map((d) => d.pageContent);
      const embedRes = await ai.models.embedContent({
        model: 'gemini-embedding-2',
        contents: batch,
      });
      const vectors = embedRes.embeddings?.map((e) => e.values || []) || [];
      if (vectors.length !== batch.length || vectors.some((v) => v.length === 0)) {
        throw new Error('Embedding service returned incomplete vector dimensions.');
      }
      for (const vec of vectors) {
        chunkEmbeddings.push(l2Normalize(vec));
      }
    }
    const embedMs = Math.round(performance.now() - tEmbedStart);

    // Step 4: Upsert L2-Normalized Vectors into FAISS IndexFlatIP Store
    const tIndexStart = performance.now();
    const docId = crypto.randomUUID().slice(0, 10);
    const embeddingDim = chunkEmbeddings[0].length;

    // Reset mismatched dimensions if index previously empty or schema changed
    if (this.store.embeddingDim !== 0 && this.store.embeddingDim !== embeddingDim) {
      this.store.vectors = [];
      this.store.documents = [];
    }
    this.store.embeddingDim = embeddingDim;

    const records: FaissVectorRecord[] = boundedDocs.map((doc, idx) => ({
      id: `${docId}_c${idx}`,
      docId,
      docTitle: title,
      source: sourceLabel,
      chunkIndex: idx,
      content: doc.pageContent,
      vector: chunkEmbeddings[idx],
    }));

    const summary: RagDocumentSummary = {
      id: docId,
      title,
      sourceType: params.mode,
      sourceUrl: params.mode === 'pdf_url' ? sourceLabel : undefined,
      pageCount,
      charCount: fullText.length,
      chunkCount: records.length,
      embeddingDim,
      indexedAt: new Date().toISOString(),
      preview: boundedDocs[0].pageContent.slice(0, 220),
    };

    this.store.documents.push(summary);
    this.store.vectors.push(...records);
    this.saveToDisk();
    const indexMs = Math.max(1, Math.round(performance.now() - tIndexStart));

    return {
      document: summary,
      chunksAdded: records.length,
      totalVectors: this.store.vectors.length,
      workflowMetrics: {
        extractMs,
        splitMs,
        embedMs,
        indexMs,
        embeddingDim,
      },
    };
  }

  public async search(ai: GoogleGenAI, query: string, topK: number = 4): Promise<RagCitation[]> {
    if (this.store.vectors.length === 0) {
      return [];
    }
    if (!process.env.GEMINI_API_KEY) {
      return [];
    }

    const embedRes = await ai.models.embedContent({
      model: 'gemini-embedding-2',
      contents: [query],
    });
    const rawQueryVec = embedRes.embeddings?.[0]?.values;
    if (!rawQueryVec || rawQueryVec.length === 0) {
      return [];
    }

    const queryVec = l2Normalize(rawQueryVec);
    const scored = this.store.vectors
      .filter((rec) => rec.vector.length === queryVec.length)
      .map((rec) => ({
        rec,
        score: innerProduct(queryVec, rec.vector),
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return scored.map(({ rec, score }) => ({
      docId: rec.docId,
      docTitle: rec.docTitle,
      source: rec.source,
      chunkIndex: rec.chunkIndex,
      score: Number(score.toFixed(4)),
      snippet: rec.content,
    }));
  }
}

export const ragEngine = new FaissRagEngine();
