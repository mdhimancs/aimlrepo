import express from 'express';
import { GoogleGenAI } from '@google/genai';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import { ragEngine } from './server/ragEngine';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '5mb' }));

  // Initialize Gemini client server-side
  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY || '',
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });

  // RAG Vector Store Status
  app.get('/api/rag/status', (_req, res) => {
    res.json(ragEngine.getStatus());
  });

  // LangChain + FAISS Ingestion Endpoint (Raw Text or PDF/Web URL)
  app.post('/api/rag/ingest', async (req, res) => {
    try {
      const { mode, title, rawText, pdfUrl, chunkSize, chunkOverlap } = req.body || {};
      const result = await ragEngine.ingestDocument(ai, {
        mode: mode === 'pdf_url' ? 'pdf_url' : 'raw_text',
        title,
        rawText,
        pdfUrl,
        chunkSize,
        chunkOverlap,
      });
      res.json(result);
    } catch (err: any) {
      console.error('RAG ingest error:', err);
      res.status(400).json({ error: err.message || 'Failed to chunk and index document into FAISS.' });
    }
  });

  // Delete single indexed document from FAISS store
  app.delete('/api/rag/documents/:id', (req, res) => {
    const deleted = ragEngine.deleteDocument(req.params.id);
    if (!deleted) {
      return res.status(404).json({ error: 'Document not found in FAISS store.' });
    }
    return res.json(ragEngine.getStatus());
  });

  // Clear entire FAISS + BM25 store
  app.post('/api/rag/clear', (_req, res) => {
    ragEngine.clearAll();
    res.json(ragEngine.getStatus());
  });

  // Hybrid Search (Dense Vector + BM25 Sparse RRF) Inspection Endpoint
  app.post('/api/rag/search', async (req, res) => {
    try {
      const { query, topK, mode, alpha } = req.body || {};
      const q = String(query || '').trim();
      if (!q) {
        return res.status(400).json({ error: 'Search query cannot be empty.' });
      }
      const searchMode = mode === 'dense' || mode === 'bm25' ? mode : 'hybrid';
      const result = await ragEngine.hybridSearch(ai, q, {
        topK: Number(topK) || 5,
        mode: searchMode,
        alpha: alpha !== undefined ? Number(alpha) : 0.55,
      });
      return res.json(result);
    } catch (err: any) {
      console.error('RAG hybrid search error:', err);
      return res.status(500).json({ error: err.message || 'Hybrid search execution failed.' });
    }
  });

  // API endpoint to generate custom AI/ML project blueprint & code
  app.post('/api/generate-project', async (req, res) => {
    try {
      const { projectName, domain, objective, dataType, targetMetric, deploymentTarget } = req.body;
      
      const prompt = `You are a Principal AI/ML Architect. Create a comprehensive, production-grade project blueprint and code architecture for a new AI/ML project with the following specifications:
- Project Name: ${projectName || 'Custom AI Project'}
- Domain: ${domain || 'Natural Language Processing'}
- Objective: ${objective || 'Build an intelligent document extraction and QA pipeline'}
- Data Type: ${dataType || 'Text Corpus & PDF'}
- Target Metric: ${targetMetric || 'F1-Score > 0.92'}
- Deployment Target: ${deploymentTarget || 'Cloud Run API'}

Provide a detailed response in strict JSON format with the following structure:
{
  "overview": "Detailed 2-sentence summary of the architecture and approach.",
  "phases": [
    {
      "stepNumber": 1,
      "title": "Data Engineering & Pipeline",
      "description": "Description of data collection, cleaning, tokenization, or feature scaling.",
      "pythonSnippet": "Python code snippet for data loading/preprocessing"
    },
    {
      "stepNumber": 2,
      "title": "Model Architecture & Design",
      "description": "Choice of model, loss functions, optimizer, and layers.",
      "pythonSnippet": "Python code snippet defining model architecture"
    },
    {
      "stepNumber": 3,
      "title": "Training & Optimization Loop",
      "description": "Hyperparameters, batch size, learning rate scheduler, epochs.",
      "pythonSnippet": "Python code snippet for training loop"
    },
    {
      "stepNumber": 4,
      "title": "Evaluation & Validation",
      "description": "Validation split, metrics calculation, confusion matrix or benchmark results.",
      "pythonSnippet": "Python code snippet for evaluation"
    },
    {
      "stepNumber": 5,
      "title": "Deployment & MLOps",
      "description": "Model serialization, FastAPI wrapper, Dockerfile, and monitoring.",
      "pythonSnippet": "Python code snippet for FastAPI serving endpoint"
    }
  ],
  "recommendedStack": ["PyTorch", "Hugging Face", "FastAPI", "Docker", "Weights & Biases"],
  "estimatedTimeWeeks": 3
}`;

      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          systemInstruction: 'You are an elite Principal AI Architect specializing in production ML systems from scratch. Respond strictly in valid JSON matching the requested schema.',
        },
      });

      const text = response.text || '{}';
      const json = JSON.parse(text);
      res.json(json);
    } catch (err: any) {
      console.error('Error generating project blueprint:', err);
      res.status(500).json({ error: err.message || 'Failed to generate project blueprint' });
    }
  });

  // API endpoint for Hybrid-RAG-Enabled AI/ML Expert Copilot Chat
  app.post('/api/ai-chat', async (req, res) => {
    try {
      const { message, useRag = true, searchMode = 'hybrid', alpha = 0.55 } = req.body || {};
      const queryText = String(message || 'Hello');

      const status = ragEngine.getStatus();
      const validMode = searchMode === 'dense' || searchMode === 'bm25' ? searchMode : 'hybrid';
      const searchResult =
        useRag && status.totalVectors > 0
          ? await ragEngine.hybridSearch(ai, queryText, {
              topK: 4,
              mode: validMode,
              alpha: Number(alpha),
            })
          : null;
      const citations = searchResult ? searchResult.results : [];

      const contextBlock =
        citations.length > 0
          ? `\n\nRETRIEVED HYBRID RAG CONTEXT (${validMode.toUpperCase()} top-${citations.length} matches):\n` +
            citations
              .map(
                (c, i) =>
                  `[Citation ${i + 1} | ${c.docTitle} · Chunk #${c.chunkIndex} · Hybrid=${c.score} · Dense=${c.denseScore} · BM25=${c.bm25Score}]\n${c.snippet}`
              )
              .join('\n\n')
          : '';

      const chat = ai.chats.create({
        model: 'gemini-4-hydrogen',
        config: {
          systemInstruction:
            'You are an expert AI/ML Engineering & Research RAG Copilot. You help developers build, train, debug, and analyze ML papers and architectures. When retrieved Hybrid RAG (Dense + BM25) context chunks are provided, prioritize and synthesize facts from those chunks and cite them explicitly as [Citation 1], [Citation 2], etc. Be concise, technical, and precise.',
        },
      });

      const promptWithRag = citations.length > 0
        ? `User Question:\n${queryText}${contextBlock}\n\nAnswer the user question grounded in the retrieved context where relevant, and cite [Citation N].`
        : queryText;

      const response = await chat.sendMessage({
        message: promptWithRag,
      });

      res.json({
        reply: response.text,
        citations,
        searchLatencyMs: searchResult?.latencyMs,
        indexedVectorCount: status.totalVectors,
      });
    } catch (err: any) {
      console.error('Error in AI chat:', err);
      res.status(500).json({ error: err.message || 'AI chat failed' });
    }
  });

  // Vite middleware for development
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: 'spa',
  });
  app.use(vite.middlewares);

  const PORT = Number(process.env.PORT) || 3000;
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server & Vite dev middleware running on port ${PORT}`);
  });
}

startServer();
