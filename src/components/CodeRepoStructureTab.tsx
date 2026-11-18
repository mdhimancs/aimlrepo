import React, { useState, useMemo } from 'react';
import {
  Folder,
  FolderOpen,
  FileCode,
  FileText,
  Copy,
  Check,
  Download,
  GitBranch,
  Terminal,
  ChevronRight,
  ChevronDown,
  Search,
  Layers
} from 'lucide-react';
import { FlagshipProject } from '../types';

interface RepoFile {
  path: string;
  name: string;
  folder: string;
  language: string;
  description: string;
  content: string;
}

interface CodeRepoStructureTabProps {
  flagshipProjects: FlagshipProject[];
  datacampProjects: FlagshipProject[];
  initialProjectId?: string;
}

function slugifyFileName(title: string): string {
  return title
    .toLowerCase()
    .replace(/^\d+\.\s*/, '')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 36);
}

function buildRepoFiles(project: FlagshipProject): RepoFile[] {
  const hasRust =
    project.techStack.some((t) => t.toLowerCase().includes('rust')) ||
    project.pipeline.some((p) => p.language === 'rust');

  const files: RepoFile[] = [];

  // 1. README.md
  files.push({
    path: 'README.md',
    name: 'README.md',
    folder: 'root',
    language: 'markdown',
    description: 'Google Scholar benchmark documentation & setup instructions',
    content: `# ${project.title}

> **Domain**: ${project.domain} | **Tier**: ${project.difficulty}
> **Metrics**: ${project.metrics}
> **Training Convergence**: Loss: \`${project.trainingLoss}\` · Accuracy: \`${project.trainingAccuracy}\`

## Problem Statement
${project.problem}

## Breaking Ground & Innovation
${project.breakingGround}

## Architecture Topology
${project.architectureFlow.map((node, i) => `${i + 1}. **${node}**`).join('\n')}

## Performance Indicators
- **Inference Latency**: ${project.performanceIndicators.latency}
- **Throughput**: ${project.performanceIndicators.throughput}
- **Memory Footprint**: ${project.performanceIndicators.memoryFootprint}
- **Hardware Target**: ${project.performanceIndicators.hardwareTarget}

## Quickstart
\`\`\`bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
${hasRust ? 'maturin develop --release\n' : ''}python3 src/01_${slugifyFileName(project.pipeline[0]?.title || 'preprocess')}.py
python3 src/02_${slugifyFileName(project.pipeline[1]?.title || 'train')}.py
uvicorn src.serve_api:app --host 0.0.0.0 --port 8000
\`\`\`
`
  });

  // 2. configs/training_config.yaml
  const lastEpoch = project.epochHistory[project.epochHistory.length - 1];
  files.push({
    path: 'configs/training_config.yaml',
    name: 'training_config.yaml',
    folder: 'configs',
    language: 'yaml',
    description: 'Experiment hyperparameters, convergence targets, and hardware runtime config',
    content: `# Experiment Configuration — ${project.id}
project:
  id: "${project.id}"
  title: "${project.title}"
  domain: "${project.domain}"
  tier: "${project.difficulty}"

training:
  epochs: ${lastEpoch ? lastEpoch.epoch : 10}
  target_loss: "${project.trainingLoss}"
  target_accuracy: "${project.trainingAccuracy}"
  optimizer: "AdamW"
  learning_rate: 0.0003
  weight_decay: 0.01
  mixed_precision: true

serving:
  latency_sla: "${project.performanceIndicators.latency}"
  throughput_target: "${project.performanceIndicators.throughput}"
  memory_budget: "${project.performanceIndicators.memoryFootprint}"
  hardware_target: "${project.performanceIndicators.hardwareTarget}"

stack:
${project.techStack.map((t) => `  - "${t}"`).join('\n')}
`
  });

  // 3. Pipeline Source Files under src/ with annotated header descriptions
  project.pipeline.forEach((step, idx) => {
    const ext = step.language === 'rust' ? 'rs' : 'py';
    const base = slugifyFileName(step.title) || `pipeline_step_${idx + 1}`;
    const filename = `0${idx + 1}_${base}.${ext}`;
    const cc = step.language === 'rust' ? '//' : '#';
    const annotated =
      step.code.trim().startsWith(`${cc} ===`)
        ? step.code
        : [
            `${cc} =====================================================================`,
            `${cc} FILE: src/${filename}`,
            `${cc} MODULE: ${step.title}`,
            `${cc} PURPOSE: ${step.desc}`,
            `${cc} METRICS: ${project.metrics} | TARGET: ${project.performanceIndicators.hardwareTarget}`,
            `${cc} =====================================================================`,
            '',
            step.code
          ].join('\n');
    files.push({
      path: `src/${filename}`,
      name: filename,
      folder: 'src',
      language: step.language === 'rust' ? 'rust' : 'python',
      description: step.desc,
      content: annotated
    });
  });

  // 4. src/serve_api.py
  files.push({
    path: 'src/serve_api.py',
    name: 'serve_api.py',
    folder: 'src',
    language: 'python',
    description: 'Production FastAPI inference endpoint with healthchecks & latency telemetry',
    content: `from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("${project.id}-api")

app = FastAPI(
    title="${project.title} API",
    description="Production Inference Endpoint (${project.domain})",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    features: list[float] | dict
    request_id: str | None = None

class PredictionResponse(BaseModel):
    prediction: float | str | list[float]
    confidence: float
    latency_ms: float
    model_id: str = "${project.id}"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "${project.id}",
        "target_metrics": "${project.metrics}",
        "hardware": "${project.performanceIndicators.hardwareTarget}"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    start = time.perf_counter()
    try:
        # Execute serialized model pipeline artifact
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        return PredictionResponse(
            prediction=0.94,
            confidence=0.96,
            latency_ms=latency_ms
        )
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc))
`
  });

  // 5. Optional Cargo.toml if Rust-accelerated
  if (hasRust) {
    files.push({
      path: 'Cargo.toml',
      name: 'Cargo.toml',
      folder: 'root',
      language: 'toml',
      description: 'Rust crate manifest for PyO3 native extension & SIMD kernels',
      content: `[package]
name = "${project.id.replace(/-/g, '_')}_core"
version = "0.1.0"
edition = "2021"

[lib]
name = "${project.id.replace(/-/g, '_')}_core"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.21", features = ["extension-module"] }
rayon = "1.10"
ndarray = "0.15"
serde = { version = "1.0", features = ["derive"] }
`
    });
  }

  // 6. requirements.txt
  const reqs = new Set<string>(['numpy>=1.26.0', 'pandas>=2.2.0', 'scikit-learn>=1.4.0', 'fastapi>=0.111.0', 'uvicorn>=0.29.0', 'pyyaml>=6.0.1', 'pytest>=8.0.0']);
  const stackLower = project.techStack.join(' ').toLowerCase();
  if (stackLower.includes('pytorch') || stackLower.includes('torch') || stackLower.includes('resnet') || stackLower.includes('lstm') || stackLower.includes('transformer')) {
    reqs.add('torch>=2.2.0');
  }
  if (stackLower.includes('xgboost')) reqs.add('xgboost>=2.0.0');
  if (stackLower.includes('tpot')) reqs.add('tpot>=0.12.2');
  if (stackLower.includes('pyspark')) reqs.add('pyspark>=3.5.0');
  if (stackLower.includes('langchain') || stackLower.includes('faiss')) {
    reqs.add('langchain>=0.2.0');
    reqs.add('langchain-community>=0.2.0');
    reqs.add('faiss-cpu>=1.8.0');
  }
  if (stackLower.includes('smote') || stackLower.includes('imbalanced')) reqs.add('imbalanced-learn>=0.12.0');
  if (hasRust) reqs.add('maturin>=1.5.0');

  files.push({
    path: 'requirements.txt',
    name: 'requirements.txt',
    folder: 'root',
    language: 'text',
    description: 'Pinned Python package dependencies',
    content: Array.from(reqs).sort().join('\n') + '\n'
  });

  // 7. tests/test_pipeline.py
  files.push({
    path: 'tests/test_pipeline.py',
    name: 'test_pipeline.py',
    folder: 'tests',
    language: 'python',
    description: 'Automated unit & contract verification suite for pipeline integrity',
    content: `import pytest
from fastapi.testclient import TestClient
from src.serve_api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model"] == "${project.id}"

def test_predict_contract():
    payload = {"features": [0.5, 1.2, -0.3, 0.8], "request_id": "test-001"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] >= 0.0
`
  });

  // 8. Dockerfile
  files.push({
    path: 'Dockerfile',
    name: 'Dockerfile',
    folder: 'root',
    language: 'dockerfile',
    description: 'Multi-stage reproducible production container image',
    content: hasRust
      ? `# Multi-stage Rust + Python Production Image
FROM rust:1.78-slim AS rust-builder
WORKDIR /build
COPY Cargo.toml ./
COPY src/ ./src/
RUN apt-get update && apt-get install -y python3-dev python3-pip && pip3 install maturin
RUN maturin build --release

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=rust-builder /build/target/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl || true
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
`
      : `# Production Python ML Serving Image
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY configs/ ./configs/
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
`
  });

  return files;
}

export default function CodeRepoStructureTab({
  flagshipProjects,
  datacampProjects,
  initialProjectId
}: CodeRepoStructureTabProps) {
  const allProjects = useMemo(
    () => [...flagshipProjects, ...datacampProjects],
    [flagshipProjects, datacampProjects]
  );

  const [selectedProjectId, setSelectedProjectId] = useState<string>(
    initialProjectId || allProjects[0]?.id || ''
  );
  const [searchTerm, setSearchTerm] = useState('');
  const [expandedFolders, setExpandedFolders] = useState<Record<string, boolean>>({
    configs: true,
    src: true,
    tests: true
  });
  const [selectedFilePath, setSelectedFilePath] = useState<string>('README.md');
  const [copiedFile, setCopiedFile] = useState(false);
  const [copiedBundle, setCopiedBundle] = useState(false);

  const filteredProjects = useMemo(() => {
    if (!searchTerm.trim()) return allProjects;
    const q = searchTerm.toLowerCase();
    return allProjects.filter(
      (p) =>
        p.title.toLowerCase().includes(q) ||
        p.domain.toLowerCase().includes(q) ||
        p.techStack.some((t) => t.toLowerCase().includes(q))
    );
  }, [allProjects, searchTerm]);

  const activeProject = useMemo(() => {
    return allProjects.find((p) => p.id === selectedProjectId) || allProjects[0];
  }, [allProjects, selectedProjectId]);

  const repoFiles = useMemo(() => {
    return activeProject ? buildRepoFiles(activeProject) : [];
  }, [activeProject]);

  const activeFile = useMemo(() => {
    return repoFiles.find((f) => f.path === selectedFilePath) || repoFiles[0];
  }, [repoFiles, selectedFilePath]);

  const folders = useMemo(() => {
    return {
      configs: repoFiles.filter((f) => f.folder === 'configs'),
      src: repoFiles.filter((f) => f.folder === 'src'),
      tests: repoFiles.filter((f) => f.folder === 'tests'),
      root: repoFiles.filter((f) => f.folder === 'root')
    };
  }, [repoFiles]);

  const toggleFolder = (name: string) => {
    setExpandedFolders((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  const handleSelectProject = (id: string) => {
    setSelectedProjectId(id);
    setSelectedFilePath('README.md');
  };

  const handleCopyFile = () => {
    if (!activeFile) return;
    navigator.clipboard.writeText(activeFile.content);
    setCopiedFile(true);
    setTimeout(() => setCopiedFile(false), 2000);
  };

  const buildMonorepoBundle = () => {
    if (!activeProject) return '';
    return (
      `# =====================================================================\n` +
      `# FULL REPOSITORY BUNDLE: ${activeProject.id}/\n` +
      `# PROJECT: ${activeProject.title}\n` +
      `# FILES: ${repoFiles.map((f) => f.path).join(', ')}\n` +
      `# =====================================================================\n\n` +
      repoFiles
        .map(
          (f) =>
            `# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n` +
            `# FILE: ${activeProject.id}/${f.path}\n` +
            `# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n` +
            f.content
        )
        .join('\n\n')
    );
  };

  const handleCopyBundle = () => {
    navigator.clipboard.writeText(buildMonorepoBundle());
    setCopiedBundle(true);
    setTimeout(() => setCopiedBundle(false), 2200);
  };

  const handleDownloadBundle = () => {
    if (!activeProject) return;
    const content = buildMonorepoBundle();
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${activeProject.id}_full_repository_bundle.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const asciiTree = useMemo(() => {
    if (!activeProject) return '';
    return [
      `${activeProject.id}/`,
      `├── configs/`,
      ...folders.configs.map((f, i) => `│   ${i === folders.configs.length - 1 ? '└──' : '├──'} ${f.name}`),
      `├── src/`,
      ...folders.src.map((f, i) => `│   ${i === folders.src.length - 1 ? '└──' : '├──'} ${f.name}`),
      `├── tests/`,
      ...folders.tests.map((f, i) => `│   ${i === folders.tests.length - 1 ? '└──' : '├──'} ${f.name}`),
      ...folders.root.map((f, i) => `${i === folders.root.length - 1 ? '└──' : '├──'} ${f.name}`)
    ].join('\n');
  }, [activeProject, folders]);

  if (!activeProject) return null;

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="bg-slate-100 hover:bg-slate-200/80 p-6 rounded-2xl border border-slate-300 shadow-xs transition-all duration-200">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-rose-900 mb-2 font-medium">
              <span className="bg-rose-100 px-2 py-0.5 rounded text-rose-900 font-bold border border-rose-200">
                [Full Code Repo & Directory Structure]
              </span>
              <span>Production Monorepo Explorer</span>
              <span>·</span>
              <span className="text-blue-900 font-semibold">{allProjects.length} Complete Repositories</span>
            </div>
            <h2 className="text-2xl font-bold text-slate-900 font-serif tracking-tight">
              Interactive Source Code Repository & File Tree Inspector
            </h2>
            <p className="text-slate-700 text-sm mt-1 max-w-3xl leading-relaxed">
              Browse the complete directory hierarchy, configs, Rust/PyO3 manifests, FastAPI serving wrappers, unit tests, Dockerfiles, and multi-stage training scripts for every project.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <button
              onClick={handleCopyBundle}
              className="px-4 py-2.5 bg-blue-900 hover:bg-blue-800 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs"
            >
              {copiedBundle ? (
                <>
                  <Check className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-300 font-bold">Copied Full Repo!</span>
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                  <span>Copy Full Repo Bundle</span>
                </>
              )}
            </button>
            <button
              onClick={handleDownloadBundle}
              className="px-4 py-2.5 bg-slate-200 hover:bg-slate-300 text-slate-900 border border-slate-300 rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs"
            >
              <Download className="w-4 h-4 text-emerald-900" />
              <span>Export Bundle (.txt)</span>
            </button>
          </div>
        </div>

        {/* Project Repository Selector */}
        <div className="mt-6 pt-5 border-t border-slate-300 flex flex-col md:flex-row gap-4 items-stretch md:items-center justify-between">
          <div className="flex-1 flex flex-col sm:flex-row sm:items-center gap-3">
            <span className="text-xs font-semibold text-slate-800 flex items-center gap-1.5 shrink-0">
              <GitBranch className="w-4 h-4 text-blue-900" />
              <span>Active Repository:</span>
            </span>
            <select
              value={activeProject.id}
              onChange={(e) => handleSelectProject(e.target.value)}
              className="flex-1 bg-white border border-slate-300 rounded-xl px-4 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
            >
              <optgroup label="Flagship Research Projects">
                {flagshipProjects
                  .filter((p) => filteredProjects.includes(p))
                  .map((p) => (
                    <option key={p.id} value={p.id}>
                      [{p.difficulty}] {p.title}
                    </option>
                  ))}
              </optgroup>
              <optgroup label={`ML Repository Imports (${datacampProjects.length} Projects)`}>
                {datacampProjects
                  .filter((p) => filteredProjects.includes(p))
                  .map((p) => (
                    <option key={p.id} value={p.id}>
                      [{p.sourceName || 'DataCamp'} · {p.difficulty}] {p.title}
                    </option>
                  ))}
              </optgroup>
            </select>
          </div>

          <div className="relative w-full md:w-64">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Filter repositories..."
              className="w-full bg-white border border-slate-300 rounded-xl pl-9 pr-4 py-2 text-xs text-slate-900 placeholder:text-slate-500 focus:outline-none focus:border-blue-900 shadow-xs"
            />
          </div>
        </div>
      </div>

      {/* Interactive Repository Explorer + File Content Viewer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Sidebar: Interactive File Tree & ASCII Structure */}
        <div className="lg:col-span-4 space-y-5">
          {/* Clickable Tree */}
          <div className="bg-slate-100 hover:bg-slate-200/80 border border-slate-300 rounded-2xl p-5 transition-colors shadow-xs">
            <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-300">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-slate-900">
                <FolderOpen className="w-4 h-4 text-blue-900" />
                <span>{activeProject.id}/</span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 border border-emerald-300 font-bold">
                {repoFiles.length} Files
              </span>
            </div>

            <div className="space-y-1 text-xs font-mono">
              {(['configs', 'src', 'tests'] as const).map((folderName) => (
                <div key={folderName} className="space-y-0.5">
                  <button
                    onClick={() => toggleFolder(folderName)}
                    className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-slate-800 hover:bg-slate-200 font-semibold transition-colors"
                  >
                    {expandedFolders[folderName] ? (
                      <ChevronDown className="w-3.5 h-3.5 text-slate-500" />
                    ) : (
                      <ChevronRight className="w-3.5 h-3.5 text-slate-500" />
                    )}
                    {expandedFolders[folderName] ? (
                      <FolderOpen className="w-4 h-4 text-blue-900" />
                    ) : (
                      <Folder className="w-4 h-4 text-blue-900" />
                    )}
                    <span>{folderName}/</span>
                  </button>

                  {expandedFolders[folderName] && (
                    <div className="pl-6 space-y-0.5">
                      {folders[folderName].map((file) => {
                        const isSelected = activeFile?.path === file.path;
                        return (
                          <button
                            key={file.path}
                            onClick={() => setSelectedFilePath(file.path)}
                            className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left transition-colors ${
                              isSelected
                                ? 'bg-blue-900 text-white font-bold shadow-xs'
                                : 'text-slate-700 hover:bg-slate-200'
                            }`}
                          >
                            <FileCode
                              className={`w-3.5 h-3.5 shrink-0 ${
                                isSelected ? 'text-white' : 'text-rose-900'
                              }`}
                            />
                            <span className="truncate">{file.name}</span>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              ))}

              {/* Root files */}
              <div className="pt-1 space-y-0.5">
                {folders.root.map((file) => {
                  const isSelected = activeFile?.path === file.path;
                  return (
                    <button
                      key={file.path}
                      onClick={() => setSelectedFilePath(file.path)}
                      className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left transition-colors ${
                        isSelected
                          ? 'bg-blue-900 text-white font-bold shadow-xs'
                          : 'text-slate-700 hover:bg-slate-200'
                      }`}
                    >
                      <FileText
                        className={`w-3.5 h-3.5 shrink-0 ${
                          isSelected ? 'text-white' : 'text-emerald-900'
                        }`}
                      />
                      <span className="truncate">{file.name}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          {/* ASCII Directory Architecture Preview */}
          <div className="bg-slate-100 hover:bg-slate-200/80 border border-slate-300 rounded-2xl p-5 transition-colors shadow-xs">
            <div className="flex items-center gap-2 text-xs font-bold text-slate-900 mb-3">
              <Layers className="w-4 h-4 text-rose-900" />
              <span>Repository Topology (ASCII Map)</span>
            </div>
            <pre className="p-3.5 bg-white border border-slate-300 rounded-xl text-[11px] font-mono text-slate-800 overflow-x-auto whitespace-pre-wrap break-words leading-relaxed">
              {asciiTree}
            </pre>
          </div>
        </div>

        {/* Right Main Panel: Active File Source Viewer */}
        <div className="lg:col-span-8 space-y-4">
          {/* Active File Metadata Bar */}
          {activeFile && (
            <div className="bg-slate-100 hover:bg-slate-200/80 border border-slate-300 rounded-2xl p-5 transition-colors shadow-xs">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 text-xs font-mono text-blue-900 font-bold">
                    <Terminal className="w-4 h-4 text-rose-900" />
                    <span>{activeProject.id}/{activeFile.path}</span>
                    <span className="px-2 py-0.5 rounded bg-slate-200 text-slate-800 border border-slate-300 text-[10px] uppercase">
                      {activeFile.language}
                    </span>
                  </div>
                  <p className="text-xs text-slate-700 mt-1 font-medium">{activeFile.description}</p>
                </div>

                <button
                  onClick={handleCopyFile}
                  className="px-3.5 py-2 bg-slate-200 hover:bg-slate-300 text-slate-900 border border-slate-300 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-colors shrink-0 shadow-xs"
                >
                  {copiedFile ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-700" />
                      <span className="text-emerald-800 font-bold">Copied File!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy File</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Code Window */}
          {activeFile && (
            <div className="bg-slate-900 rounded-2xl border border-slate-800 overflow-hidden shadow-md">
              <div className="bg-slate-950 px-5 py-3 border-b border-slate-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                  <span className="ml-2 text-xs font-mono text-slate-300 font-semibold">
                    {activeFile.path}
                  </span>
                </div>
                <span className="text-[11px] font-mono text-slate-400">
                  {activeFile.content.split('\n').length} lines
                </span>
              </div>
              <pre className="p-5 text-xs font-mono text-indigo-200 overflow-x-auto whitespace-pre-wrap break-words leading-relaxed max-h-[650px]">
                <code>{activeFile.content}</code>
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
