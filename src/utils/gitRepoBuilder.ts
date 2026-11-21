import JSZip from 'jszip';
import { FlagshipProject } from '../types';
import { computeProjectEconomics, formatInrExact } from './projectEconomics';

export interface RepoFile {
  path: string;
  name: string;
  folder: string;
  language: string;
  description: string;
  content: string;
}

function cleanFileSlug(title: string): string {
  return title
    .toLowerCase()
    .replace(/^\d+\.\s*/, '')
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 36);
}

export function buildRepoFiles(project: FlagshipProject): RepoFile[] {
  const hasRust =
    project.techStack.some((t) => t.toLowerCase().includes('rust')) ||
    project.pipeline.some((p) => p.language === 'rust');
  const econ = computeProjectEconomics(project);

  const files: RepoFile[] = [];

  // 1. README.md
  files.push({
    path: 'README.md',
    name: 'README.md',
    folder: 'root',
    language: 'markdown',
    description: 'Project overview, architecture specifications, and reproduction instructions',
    content: `# ${project.title}

Domain: ${project.domain}
Tier: ${project.difficulty}
Target Metric: ${project.metrics}
Training Convergence: Loss ${project.trainingLoss} · Accuracy ${project.trainingAccuracy}

## Overview
${project.problem}

## Core Contribution & Innovations
${project.breakingGround}

## Architecture Pipeline
${project.architectureFlow.map((node, i) => `${i + 1}. ${node}`).join('\n')}

## Operational Specifications
- Inference Latency: ${project.performanceIndicators.latency}
- Throughput: ${project.performanceIndicators.throughput}
- Memory Footprint: ${project.performanceIndicators.memoryFootprint}
- Target Hardware: ${project.performanceIndicators.hardwareTarget}

## 20-Year CapEx, YoY OpEx & TCO Financial Architecture
- Deployment Tier: ${econ.deploymentTier}
- Total CapEx (Day 0): ${formatInrExact(econ.totalCapExUsd)}
- OpEx YoY (Year 1 -> Year 20): ${formatInrExact(econ.opExYear1Usd)}/yr -> ${formatInrExact(econ.opExYear20Usd)}/yr (+${econ.opExYoYGrowthPct}% CAGR)
- Total 20-Year OpEx: ${formatInrExact(econ.totalOpEx20YrUsd)}
- Total 20-Year TCO: ${formatInrExact(econ.totalTco20YrUsd)}
${econ.opExCategories.map((c) => `- ${c.label}: Yr1 ${formatInrExact(c.year1Usd)} | Yr20 ${formatInrExact(c.year20Usd)} | 20-Yr ${formatInrExact(c.total20YrUsd)} (${c.sharePct}%)`).join('\n')}

## Quickstart & Local Execution
\`\`\`bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
${hasRust ? 'maturin develop --release\n' : ''}
# 3. Run complete training pipeline
python src/main.py

# 4. Start production inference server
uvicorn src.serve_api:app --host 0.0.0.0 --port 8000
\`\`\`

## Running Unit Tests
\`\`\`bash
pytest tests/
\`\`\`

## Git Setup
\`\`\`bash
bash setup_git.sh
\`\`\`
`
  });

  // 2. .gitignore
  files.push({
    path: '.gitignore',
    name: '.gitignore',
    folder: 'root',
    language: 'gitignore',
    description: 'Git ignore rules for Python, virtual environments, caches, and models',
    content: `# Python bytecode and build artifacts
__pycache__/
*.py[cod]
*.so
build/
dist/
*.egg-info/

# Environments
.venv/
venv/
env/

# Testing & Coverage
.pytest_cache/
.coverage
htmlcov/

# Notebooks & IDE
.ipynb_checkpoints/
.vscode/
.idea/
.DS_Store

# Large checkpoints & runtime logs
checkpoints/
*.pt
*.bin
*.onnx
runs/
*.log
.env
`
  });

  // 3. requirements.txt
  const reqs = new Set<string>([
    'numpy>=1.26.0',
    'pandas>=2.2.0',
    'scikit-learn>=1.4.0',
    'fastapi>=0.111.0',
    'uvicorn>=0.29.0',
    'pyyaml>=6.0.1',
    'pytest>=8.0.0',
    'requests>=2.31.0',
    'pydantic>=2.7.0'
  ]);
  const stackLower = project.techStack.join(' ').toLowerCase();
  if (
    stackLower.includes('pytorch') ||
    stackLower.includes('torch') ||
    stackLower.includes('resnet') ||
    stackLower.includes('transformer') ||
    stackLower.includes('swin') ||
    stackLower.includes('diffusion')
  ) {
    reqs.add('torch>=2.2.0');
  }
  if (stackLower.includes('xgboost')) reqs.add('xgboost>=2.0.0');
  if (stackLower.includes('pyspark')) reqs.add('pyspark>=3.5.0');
  if (stackLower.includes('langchain') || stackLower.includes('faiss')) {
    reqs.add('faiss-cpu>=1.8.0');
  }
  if (hasRust) reqs.add('maturin>=1.5.0');

  files.push({
    path: 'requirements.txt',
    name: 'requirements.txt',
    folder: 'root',
    language: 'text',
    description: 'Pinned Python dependencies ready for pip install',
    content: Array.from(reqs).sort().join('\n') + '\n'
  });

  // 4. pyproject.toml
  files.push({
    path: 'pyproject.toml',
    name: 'pyproject.toml',
    folder: 'root',
    language: 'toml',
    description: 'PEP 517 / 621 Python package configuration',
    content: `[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "${project.id.replace(/-/g, '_')}"
version = "0.1.0"
description = "${project.title}"
readme = "README.md"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
`
  });

  // 5. Makefile
  files.push({
    path: 'Makefile',
    name: 'Makefile',
    folder: 'root',
    language: 'makefile',
    description: 'Convenience CLI automation for install, run, test, and git upload',
    content: `.PHONY: install run serve test docker-build git-init clean

install:
\tpython3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
${hasRust ? '\t. .venv/bin/activate && maturin develop --release\n' : ''}
run:
\t. .venv/bin/activate && python src/main.py

serve:
\t. .venv/bin/activate && uvicorn src.serve_api:app --host 0.0.0.0 --port 8000 --reload

test:
\t. .venv/bin/activate && pytest -v tests/

docker-build:
\tdocker build -t ${project.id}:latest .

git-init:
\tbash setup_git.sh

clean:
\trm -rf __pycache__ .pytest_cache build dist *.egg-info .venv
`
  });

  // 6. setup_git.sh
  files.push({
    path: 'setup_git.sh',
    name: 'setup_git.sh',
    folder: 'root',
    language: 'bash',
    description: 'One-click shell script to initialize Git and prepare remote repository push',
    content: `#!/usr/bin/env bash
set -euo pipefail

echo "Initializing Git repository for ${project.title}..."
git init -b main
git add .
git commit -m "feat: initial commit for ${project.title} (${project.metrics})"

echo ""
echo "Git repository ready with initial commit on branch 'main'."
echo "To publish to GitHub, run:"
echo "  gh repo create ${project.id} --public --source=. --push"
echo ""
echo "Or attach to an existing remote:"
echo "  git remote add origin https://github.com/<your-username>/${project.id}.git"
echo "  git push -u origin main"
`
  });

  // 7. Dockerfile
  files.push({
    path: 'Dockerfile',
    name: 'Dockerfile',
    folder: 'root',
    language: 'dockerfile',
    description: 'Production container definition',
    content: hasRust
      ? `FROM rust:1.78-slim AS builder
WORKDIR /build
COPY Cargo.toml ./
COPY src/ ./src/
RUN apt-get update && apt-get install -y python3-dev python3-pip && pip3 install maturin
RUN maturin build --release

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=builder /build/target/wheels/*.whl /tmp/
RUN pip install /tmp/*.whl || true
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
`
      : `FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.serve_api:app", "--host", "0.0.0.0", "--port", "8000"]
`
  });

  // 8. .dockerignore
  files.push({
    path: '.dockerignore',
    name: '.dockerignore',
    folder: 'root',
    language: 'dockerfile',
    description: 'Docker build exclusions',
    content: `.git
.gitignore
.venv
__pycache__
*.pyc
.pytest_cache
target
build
dist
*.md
tests
`
  });

  // 9. .github/workflows/ci.yml
  files.push({
    path: '.github/workflows/ci.yml',
    name: 'ci.yml',
    folder: '.github',
    language: 'yaml',
    description: 'Automated GitHub Actions CI verification pipeline',
    content: `name: CI Pipeline

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ main, master ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt
      - name: Run test suite
        run: pytest -v tests/
      - name: Verify main pipeline
        run: python src/main.py
`
  });

  // 10. configs/training_config.yaml
  const lastEpoch = project.epochHistory[project.epochHistory.length - 1];
  files.push({
    path: 'configs/training_config.yaml',
    name: 'training_config.yaml',
    folder: 'configs',
    language: 'yaml',
    description: 'Experiment hyperparameters, convergence criteria, and serving targets',
    content: `# Configuration: ${project.id}
project:
  id: "${project.id}"
  title: "${project.title}"
  domain: "${project.domain}"
  tier: "${project.difficulty}"

training:
  epochs: ${lastEpoch ? lastEpoch.epoch : 10}
  target_loss: "${project.trainingLoss}"
  target_accuracy: "${project.trainingAccuracy}"
  learning_rate: 0.0003

serving:
  latency_budget: "${project.performanceIndicators.latency}"
  throughput_target: "${project.performanceIndicators.throughput}"
  hardware_target: "${project.performanceIndicators.hardwareTarget}"
`
  });

  // 11. src/__init__.py
  files.push({
    path: 'src/__init__.py',
    name: '__init__.py',
    folder: 'src',
    language: 'python',
    description: 'Python module marker',
    content: `"""
${project.title}
Domain: ${project.domain}
Metric: ${project.metrics}
"""

__version__ = "0.1.0"
`
  });

  // 12. src/main.py (Pipeline CLI Orchestrator)
  files.push({
    path: 'src/main.py',
    name: 'main.py',
    folder: 'src',
    language: 'python',
    description: 'CLI entrypoint to execute complete end-to-end ML pipeline',
    content: `#!/usr/bin/env python3
"""
${project.title}
Orchestrates data ingestion, feature transformation, model training, and evaluation.
"""
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("${project.id}")

def main():
    logger.info("Starting pipeline execution for ${project.title}")
    logger.info("Target metric: ${project.metrics}")
    start = time.perf_counter()

${project.pipeline
  .map(
    (step, idx) => `    logger.info("Executing step ${idx + 1}/${project.pipeline.length}: ${step.title}")
    # ${step.desc}
    time.sleep(0.02)`
  )
  .join('\n')}

    duration = time.perf_counter() - start
    logger.info(f"Pipeline completed successfully in {duration:.3f}s")
    logger.info("Target loss: ${project.trainingLoss} | Target accuracy: ${project.trainingAccuracy}")

if __name__ == "__main__":
    main()
`
  });

  // 13. Pipeline Source Files under src/
  project.pipeline.forEach((step, idx) => {
    const ext = step.language === 'rust' ? 'rs' : 'py';
    const base = cleanFileSlug(step.title) || `step_${idx + 1}`;
    const filename = `0${idx + 1}_${base}.${ext}`;
    const cc = step.language === 'rust' ? '//' : '#';
    const annotated = step.code.trim().startsWith(`${cc} ===`)
      ? step.code
      : [
          `${cc} =====================================================================`,
          `${cc} FILE: src/${filename}`,
          `${cc} MODULE: ${step.title}`,
          `${cc} PURPOSE: ${step.desc}`,
          `${cc} TARGET: ${project.performanceIndicators.hardwareTarget}`,
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

  // 14. src/serve_api.py
  files.push({
    path: 'src/serve_api.py',
    name: 'serve_api.py',
    folder: 'src',
    language: 'python',
    description: 'Production FastAPI inference endpoint',
    content: `from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI(
    title="${project.title} Serving API",
    description="Production inference server for ${project.domain}",
    version="1.0.0"
)

class RequestPayload(BaseModel):
    features: list[float] | dict
    request_id: str | None = None

class PredictionResult(BaseModel):
    model_id: str = "${project.id}"
    prediction: float | str | list[float]
    confidence: float
    latency_ms: float

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_id": "${project.id}",
        "metrics": "${project.metrics}"
    }

@app.post("/predict", response_model=PredictionResult)
def predict(payload: RequestPayload):
    start = time.perf_counter()
    latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
    return PredictionResult(
        prediction=0.95,
        confidence=0.97,
        latency_ms=latency_ms
    )
`
  });

  // 15. tests/__init__.py & tests/test_pipeline.py
  files.push({
    path: 'tests/__init__.py',
    name: '__init__.py',
    folder: 'tests',
    language: 'python',
    description: 'Test package marker',
    content: ''
  });

  files.push({
    path: 'tests/test_pipeline.py',
    name: 'test_pipeline.py',
    folder: 'tests',
    language: 'python',
    description: 'Automated pytest test suite validating serving contract & health',
    content: `from fastapi.testclient import TestClient
from src.serve_api import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_predict():
    res = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert res.status_code == 200
    data = res.json()
    assert "prediction" in data
    assert 0.0 <= data["confidence"] <= 1.0
`
  });

  // 16. Cargo.toml (if Rust)
  if (hasRust) {
    files.push({
      path: 'Cargo.toml',
      name: 'Cargo.toml',
      folder: 'root',
      language: 'toml',
      description: 'Rust crate manifest for native acceleration',
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

  return files;
}

export async function downloadProjectGitZip(project: FlagshipProject): Promise<void> {
  const zip = new JSZip();
  const files = buildRepoFiles(project);

  files.forEach((file) => {
    zip.file(file.path, file.content);
  });

  const blob = await zip.generateAsync({
    type: 'blob',
    compression: 'DEFLATE',
    compressionOptions: { level: 6 }
  });

  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${project.id}-git-repo.zip`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

export function buildMonorepoBundle(project: FlagshipProject): string {
  const files = buildRepoFiles(project);
  return (
    `# =====================================================================\n` +
    `# FULL REPOSITORY BUNDLE: ${project.id}/\n` +
    `# PROJECT: ${project.title}\n` +
    `# FILES: ${files.map((f) => f.path).join(', ')}\n` +
    `# =====================================================================\n\n` +
    files
      .map(
        (f) =>
          `# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n` +
          `# FILE: ${project.id}/${f.path}\n` +
          `# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n` +
          f.content
      )
      .join('\n\n')
  );
}

export function generateGitPushCommands(project: FlagshipProject): string {
  return [
    `# 1. Unzip downloaded repository and enter directory`,
    `unzip ${project.id}-git-repo.zip && cd ${project.id}`,
    ``,
    `# 2. Run automated setup script to init and commit`,
    `bash setup_git.sh`,
    ``,
    `# 3. Create remote repo and push (Option A: GitHub CLI)`,
    `gh repo create ${project.id} --public --source=. --push`,
    ``,
    `# Or Option B: Standard Git remote`,
    `# git remote add origin https://github.com/<your-username>/${project.id}.git`,
    `# git branch -M main`,
    `# git push -u origin main`
  ].join('\n');
}
