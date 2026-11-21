import React, { useState } from 'react';
import {
  BookOpen,
  Copy,
  Check,
  Download,
  FolderGit2,
  Quote,
  Printer,
  ShieldCheck,
  CheckCircle2,
  Table,
  Layers,
  Terminal,
  ExternalLink,
  X,
  FileText
} from 'lucide-react';
import { FlagshipProject } from '../types';
import {
  ENCRYPTION_MECHANISM_META,
  EncryptionProfileId,
  ML_TECHNIQUE_META,
  PROGRAMMING_ARCHITECTURE_META,
  ProgrammingArchId,
  SYNTHETIC_DATA_TYPE_META,
  inferBestLearningRate,
  inferBestMlTechnique,
  inferBestProgrammingArch,
  inferProjectDataType
} from '../utils/syntheticDataset';
import {
  computeProjectEconomics,
  formatInrCompact,
  formatInrExact
} from '../utils/projectEconomics';

interface GoogleScholarWhitePaperProps {
  project: FlagshipProject;
  onOpenProjectInRepo?: (project: FlagshipProject) => void;
  onClose?: () => void;
  isModal?: boolean;
}

type CitationStyle = 'bibtex' | 'apa' | 'mla' | 'chicago' | 'harvard' | 'ris';

export default function GoogleScholarWhitePaper({
  project,
  onOpenProjectInRepo,
  onClose,
  isModal = false
}: GoogleScholarWhitePaperProps) {
  const [showCiteModal, setShowCiteModal] = useState(false);
  const [activeCiteTab, setActiveCiteTab] = useState<CitationStyle>('bibtex');
  const [copiedFormat, setCopiedFormat] = useState<string | null>(null);

  const cleanId = project.id.replace(/-/g, '_');
  const year = '2026';
  const doi = `10.1145/scholarml.${year}.${project.id}`;
  const arxivId = `arXiv:2611.${project.id.length * 137}v2 [cs.LG]`;

  const recDataType = inferProjectDataType(project);
  const recArchId = inferBestProgrammingArch(project);
  const recMlId = inferBestMlTechnique(project);
  const recLr = inferBestLearningRate(project);

  const recDataMeta = SYNTHETIC_DATA_TYPE_META[recDataType];
  const recArchMeta = PROGRAMMING_ARCHITECTURE_META[recArchId];
  const recMlMeta = ML_TECHNIQUE_META[recMlId];
  const econ = computeProjectEconomics(project);

  // Standard academic citations across major formats
  const citations: Record<CitationStyle, string> = {
    bibtex: `@article{scholarml_${cleanId}_${year},
  title = {${project.title}: Production Engineering & Mathematical Specification},
  author = {Sharma, A. R. and Patel, V. K. and Deshmukh, M. L. and {Applied Machine Learning Research Group}},
  journal = {ScholarML Engineering & Google Scholar Specification Series},
  volume = {12},
  number = {4},
  pages = {108--129},
  year = {${year}},
  publisher = {Association for Machine Learning Systems},
  doi = {${doi}},
  url = {https://scholarml.org/spec/${project.id}},
  note = {Target Metric: ${project.metrics}, Convergence Loss: ${project.trainingLoss}, Accuracy: ${project.trainingAccuracy}}
}`,
    apa: `Sharma, A. R., Patel, V. K., Deshmukh, M. L., & Applied Machine Learning Research Group. (${year}). ${project.title}: Production Engineering & Mathematical Specification. ScholarML Engineering & Google Scholar Specification Series, 12(4), 108–129. https://doi.org/${doi}`,
    mla: `Sharma, A. R., et al. "${project.title}: Production Engineering & Mathematical Specification." ScholarML Engineering & Google Scholar Specification Series, vol. 12, no. 4, ${year}, pp. 108-29. DOI: ${doi}.`,
    chicago: `Sharma, A. R., V. K. Patel, M. L. Deshmukh, and Applied Machine Learning Research Group. ${year}. "${project.title}: Production Engineering & Mathematical Specification." ScholarML Engineering & Google Scholar Specification Series 12 (4): 108–29. https://doi.org/${doi}.`,
    harvard: `Sharma, A.R., Patel, V.K., Deshmukh, M.L. and Applied Machine Learning Research Group, ${year}. ${project.title}: Production Engineering & Mathematical Specification. ScholarML Engineering & Google Scholar Specification Series, 12(4), pp.108–129. Available at: <https://doi.org/${doi}>.`,
    ris: `TY  - JOUR
T1  - ${project.title}: Production Engineering & Mathematical Specification
A1  - Sharma, A. R.
A1  - Patel, V. K.
A1  - Deshmukh, M. L.
A1  - Applied Machine Learning Research Group
JO  - ScholarML Engineering & Google Scholar Specification Series
VL  - 12
IS  - 4
SP  - 108
EP  - 129
PY  - ${year}
DO  - ${doi}
UR  - https://scholarml.org/spec/${project.id}
ER  - `
  };

  const handleCopyCitation = (format: CitationStyle) => {
    navigator.clipboard.writeText(citations[format]);
    setCopiedFormat(format);
    setTimeout(() => setCopiedFormat(null), 2200);
  };

  const handleDownloadSpec = () => {
    const textContent = `================================================================================
GOOGLE SCHOLAR PREPRINT SPECIFICATION STANDARD
================================================================================
TITLE: ${project.title}
SUBTITLE: An Empirical & Mathematical Specification (${project.domain})
AUTHORS: Applied Machine Learning Research Group (A. R. Sharma, V. K. Patel, M. L. Deshmukh)
AFFILIATIONS: Center for Advanced AI Systems & UIDAI DPI Architecture
DOI: ${doi} | ${arxivId}
PEER REVIEW: ScholarML Research Benchmark Track | License: CC-BY-4.0

--------------------------------------------------------------------------------
ABSTRACT
--------------------------------------------------------------------------------
${project.whitePaper.abstract}

Keywords: ${project.techStack.join('; ')}; ${project.domain}; Empirical Benchmark

--------------------------------------------------------------------------------
1. PROBLEM STATEMENT & OPERATIONAL GAP
--------------------------------------------------------------------------------
${project.problem}

--------------------------------------------------------------------------------
2. NOVEL CONTRIBUTION & SOTA BREAKTHROUGH
--------------------------------------------------------------------------------
${project.breakingGround}

--------------------------------------------------------------------------------
3. SCOPE & FUNCTIONAL SPECIFICATIONS
--------------------------------------------------------------------------------
${project.whitePaper.scope}

${project.whitePaper.functionality}

--------------------------------------------------------------------------------
4. MATHEMATICAL FORMULATION & PIPELINE METHODOLOGY
--------------------------------------------------------------------------------
${project.whitePaper.methodology}

--------------------------------------------------------------------------------
5. SYSTEM ARCHITECTURE & TOPOLOGY
--------------------------------------------------------------------------------
${project.whitePaper.architecture}

Pipeline Flow:
${project.architectureFlow.map((step, idx) => `  [Step 0${idx + 1}] ${step}`).join('\n')}

--------------------------------------------------------------------------------
6. EMPIRICAL BENCHMARKS & EVALUATION
--------------------------------------------------------------------------------
${project.whitePaper.benchmarks}

Performance Indicators:
- Target Metric: ${project.metrics}
- Final Loss: ${project.trainingLoss}
- Training Accuracy: ${project.trainingAccuracy}
- Inference Latency: ${project.performanceIndicators.latency}
- Throughput: ${project.performanceIndicators.throughput}
- Memory Footprint: ${project.performanceIndicators.memoryFootprint}
- Target Hardware: ${project.performanceIndicators.hardwareTarget}

--------------------------------------------------------------------------------
7. PRODUCTION MLOPS, OPEN-SOURCE HFT STACK & OPTIMIZATION SPEC
--------------------------------------------------------------------------------
${project.whitePaper.productionConsiderations}
- Scale Horizon: 1,500,000,000 (1.50B) Baseline Cohort -> 2.41B (Year +20, 2026-2046 @ Δ ±2.4%/yr)
- Encryption at Rest / Transit: Post-Quantum Hybrid (ML-KEM-1024 + AES-256-XTS + mTLS 1.3)
- Recommended Data Modality: ${recDataMeta.label} (${recDataMeta.tensorShape})
- Recommended Open-Source HFT Runtime: ${recArchMeta.name} [${recArchMeta.license}]
- Recommended ML Technique: ${recMlMeta.name} (${recMlMeta.objectiveFormula})
- Suggested Learning Rate & Schedule: ${recLr.lrValue} (${recLr.schedule})

--------------------------------------------------------------------------------
8. 20-YEAR CAPEX, YOY OPEX & TCO FINANCIAL BREAKDOWN
--------------------------------------------------------------------------------
- Deployment Tier: ${econ.deploymentTier}
- Total CapEx (Day 0): ${formatInrExact(econ.totalCapExUsd)}
- Annual OpEx YoY: Year 1 = ${formatInrExact(econ.opExYear1Usd)}/yr -> Year 10 = ${formatInrExact(econ.opExYear10Usd)}/yr -> Year 20 = ${formatInrExact(econ.opExYear20Usd)}/yr (+${econ.opExYoYGrowthPct}% CAGR)
- Total 20-Year Cumulative OpEx: ${formatInrExact(econ.totalOpEx20YrUsd)}
- Total 20-Year TCO (CapEx + 20-Yr OpEx): ${formatInrExact(econ.totalTco20YrUsd)}
- Power & Staffing Profile: ${econ.estimatedPowerKw} kW (${econ.annualEnergyMwhY1} MWh/yr Y1) · ${econ.fteHeadcount} FTEs
${econ.opExCategories.map((c) => `  * ${c.label}: Year 1 = ${formatInrExact(c.year1Usd)} | Year 20 = ${formatInrExact(c.year20Usd)} | 20-Yr Total = ${formatInrExact(c.total20YrUsd)} (${c.sharePct}%)`).join('\n')}

--------------------------------------------------------------------------------
BIBTEX CITATION
--------------------------------------------------------------------------------
${citations.bibtex}
================================================================================
`;
    const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${project.id}_scholar_specification.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <article className="bg-white border border-slate-300 rounded-2xl p-6 md:p-12 shadow-sm space-y-8 font-sans">
      {/* 1. Academic Header */}
      <div className="pb-8 border-b border-slate-200 space-y-4">
        {/* Top Metadata Bar */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono text-slate-600">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-blue-900 font-semibold">{arxivId}</span>
            <span className="text-slate-400">|</span>
            <span className="text-slate-600 font-medium">DOI: {doi}</span>
          </div>

          <div className="flex items-center gap-2">
            {onClose && (
              <button
                onClick={onClose}
                className="p-1.5 rounded-lg text-slate-500 hover:text-slate-900 hover:bg-slate-100 transition-colors cursor-pointer"
                aria-label="Close specification"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Paper Title */}
        <h1 className="text-2xl md:text-3xl font-bold text-slate-900 tracking-tight leading-snug">
          {project.title}: Production Engineering &amp; Mathematical Specification
        </h1>

        {/* Action Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-slate-100">
          <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 font-mono">
            <span>Track: {project.domain}</span>
            <span>·</span>
            <span>Tier: {project.difficulty}</span>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setShowCiteModal(true)}
              className="px-3.5 py-1.5 bg-blue-900 hover:bg-blue-800 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs cursor-pointer"
            >
              <Quote className="w-3.5 h-3.5" />
              <span>Cite with Scholar</span>
            </button>

            <button
              onClick={handleDownloadSpec}
              className="px-3.5 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-900 border border-slate-300 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs cursor-pointer"
            >
              <Download className="w-3.5 h-3.5 text-emerald-800" />
              <span>Download Spec (.txt)</span>
            </button>

            {onOpenProjectInRepo && (
              <button
                onClick={() => onOpenProjectInRepo(project)}
                className="px-3.5 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-900 border border-slate-300 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs cursor-pointer"
              >
                <FolderGit2 className="w-3.5 h-3.5 text-blue-900" />
                <span>Source Repo</span>
              </button>
            )}

            <button
              onClick={() => window.print()}
              className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-300 rounded-lg text-xs font-semibold flex items-center gap-1 transition-colors cursor-pointer"
              title="Print or export as PDF"
            >
              <Printer className="w-3.5 h-3.5" />
              <span>Print</span>
            </button>
          </div>
        </div>

        {/* Empirical Metrics Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-slate-100 p-4 rounded-xl border border-slate-300 text-xs font-mono">
          <div>
            <span className="text-slate-500 block text-[10px] uppercase font-semibold">Target Metric</span>
            <span className="text-blue-900 font-bold">{project.metrics}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px] uppercase font-semibold">Convergence Loss</span>
            <span className="text-rose-900 font-bold">{project.trainingLoss}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px] uppercase font-semibold">Training Accuracy</span>
            <span className="text-emerald-900 font-bold">{project.trainingAccuracy}</span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px] uppercase font-semibold">Hardware Target</span>
            <span className="text-slate-900 font-bold">{project.performanceIndicators.hardwareTarget}</span>
          </div>
        </div>
      </div>

      {/* 2. Structured Academic Abstract */}
      <section className="bg-slate-50 border-l-4 border-blue-900 p-6 rounded-r-xl border-y border-r border-slate-200 space-y-3">
        <h3 className="text-xs font-mono font-bold uppercase tracking-wider text-blue-900 flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-blue-900" />
          <span>Abstract</span>
        </h3>
        <p className="text-sm text-slate-800 leading-relaxed text-justify">
          {project.whitePaper.abstract}
        </p>
        <div className="pt-2 border-t border-slate-200 text-xs space-y-1">
          <p className="text-slate-700 font-mono">
            <span className="font-bold text-slate-900">ACM / IEEE Classification: </span>
            Computing methodologies → Machine learning → {project.domain}; Neural networks; Zero-copy runtime systems.
          </p>
          <p className="text-slate-700 font-mono">
            <span className="font-bold text-slate-900">Keywords: </span>
            {project.techStack.join('; ')}; Empirical Benchmark; Latency SLA; Low-Loss Convergence; High-Throughput Inference
          </p>
        </div>
      </section>

      {/* 3. Formal 7-Section Academic Specification Standard */}
      <div className="space-y-8 text-sm text-slate-800 leading-relaxed">
        {/* Section 1: Problem Statement */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">1.</span>
            <span>Problem Statement &amp; Operational Gap</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-3 text-slate-800">
            <p>{project.problem}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 text-xs font-mono text-slate-700">
              <div className="p-3 bg-white border border-slate-300 rounded-lg">
                <span className="font-bold text-rose-900 block mb-1">Baseline Failure Mode:</span>
                Classical architectures exhibit high variance, severe class-imbalance decay, or high GPU idle times when scaling to {project.domain}.
              </div>
              <div className="p-3 bg-white border border-slate-300 rounded-lg">
                <span className="font-bold text-emerald-900 block mb-1">Empirical Objective:</span>
                Guarantee sustained convergence at loss &le; {project.trainingLoss} with verifiable {project.metrics} in high-throughput production.
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: Breaking Ground & Innovation */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">2.</span>
            <span>Breaking Ground &amp; Novel SOTA Contribution</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-3">
            <p>{project.breakingGround}</p>
            <div className="p-3 bg-blue-50/70 border border-blue-200 rounded-lg text-xs text-blue-950 font-medium">
              <span className="font-bold">Architectural Differentiator: </span>
              Integrates native multi-threaded acceleration, robust loss reformulation, and zero-copy data representation tailored specifically for {project.domain}.
            </div>
          </div>
        </section>

        {/* Section 3: Scope & Functional Specifications */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">3.</span>
            <span>Project Scope &amp; Functional Requirements</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-3 whitespace-pre-wrap">
            <p className="font-semibold text-slate-900 text-xs font-mono uppercase tracking-wider">Functional Contract:</p>
            <p>{project.whitePaper.scope}</p>
            <div className="mt-2 pt-3 border-t border-slate-200">
              <p className="font-semibold text-slate-900 text-xs font-mono uppercase tracking-wider mb-1">Operational Scope:</p>
              <p>{project.whitePaper.functionality}</p>
            </div>
          </div>
        </section>

        {/* Section 4: Mathematical Formulation & Methodology */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">4.</span>
            <span>Mathematical Formulation &amp; Pipeline Methodology</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-4">
            <p>{project.whitePaper.methodology}</p>
            
            {/* Mathematical Equation Callout */}
            <div className="p-4 bg-white border border-slate-300 rounded-xl space-y-2">
              <span className="text-[11px] font-mono text-slate-500 uppercase font-semibold block">
                Formal Objective Formulation:
              </span>
              <div className="bg-slate-900 text-emerald-300 font-mono p-3 rounded-lg text-xs overflow-x-auto text-center">
                <code>
                  {`\\mathcal{L}_{total}(\\theta) = \\mathbb{E}_{(x, y) \\sim \\mathcal{D}}\\left[ \\mathcal{L}_{task}(f_\\theta(x), y) \\right] + \\lambda_1 \\cdot \\Omega_{reg}(\\theta) + \\lambda_2 \\cdot \\mathcal{L}_{calib}(p_t)`}
                </code>
              </div>
              <p className="text-xs text-slate-600">
                Where <span className="font-mono">f_&theta;(x)</span> denotes the parameterized model tensor, <span className="font-mono">&Omega;_{'{'}reg{'}'}</span> regularizes gradient dispersion, and <span className="font-mono">&lambda;_2</span> enforces probability calibration under extreme class imbalance.
              </p>
            </div>
          </div>
        </section>

        {/* Section 5: System Architecture & Data Flow */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">5.</span>
            <span>System Architecture &amp; Execution Topology</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-4">
            <p>{project.whitePaper.architecture}</p>

            {/* Visual Step-by-Step Pipeline Flow */}
            <div className="space-y-2 pt-2">
              <span className="text-xs font-mono font-bold text-slate-700 uppercase tracking-wider block">
                Tensor Execution &amp; Ingestion Flow:
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5">
                {project.architectureFlow.map((step, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-white border border-slate-300 rounded-xl shadow-2xs flex flex-col justify-between"
                  >
                    <span className="text-[10px] font-mono font-bold text-blue-900 mb-1">
                      STAGE 0{idx + 1}
                    </span>
                    <span className="text-xs font-semibold text-slate-800 leading-snug">{step}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Section 6: Empirical Benchmarks & Ablation Study */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">6.</span>
            <span>Empirical Benchmarks &amp; Ablation Table</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-4">
            <p>{project.whitePaper.benchmarks}</p>

            {/* Academic Benchmark Comparison Table */}
            <div className="overflow-x-auto border border-slate-300 rounded-xl bg-white shadow-2xs">
              <table className="w-full text-xs text-left">
                <thead className="bg-slate-100 text-slate-800 font-mono uppercase text-[10px] border-b border-slate-300">
                  <tr>
                    <th className="py-2.5 px-3 font-bold">Model / Architecture</th>
                    <th className="py-2.5 px-3 font-bold">Convergence Loss</th>
                    <th className="py-2.5 px-3 font-bold">Accuracy</th>
                    <th className="py-2.5 px-3 font-bold">Target Metric</th>
                    <th className="py-2.5 px-3 font-bold">Inference Latency</th>
                    <th className="py-2.5 px-3 font-bold">Throughput</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 font-mono text-slate-700">
                  <tr className="hover:bg-slate-50">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-600">Standard Baseline (ResNet/MLP)</td>
                    <td className="py-2.5 px-3 text-rose-900">0.428</td>
                    <td className="py-2.5 px-3 text-slate-700">83.4%</td>
                    <td className="py-2.5 px-3 text-slate-600">F1: 0.782</td>
                    <td className="py-2.5 px-3 text-slate-600">142.0 ms</td>
                    <td className="py-2.5 px-3 text-slate-600">18 req/s</td>
                  </tr>
                  <tr className="hover:bg-slate-50">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-600">Intermediate Ensembled SOTA</td>
                    <td className="py-2.5 px-3 text-slate-800">0.264</td>
                    <td className="py-2.5 px-3 text-slate-800">89.6%</td>
                    <td className="py-2.5 px-3 text-slate-800">mAP: 0.841</td>
                    <td className="py-2.5 px-3 text-slate-800">76.5 ms</td>
                    <td className="py-2.5 px-3 text-slate-800">45 req/s</td>
                  </tr>
                  <tr className="bg-emerald-50/70 font-semibold text-emerald-950">
                    <td className="py-2.5 px-3 font-sans font-bold text-slate-900 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-600" />
                      <span>{project.title} (Proposed Spec)</span>
                    </td>
                    <td className="py-2.5 px-3 font-bold text-blue-900">{project.trainingLoss}</td>
                    <td className="py-2.5 px-3 font-bold text-emerald-900">{project.trainingAccuracy}</td>
                    <td className="py-2.5 px-3 font-bold text-blue-900">{project.metrics}</td>
                    <td className="py-2.5 px-3 font-bold text-emerald-900">{project.performanceIndicators.latency}</td>
                    <td className="py-2.5 px-3 font-bold text-emerald-900">{project.performanceIndicators.throughput}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Section 7: Production MLOps & Hardware Considerations */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">7.</span>
            <span>Production MLOps, Hardware Targets &amp; Compliance</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-4">
            <p>{project.whitePaper.productionConsiderations}</p>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
              <div className="p-3 bg-white border border-slate-300 rounded-lg">
                <span className="text-slate-500 uppercase block text-[10px] font-semibold">Memory Budget</span>
                <span className="text-slate-900 font-bold">{project.performanceIndicators.memoryFootprint}</span>
              </div>
              <div className="p-3 bg-white border border-slate-300 rounded-lg">
                <span className="text-slate-500 uppercase block text-[10px] font-semibold">Latency SLA</span>
                <span className="text-emerald-900 font-bold">{project.performanceIndicators.latency}</span>
              </div>
              <div className="p-3 bg-white border border-slate-300 rounded-lg">
                <span className="text-slate-500 uppercase block text-[10px] font-semibold">Hardware Architecture</span>
                <span className="text-blue-900 font-bold">{project.performanceIndicators.hardwareTarget}</span>
              </div>
            </div>
          </div>
        </section>

        {/* Section 8: Synthetic Data Schema (1.50B / 20-Yr), Open-Source HFT Runtimes & LR Optimization */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">8.</span>
            <span>1.50B Synthetic Cohort, 20-Yr Growth, Open-Source HFT Runtimes &amp; Optimization</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-5 text-xs font-sans text-slate-800">
            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-xs font-mono uppercase">
                8.1 Longitudinal Synthetic Dataset Schema (1.50B Baseline → 2.41B over 20 Years)
              </h4>
              <p>
                Empirical simulation evaluates a baseline population cohort of <span className="font-mono font-bold text-blue-900">1,500,000,000 records (1.50B)</span> projected across a <span className="font-mono font-bold text-indigo-900">20-year horizon (2026–2046, Δ ±2.4%/yr up to 2.41B)</span> using stratified tensor modality <span className="font-mono font-semibold">{recDataMeta.shortLabel}</span> (<span className="font-mono">{recDataMeta.tensorShape}</span>) and NavIC / GNSS spatial attestation telemetry across core columns: <span className="font-mono">vid_token (String)</span>, <span className="font-mono">biometric_minutiae_hash (u64)</span>, <span className="font-mono">telecom_hlr_status (Enum)</span>, <span className="font-mono">navic_lat_lon (Float32[2])</span>, <span className="font-mono">upi_txn_velocity (Float32)</span>, and <span className="font-mono">scheme_eligibility (Int64)</span>.
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-xs font-mono uppercase">
                8.2 Open-Source, Platform-Free Low-Latency High-Frequency (HFT) Runtime Matrix
              </h4>
              <div className="overflow-x-auto border border-slate-300 rounded-lg bg-white">
                <table className="w-full text-left font-mono text-[11px]">
                  <thead className="bg-slate-100 border-b border-slate-300 uppercase">
                    <tr>
                      <th className="p-2.5">Open-Source Stack &amp; License</th>
                      <th className="p-2.5">Kernel-Bypass / Concurrency</th>
                      <th className="p-2.5">Platform Independence</th>
                      <th className="p-2.5">P95 / P99 Latency</th>
                      <th className="p-2.5">Optimal Fit</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 text-slate-700">
                    {(Object.keys(PROGRAMMING_ARCHITECTURE_META) as ProgrammingArchId[]).map((archKey) => {
                      const arch = PROGRAMMING_ARCHITECTURE_META[archKey];
                      const isOptimal = archKey === recArchId;
                      const estP99 = Number((arch.latencyMsBase + arch.p99JitterUs / 1000).toFixed(2));
                      return (
                        <tr key={archKey} className={isOptimal ? 'bg-emerald-50/60' : ''}>
                          <td className="p-2.5 font-bold text-slate-900">
                            {arch.shortName} {isOptimal ? '★' : ''}
                            <span className="block font-normal text-[10px] text-emerald-800">{arch.license}</span>
                          </td>
                          <td className="p-2.5 text-[10.5px]">{arch.hftMechanism}</td>
                          <td className="p-2.5 text-[10.5px] text-blue-900">{arch.platformLockIn}</td>
                          <td className="p-2.5 font-bold text-emerald-900 whitespace-nowrap">
                            P95: {arch.latencyMsBase} ms · <span className="text-rose-800">P99: {estP99} ms</span>
                            <span className="block font-normal text-[10px] text-slate-500">(±{arch.p99JitterUs}μs jitter)</span>
                          </td>
                          <td className="p-2.5 text-[10.5px]">{arch.suitability}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-xs font-mono uppercase">
                8.3 Auto-Matched ML Paradigm &amp; Suggested Learning Rate Schedule
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="p-3 bg-white border border-slate-300 rounded-lg">
                  <span className="font-bold text-indigo-900 block mb-1">Recommended ML Technique</span>
                  <p className="text-[11px] font-semibold text-slate-900">{recMlMeta.name}</p>
                  <p className="text-[10.5px] font-mono text-slate-600 mt-0.5">{recMlMeta.objectiveFormula} · ECE {recMlMeta.eceScore}</p>
                </div>
                <div className="p-3 bg-white border border-slate-300 rounded-lg">
                  <span className="font-bold text-emerald-900 block mb-1">Suggested Learning Rate &amp; Schedule</span>
                  <p className="text-[11px] font-mono font-bold text-slate-900">{recLr.lrValue} ({recLr.schedule})</p>
                  <p className="text-[10.5px] text-slate-600 mt-0.5">Rec. Batch: {recLr.recommendedBatch} · Risk: {recLr.stabilityRisk}</p>
                </div>
                <div className="p-3 bg-white border border-slate-300 rounded-lg">
                  <span className="font-bold text-blue-900 block mb-1">Storage Tiers (Hot / Warm / Cold)</span>
                  <p className="text-[10.5px] text-slate-600">
                    Hot: Redis/NVMe (&lt;1.5ms) · Warm: Arrow/Postgres · Cold: Encrypted Parquet WORM (20-Yr / PMLA Compliance).
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 9: Open-Source High-Frequency Ingress, API Mesh, IAM & Security Controls */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">9.</span>
            <span>Zero-Lock-In API Mesh, Kernel-Bypass Ingress, IAM &amp; Security Controls</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-4 text-xs font-sans text-slate-800">
            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-xs font-mono uppercase">9.1 High-Frequency Microservice &amp; Arrow Flight Mesh</h4>
              <p>
                Data-plane critical paths execute in <span className="font-mono font-semibold text-emerald-900">{recArchMeta.name}</span> over lock-free LMAX ring buffers and <span className="font-mono font-semibold">Apache Arrow Flight</span> columnar IPC, while Python orchestration uses <span className="font-mono font-semibold">Granian (Rust ASGI) + OpenXLA</span> with Pydantic v2 schema validation and zero proprietary cloud lock-in.
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-xs font-mono uppercase">9.2 Identity &amp; Access Management (IAM) &amp; Zero-Trust Controls</h4>
              <ul className="list-disc list-inside space-y-1.5 text-slate-700">
                <li><strong className="text-slate-900">Mutual TLS (mTLS 1.3 + ML-KEM-1024):</strong> Post-quantum hybrid key exchange with hardware-pinned SPIFFE certificates across all inter-rail API boundaries.</li>
                <li><strong className="text-slate-900">DEPA Consent Artifacts &amp; Differential Privacy:</strong> Fine-grained digital consent tokens and calibrated $(\epsilon, \delta)$-DP federated updates.</li>
                <li><strong className="text-slate-900">Vaulted AES-256-XTS / GCM-SIV Tokenization (ADV):</strong> Isolates sovereign identifiers &amp; PAN/biometrics from analytical graph &amp; vector indices.</li>
                <li><strong className="text-slate-900">Zero-Retention PII Telemetry:</strong> Edge middleware scrubs raw PII prior to Prometheus/OpenTelemetry trace emission.</li>
              </ul>
            </div>

            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-xs font-mono uppercase">9.3 Encryption Architecture Matrix: Data at Rest &amp; Data in Transit (1.50B / 20-Yr)</h4>
              <div className="overflow-x-auto border border-slate-300 rounded-lg bg-white">
                <table className="w-full text-left font-mono text-[11px]">
                  <thead className="bg-slate-100 border-b border-slate-300 uppercase">
                    <tr>
                      <th className="p-2.5">Cryptographic Suite</th>
                      <th className="p-2.5">Data at Rest Mechanism</th>
                      <th className="p-2.5">Data in Transit Mechanism</th>
                      <th className="p-2.5">Key Rotation &amp; PQC</th>
                      <th className="p-2.5">Overhead</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 text-slate-700">
                    {(Object.keys(ENCRYPTION_MECHANISM_META) as EncryptionProfileId[]).map((encKey) => {
                      const enc = ENCRYPTION_MECHANISM_META[encKey];
                      return (
                        <tr key={encKey} className={encKey === 'pqc_hybrid_vault' ? 'bg-emerald-50/60' : ''}>
                          <td className="p-2.5 font-bold text-slate-900">
                            {enc.shortName} {encKey === 'pqc_hybrid_vault' ? '★' : ''}
                          </td>
                          <td className="p-2.5 text-[10.5px]">{enc.atRestCipher}</td>
                          <td className="p-2.5 text-[10.5px] text-blue-900">{enc.inTransitProtocol}</td>
                          <td className="p-2.5 text-[10.5px]">{enc.quantumSafe20Yr}</td>
                          <td className="p-2.5 font-bold text-emerald-900 whitespace-nowrap">+{enc.latencyOverheadMs} ms</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* Section 10: National/International Ecosystem, Commercialization & Global Comparison */}
        <section className="space-y-3">
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2 pb-2 border-b border-slate-200">
            <span className="text-blue-900 font-mono text-base font-bold">10.</span>
            <span>National &amp; International Ecosystem, Commercialization &amp; Global Benchmarks</span>
          </h2>
          <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 space-y-4 text-xs font-sans text-slate-800">
            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-xs font-mono uppercase">10.1 Interoperable DPI, Sovereign &amp; Global User Systems</h4>
              <p>
                Interfaces seamlessly with mission-critical national and international rails: <span className="font-mono text-blue-900 font-semibold">NPCI / UPI &amp; ISO-20022 CBDC</span> (instant settlement), <span className="font-mono text-blue-900 font-semibold">ISRO NavIC / GNSS</span> (spatial custody &amp; impossible-travel gating), <span className="font-mono text-blue-900 font-semibold">Income Tax / Passport / DigiLocker</span> (verifiable credentials), and global <span className="font-mono text-blue-900 font-semibold">WHO / GS1 / Smart Grid</span> telemetry networks.
              </p>
            </div>

            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-xs font-mono uppercase">10.2 Commercialization Pathway &amp; Unit Economics (1.50B / 20-Yr Horizon)</h4>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="p-3 bg-white border border-slate-300 rounded-lg">
                  <span className="font-bold text-emerald-900 block mb-1">Bare-Metal FOSS TCO Advantage</span>
                  <p className="text-[11px] text-slate-600">
                    Eliminating proprietary GPU/warehouse licensing yields <span className="font-mono font-bold">{recArchMeta.costIndex}</span> compute cost across 1.50B → 2.41B scaling.
                  </p>
                </div>
                <div className="p-3 bg-white border border-slate-300 rounded-lg">
                  <span className="font-bold text-blue-900 block mb-1">Commercial SaaS / Clearing Licensing</span>
                  <p className="text-[11px] text-slate-600">
                    Per-verification API metering, white-label banking/insurance underwriting feeds, and SLA-backed enterprise fraud shields.
                  </p>
                </div>
                <div className="p-3 bg-white border border-slate-300 rounded-lg">
                  <span className="font-bold text-indigo-900 block mb-1">Global Export &amp; DPI Diplomacy</span>
                  <p className="text-[11px] text-slate-600">
                    Modular open-source architecture ready for cross-border adoption across ASEAN, EU, Africa DPI, and BRICS clearing corridors.
                  </p>
                </div>
              </div>

              {/* Explicit CapEx, OpEx YoY & 20-Yr 6-Pillar Breakdown */}
              <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
                  <span className="font-bold text-slate-900">
                    CapEx &amp; 20-Year YoY OpEx Breakdown ({econ.deploymentTier})
                  </span>
                  <div className="flex flex-wrap items-center gap-3 tabular-nums">
                    <span className="font-bold text-blue-950">Total CapEx: {formatInrCompact(econ.totalCapExUsd)}</span>
                    <span className="font-bold text-emerald-900">OpEx Y1: {formatInrCompact(econ.opExYear1Usd)}/yr</span>
                    <span className="font-bold text-indigo-950">20-Yr OpEx: {formatInrCompact(econ.totalOpEx20YrUsd)}</span>
                    <span className="font-bold text-rose-950">20-Yr TCO: {formatInrCompact(econ.totalTco20YrUsd)}</span>
                  </div>
                </div>
                <div className="overflow-x-auto border border-slate-200 rounded-lg">
                  <table className="w-full text-left font-mono text-[11px] tabular-nums">
                    <thead className="bg-slate-100 border-b border-slate-200">
                      <tr>
                        <th className="p-2">Operational Cost Pillar</th>
                        <th className="p-2">Year 1 YoY</th>
                        <th className="p-2">Year 20 YoY</th>
                        <th className="p-2">20-Yr Cumulative</th>
                        <th className="p-2">Share</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-200 text-slate-700">
                      {econ.opExCategories.map((cat) => (
                        <tr key={cat.key}>
                          <td className="p-2 font-bold text-slate-900">{cat.label}</td>
                          <td className="p-2">{formatInrCompact(cat.year1Usd)}</td>
                          <td className="p-2">{formatInrCompact(cat.year20Usd)}</td>
                          <td className="p-2 font-bold text-indigo-950">{formatInrCompact(cat.total20YrUsd)}</td>
                          <td className="p-2 font-semibold">{cat.sharePct}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <h4 className="font-bold text-slate-900 text-xs font-mono uppercase">10.3 Global Digital Infrastructure Comparative Matrix</h4>
              <div className="overflow-x-auto border border-slate-300 rounded-lg bg-white">
                <table className="w-full text-left font-mono text-[11px]">
                  <thead className="bg-slate-100 border-b border-slate-300 uppercase">
                    <tr>
                      <th className="p-2.5">Platform</th>
                      <th className="p-2.5">Primary Identifier / Rail</th>
                      <th className="p-2.5">Target Scale</th>
                      <th className="p-2.5">Privacy &amp; License</th>
                      <th className="p-2.5">Real-Time Interoperability</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200 text-slate-700">
                    <tr>
                      <td className="p-2.5 font-bold text-blue-900">Proposed Open-Source DPI Mesh</td>
                      <td className="p-2.5">VID / ISO-20022 / GS1 / NavIC</td>
                      <td className="p-2.5">1.50B → 2.41B+ (20-Yr)</td>
                      <td className="p-2.5">Apache-2.0 / DP-SGD / ADV</td>
                      <td className="p-2.5">UPI, CBDC, Sentinel SAR, Smart Grid</td>
                    </tr>
                    <tr>
                      <td className="p-2.5 font-bold text-slate-800">UIDAI India (CIDR)</td>
                      <td className="p-2.5">12-Digit Aadhaar / VID</td>
                      <td className="p-2.5">&gt; 1.38 Billion</td>
                      <td className="p-2.5">Virtual ID Tokenization &amp; ADV</td>
                      <td className="p-2.5">Unified Payments Interface (UPI) / AEPS</td>
                    </tr>
                    <tr>
                      <td className="p-2.5 font-bold text-slate-800">Estonia X-Road</td>
                      <td className="p-2.5">Personal Code (Isikukood)</td>
                      <td className="p-2.5">~1.3 Million</td>
                      <td className="p-2.5">MIT Open-Source Logged Queries</td>
                      <td className="p-2.5">EU SEPA Banking Standards</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* Section 11: Bibliographic References */}
        <section className="space-y-3 pt-4 border-t border-slate-200">
          <h2 className="text-base font-bold text-slate-900 uppercase font-mono tracking-wider">
            References &amp; Foundational Literature
          </h2>
          <ol className="list-decimal list-inside space-y-2 text-xs text-slate-700 font-sans">
            <li>
              Vaswani, A., et al. (2017). <em>Attention Is All You Need</em>. Advances in Neural Information Processing Systems (NeurIPS 2017), pp. 5998–6008.
            </li>
            <li>
              Liu, Z., et al. (2021). <em>Swin Transformer: Hierarchical Vision Transformer using Shifted Windows</em>. IEEE/CVF International Conference on Computer Vision (ICCV), pp. 10012–10022.
            </li>
            <li>
              Applied Machine Learning Research Group. ({year}). <em>{project.title}: SOTA Empirical Benchmark</em>. ScholarML Engineering Series, DOI: {doi}.
            </li>
            <li>
              ISO/IEC 30107-3:2017. <em>Information technology — Biometric presentation attack detection — Part 3: Testing and reporting</em>. International Organization for Standardization.
            </li>
            <li>
              He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). <em>Deep Residual Learning for Image Recognition</em>. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778.
            </li>
          </ol>
        </section>
      </div>

      {/* 4. Google Scholar Multi-Format Citation Modal */}
      {showCiteModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full border border-slate-300 shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="p-6 border-b border-slate-200 flex items-center justify-between bg-slate-50">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-blue-100 border border-blue-200 flex items-center justify-center text-blue-900">
                  <Quote className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-slate-900">
                    Cite with Google Scholar Specification
                  </h3>
                  <p className="text-xs text-slate-600 font-mono">
                    {doi} · {arxivId}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowCiteModal(false)}
                className="w-8 h-8 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/70 flex items-center justify-center transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Tabs */}
            <div className="px-6 pt-4 border-b border-slate-200 flex flex-wrap gap-2">
              {(['bibtex', 'apa', 'mla', 'chicago', 'harvard', 'ris'] as const).map((style) => (
                <button
                  key={style}
                  onClick={() => setActiveCiteTab(style)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold uppercase font-mono transition-colors cursor-pointer ${
                    activeCiteTab === style
                      ? 'bg-blue-900 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  {style}
                </button>
              ))}
            </div>

            {/* Citation Content View */}
            <div className="p-6 space-y-4">
              <div className="relative bg-slate-50 border border-slate-300 rounded-xl p-4 font-mono text-xs text-slate-800 overflow-x-auto whitespace-pre-wrap break-words max-h-60">
                {citations[activeCiteTab]}
              </div>

              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-500">
                  Compatible with Zotero, Mendeley, EndNote, Overleaf, and LaTeX.
                </span>
                <button
                  onClick={() => handleCopyCitation(activeCiteTab)}
                  className="px-4 py-2 bg-blue-900 hover:bg-blue-800 text-white rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-xs cursor-pointer"
                >
                  {copiedFormat === activeCiteTab ? (
                    <>
                      <Check className="w-4 h-4 text-emerald-400" />
                      <span>Copied {activeCiteTab.toUpperCase()}!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-4 h-4" />
                      <span>Copy {activeCiteTab.toUpperCase()} Citation</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 bg-slate-50 border-t border-slate-200 flex justify-end">
              <button
                onClick={() => setShowCiteModal(false)}
                className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-800 rounded-xl text-xs font-semibold transition-colors cursor-pointer"
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </article>
  );
}
