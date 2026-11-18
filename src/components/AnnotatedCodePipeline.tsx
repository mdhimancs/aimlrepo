import React from 'react';
import { Code2, Copy, Check, FileText, Cpu, Layers } from 'lucide-react';
import { FlagshipProject } from '../types';

interface AnnotatedCodePipelineProps {
  project: FlagshipProject;
  bannerTitle: string;
  bannerSubtitle: string;
  copiedCodeIndex: number | null;
  copiedAllId: string | null;
  indexOffset: number;
  onCopyCode: (code: string, index: number) => void;
  onCopyAll: (project: FlagshipProject) => void;
}

function buildAnnotatedCode(
  project: FlagshipProject,
  step: FlagshipProject['pipeline'][number],
  stepIdx: number
): string {
  const commentChar = step.language === 'rust' ? '//' : '#';
  const trimmed = step.code.trim();
  const alreadyHasBanner =
    trimmed.startsWith(`${commentChar} ===`) || trimmed.startsWith(`${commentChar} ---`);

  if (alreadyHasBanner) return step.code;

  const header = [
    `${commentChar} =========================================================================`,
    `${commentChar} MODULE 0${stepIdx + 1}: ${step.title}`,
    `${commentChar} PROJECT: ${project.title} (${project.domain})`,
    `${commentChar} DESCRIPTION: ${step.desc}`,
    `${commentChar} TARGET RUNTIME: ${
      step.language === 'rust'
        ? 'Rust 2021 (PyO3 / Rayon Native Kernel)'
        : 'Python 3.10+ (Production Pipeline)'
    }`,
    `${commentChar} BENCHMARK METRICS: ${project.metrics}`,
    `${commentChar} =========================================================================`,
    ''
  ].join('\n');

  return header + step.code;
}

function extractLineAnnotations(code: string, language?: 'python' | 'rust') {
  const lines = code.split('\n');
  const notes: { label: string; detail: string }[] = [];

  if (lines.some((l) => l.includes('import ') || l.includes('use '))) {
    notes.push({
      label: 'Dependencies & Imports',
      detail:
        language === 'rust'
          ? 'Loads zero-copy tensor/ndarray crates and parallel thread-pool bindings.'
          : 'Initializes numerical computing, preprocessing, and model runtime packages.'
    });
  }
  if (
    lines.some(
      (l) =>
        l.includes('read_csv') ||
        l.includes('DataLoader') ||
        l.includes('train_test_split') ||
        l.includes('StandardScaler') ||
        l.includes('Tokenizer')
    )
  ) {
    notes.push({
      label: 'Data Ingestion & Transform',
      detail: 'Loads raw records, sanitizes missing/noisy inputs, and splits/normalizes features without leakage.'
    });
  }
  if (
    lines.some(
      (l) =>
        l.includes('class ') ||
        l.includes('fn ') ||
        l.includes('def ') ||
        l.includes('.fit(') ||
        l.includes('forward(')
    )
  ) {
    notes.push({
      label: 'Core Model / Kernel Logic',
      detail: 'Defines architecture forward pass, objective optimization, or multi-threaded kernel execution.'
    });
  }
  if (
    lines.some(
      (l) =>
        l.includes('joblib.dump') ||
        l.includes('torch.save') ||
        l.includes('export') ||
        l.includes('roc_auc') ||
        l.includes('classification_report') ||
        l.includes('evaluate')
    )
  ) {
    notes.push({
      label: 'Evaluation & Artifact Export',
      detail: 'Computes validation benchmarks and serializes weights/pipelines for production deployment.'
    });
  }

  return notes;
}

export default function AnnotatedCodePipeline({
  project,
  bannerTitle,
  bannerSubtitle,
  copiedCodeIndex,
  copiedAllId,
  indexOffset,
  onCopyCode,
  onCopyAll
}: AnnotatedCodePipelineProps) {
  return (
    <div className="space-y-8">
      {/* Top Pipeline Banner */}
      <div className="bg-slate-200/90 hover:bg-slate-300/80 p-5 rounded-2xl border border-slate-300 flex flex-col md:flex-row md:items-center justify-between gap-4 transition-all duration-200 shadow-xs">
        <div>
          <div className="flex items-center gap-2 text-rose-900 font-bold text-xs uppercase tracking-wider mb-1 font-mono">
            <Code2 className="w-4 h-4 text-rose-900" />
            <span>{bannerTitle}</span>
          </div>
          <p className="text-slate-800 text-xs leading-relaxed max-w-2xl font-medium">
            {bannerSubtitle}
          </p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={() => onCopyAll(project)}
            className="px-4 py-2.5 bg-blue-900 hover:bg-blue-800 text-white rounded-xl text-xs font-semibold flex items-center gap-2 shadow-xs transition-colors"
          >
            {copiedAllId === project.id ? (
              <>
                <Check className="w-4 h-4 text-emerald-400" />
                <span className="text-emerald-300 font-bold">Copied Complete Script!</span>
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                <span>Copy Complete Code Pipeline</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Annotated Steps */}
      {project.pipeline.map((step, idx) => {
        const annotatedCode = buildAnnotatedCode(project, step, idx);
        const annotations = extractLineAnnotations(step.code, step.language);
        const copyIdx = indexOffset + idx;

        return (
          <div key={idx} className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Description & Code Annotations */}
            <div className="lg:col-span-5 bg-slate-100 hover:bg-slate-200/90 border border-slate-300 rounded-2xl p-5 space-y-4 transition-all duration-200 shadow-xs">
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-[11px] font-mono font-bold px-2 py-0.5 rounded bg-blue-100 text-blue-900 border border-blue-300">
                    STEP 0{idx + 1}
                  </span>
                  {step.language === 'rust' ? (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-amber-100 text-amber-900 border border-amber-300 font-mono font-bold">
                      RUST ACCELERATED
                    </span>
                  ) : (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 border border-emerald-300 font-mono font-bold">
                      PYTHON PIPELINE
                    </span>
                  )}
                </div>
                <h3 className="text-base font-bold text-slate-900">{step.title}</h3>
                <p className="text-slate-700 text-xs leading-relaxed">{step.desc}</p>
              </div>

              {/* Structural Code Annotations */}
              <div className="pt-3 border-t border-slate-300 space-y-2.5">
                <div className="flex items-center gap-1.5 text-[11px] font-mono font-bold uppercase text-rose-900">
                  <FileText className="w-3.5 h-3.5" />
                  <span>Code Implementation Breakdown</span>
                </div>
                <div className="space-y-2">
                  {annotations.map((note, nIdx) => (
                    <div
                      key={nIdx}
                      className="bg-white p-2.5 rounded-xl border border-slate-300 text-xs space-y-0.5"
                    >
                      <span className="font-mono font-bold text-blue-900 block text-[11px]">
                        • {note.label}
                      </span>
                      <span className="text-slate-700 text-[11px] leading-snug block">
                        {note.detail}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="pt-2 border-t border-slate-300 flex items-center justify-between text-[11px] font-mono text-slate-600">
                <span className="flex items-center gap-1">
                  <Cpu className="w-3 h-3 text-blue-900" />
                  {project.performanceIndicators.hardwareTarget}
                </span>
                <span className="flex items-center gap-1">
                  <Layers className="w-3 h-3 text-emerald-900" />
                  {annotatedCode.split('\n').length} lines
                </span>
              </div>
            </div>

            {/* Right Annotated Source Viewer */}
            <div className="lg:col-span-7 bg-slate-900 rounded-2xl border border-slate-800 overflow-hidden shadow-sm">
              <div className="bg-slate-950 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                <span className="text-xs font-mono text-slate-300">
                  {step.language === 'rust'
                    ? `src/0${idx + 1}_kernel.rs`
                    : `src/0${idx + 1}_pipeline.py`}{' '}
                  <span className="text-slate-500 ml-2">// Annotated Source</span>
                </span>
                <button
                  onClick={() => onCopyCode(annotatedCode, copyIdx)}
                  className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white transition-colors px-2.5 py-1 rounded bg-slate-800"
                >
                  {copiedCodeIndex === copyIdx ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-400" />
                      <span className="text-emerald-400 font-semibold">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy Annotated Code</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="p-4 text-xs font-mono text-indigo-200 overflow-x-auto whitespace-pre-wrap break-words leading-relaxed max-h-[460px]">
                <code>{annotatedCode}</code>
              </pre>
            </div>
          </div>
        );
      })}
    </div>
  );
}
