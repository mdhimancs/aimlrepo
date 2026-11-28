import React, { useState, useMemo, useEffect } from 'react';
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
  Layers,
  X,
  Zap,
  Plus,
  Minus,
  BookOpen,
  Play
} from 'lucide-react';
import { FlagshipProject } from '../types';
import {
  RepoFile,
  buildRepoFiles,
  downloadProjectGitZip,
  downloadProjectNotebook,
  buildColabNotebookCells,
  buildColabPercentScript,
  buildMonorepoBundle,
  generateGitPushCommands
} from '../utils/gitRepoBuilder';

interface CodeRepoStructureTabProps {
  flagshipProjects: FlagshipProject[];
  datacampProjects: FlagshipProject[];
  initialProjectId?: string;
}

// VS Code style syntax highlighter helper
function highlightCodeToJSX(code: string) {
  const lines = code.split('\n');
  return lines.map((line, lineIdx) => {
    if (line.trim().startsWith('#') || line.trim().startsWith('//')) {
      return (
        <div key={lineIdx} className="whitespace-pre text-slate-600 italic">
          {line}
        </div>
      );
    }

    const regex = /(\b(?:import|from|def|class|return|pub|fn|struct|impl|use|if|else|for|in|self|super|let|mut|async|await|const|var|pub|crate|mod)\b|\b(?:Tensor|Module|HashMap|String|Result|Option|Self|Vec|f32|u64|u32|int|float|str|bool|torch|nn|F|rayon)\b|"[^"]*"|'[^']*'|\b\d+(?:\.\d+)?\b|#.*|\/\/.*)/g;

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;

    while ((match = regex.exec(line)) !== null) {
      if (match.index > lastIndex) {
        parts.push(line.substring(lastIndex, match.index));
      }

      const matchedText = match[0];

      if (matchedText.startsWith('#') || matchedText.startsWith('//')) {
        parts.push(<span key={match.index} className="text-slate-600 italic">{matchedText}</span>);
      } else if (matchedText.startsWith('"') || matchedText.startsWith("'")) {
        parts.push(<span key={match.index} className="text-emerald-800 font-medium">{matchedText}</span>);
      } else if (!isNaN(Number(matchedText))) {
        parts.push(<span key={match.index} className="text-amber-800 font-medium">{matchedText}</span>);
      } else if (/^(?:Tensor|Module|HashMap|String|Result|Option|Self|Vec|f32|u64|u32|int|float|str|bool|torch|nn|F|rayon)$/.test(matchedText)) {
        parts.push(<span key={match.index} className="text-sky-800 font-semibold">{matchedText}</span>);
      } else {
        parts.push(<span key={match.index} className="text-indigo-800 font-semibold">{matchedText}</span>);
      }

      lastIndex = regex.lastIndex;
    }

    if (lastIndex < line.length) {
      parts.push(line.substring(lastIndex));
    }

    return (
      <div key={lineIdx} className="whitespace-pre">
        {parts.length > 0 ? parts : line}
      </div>
    );
  });
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
    notebooks: true,
    github: true,
    configs: true,
    src: true,
    tests: true,
    root: true
  });
  const [selectedFilePath, setSelectedFilePath] = useState<string>(
    `notebooks/${(initialProjectId || allProjects[0]?.id || '')}_colab.ipynb`
  );
  const [notebookViewMode, setNotebookViewMode] = useState<'cells' | 'json'>('cells');
  const [leftSidebarTab, setLeftSidebarTab] = useState<'tree' | 'topology'>('tree');
  const [copiedCellId, setCopiedCellId] = useState<string | null>(null);
  const [copiedFile, setCopiedFile] = useState(false);
  const [copiedBundle, setCopiedBundle] = useState(false);
  const [copiedGitScript, setCopiedGitScript] = useState(false);
  const [isDownloadingZip, setIsDownloadingZip] = useState(false);
  const [showGitModal, setShowGitModal] = useState(false);
  // Default font size 2 sizes bigger (default 16px) with +/- adjusters
  const [codeFontSize, setCodeFontSize] = useState<number>(16);

  useEffect(() => {
    if (initialProjectId) {
      setSelectedProjectId(initialProjectId);
      setSelectedFilePath(`notebooks/${initialProjectId}_colab.ipynb`);
    }
  }, [initialProjectId]);

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

  const notebookCells = useMemo(() => {
    return activeProject ? buildColabNotebookCells(activeProject) : [];
  }, [activeProject]);

  const folders = useMemo(() => {
    return {
      notebooks: repoFiles.filter((f) => f.folder === 'notebooks'),
      github: repoFiles.filter((f) => f.folder === '.github' || f.path.startsWith('.github/')),
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
    setSelectedFilePath(`notebooks/${id}_colab.ipynb`);
  };

  const handleCopyCell = (cellId: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCellId(cellId);
    setTimeout(() => setCopiedCellId(null), 1800);
  };

  const handleCopyFile = () => {
    if (!activeFile) return;
    navigator.clipboard.writeText(activeFile.content);
    setCopiedFile(true);
    setTimeout(() => setCopiedFile(false), 2000);
  };

  const handleCopyBundle = () => {
    if (!activeProject) return;
    navigator.clipboard.writeText(buildMonorepoBundle(activeProject));
    setCopiedBundle(true);
    setTimeout(() => setCopiedBundle(false), 2200);
  };

  const handleCopyGitScript = () => {
    if (!activeProject) return;
    navigator.clipboard.writeText(generateGitPushCommands(activeProject));
    setCopiedGitScript(true);
    setTimeout(() => setCopiedGitScript(false), 2200);
  };

  const handleDownloadZip = async () => {
    if (!activeProject) return;
    setIsDownloadingZip(true);
    try {
      await downloadProjectGitZip(activeProject);
    } catch (err) {
      console.error('Failed to create Git zip package:', err);
    } finally {
      setIsDownloadingZip(false);
    }
  };

  const asciiTree = useMemo(() => {
    if (!activeProject) return '';
    return [
      `${activeProject.id}/`,
      `├── notebooks/`,
      ...folders.notebooks.map((f, i) => `│   ${i === folders.notebooks.length - 1 ? '└──' : '├──'} ${f.name}`),
      `├── .github/`,
      `│   └── workflows/`,
      ...folders.github.map((f, i) => `│       ${i === folders.github.length - 1 ? '└──' : '├──'} ${f.name}`),
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
        <div className="flex flex-col gap-4">
          <div className="space-y-1.5">
            <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
              Interactive Code Repository &amp; Colab / Jupyter Notebook Studio
            </h2>
            <p className="text-slate-700 text-sm max-w-4xl leading-relaxed">
              Every project includes a runnable <code className="font-mono text-blue-900 font-semibold">.ipynb</code> notebook formatted for Google Colab &amp; Jupyter alongside a full Git-ready repository (.zip). Download the notebook directly or push the whole repo to GitHub.
            </p>
          </div>

          {/* Action Toolbar (Last Line - Compact Size) */}
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={() => downloadProjectNotebook(activeProject)}
              className="px-2 py-1 bg-amber-600 hover:bg-amber-500 text-white rounded-md text-[10px] font-semibold flex items-center gap-1 transition-colors shadow-xs cursor-pointer"
              title="Download .ipynb notebook ready to upload directly to Google Colab or Jupyter"
            >
              <BookOpen className="w-3 h-3" />
              <span>Download .ipynb (Colab / Jupyter)</span>
            </button>

            <button
              onClick={handleDownloadZip}
              disabled={isDownloadingZip}
              className="px-2 py-1 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-60 text-white rounded-md text-[10px] font-semibold flex items-center gap-1 transition-colors shadow-xs cursor-pointer"
            >
              <Download className="w-3 h-3" />
              <span>{isDownloadingZip ? 'Packaging ZIP...' : 'Download Git Repo (.zip)'}</span>
            </button>

            <button
              onClick={() => setShowGitModal(true)}
              className="px-2 py-1 bg-blue-900 hover:bg-blue-800 text-white rounded-md text-[10px] font-semibold flex items-center gap-1 transition-colors shadow-xs cursor-pointer"
            >
              <GitBranch className="w-3 h-3 text-emerald-300" />
              <span>Git &amp; Colab Guide</span>
            </button>

            <button
              onClick={handleCopyGitScript}
              className="px-2 py-1 bg-slate-200 hover:bg-slate-300 text-slate-900 border border-slate-300 rounded-md text-[10px] font-semibold flex items-center gap-1 transition-colors shadow-xs cursor-pointer"
            >
              {copiedGitScript ? (
                <>
                  <Check className="w-3 h-3 text-emerald-700" />
                  <span className="text-emerald-800 font-bold">Copied CLI!</span>
                </>
              ) : (
                <>
                  <Terminal className="w-3 h-3 text-indigo-700" />
                  <span>Copy Push Commands</span>
                </>
              )}
            </button>

            <button
              onClick={handleCopyBundle}
              className="px-2 py-1 bg-slate-200 hover:bg-slate-300 text-slate-800 border border-slate-300 rounded-md text-[10px] font-semibold flex items-center gap-1 transition-colors shadow-xs cursor-pointer"
              title="Copy full Colab/Jupyter cell-delimited (# %%) monorepo bundle"
            >
              {copiedBundle ? (
                <>
                  <Check className="w-3 h-3 text-emerald-700" />
                  <span className="text-emerald-800 font-bold">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy Bundle</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Project Repository Selector */}
        <div className="mt-6 pt-5 border-t border-slate-300 flex flex-col md:flex-row gap-4 items-stretch md:items-center justify-between">
          <div className="flex-1 flex flex-col sm:flex-row sm:items-center gap-3">
            <span className="text-xs font-semibold text-slate-800 flex items-center gap-1.5 shrink-0">
              <GitBranch className="w-4 h-4 text-blue-900" />
              <span>Active Repository ({allProjects.length} Total):</span>
            </span>
            <select
              value={activeProject.id}
              onChange={(e) => handleSelectProject(e.target.value)}
              className="flex-1 bg-white border border-slate-300 rounded-xl px-4 py-2 text-xs font-semibold text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
            >
              <optgroup label={`Storyboard — Starred (${flagshipProjects.length})`}>
                {flagshipProjects
                  .filter((p) => filteredProjects.includes(p))
                  .map((p) => (
                    <option key={p.id} value={p.id}>
                      ★ [{p.difficulty}] {p.title}
                    </option>
                  ))}
              </optgroup>
              <optgroup label={`ML Case Studies Catalog (${datacampProjects.length} Unpinned · ${allProjects.length} Total)`}>
                {datacampProjects
                  .filter((p) => filteredProjects.includes(p))
                  .map((p) => (
                    <option key={p.id} value={p.id}>
                      [{p.sourceName || 'Case Study'} · {p.difficulty}] {p.title}
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
        {/* Left Sidebar: Interactive Directory Structure + Repository Topology Tabs */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-slate-100 border border-slate-300 rounded-2xl p-3.5 shadow-xs space-y-3">
            {/* Sidebar Mode Tabs */}
            <div className="grid grid-cols-2 gap-1 bg-slate-200/80 p-1 rounded-xl border border-slate-300">
              <button
                type="button"
                onClick={() => setLeftSidebarTab('tree')}
                className={`px-2.5 py-1.5 rounded-lg text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                  leftSidebarTab === 'tree'
                    ? 'bg-white text-blue-900 shadow-2xs border border-slate-300'
                    : 'text-slate-700 hover:text-slate-900'
                }`}
              >
                <FolderOpen className="w-3.5 h-3.5 shrink-0" />
                <span>Directory ({repoFiles.length})</span>
              </button>
              <button
                type="button"
                onClick={() => setLeftSidebarTab('topology')}
                className={`px-2.5 py-1.5 rounded-lg text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                  leftSidebarTab === 'topology'
                    ? 'bg-white text-rose-900 shadow-2xs border border-slate-300'
                    : 'text-slate-700 hover:text-slate-900'
                }`}
              >
                <Layers className="w-3.5 h-3.5 shrink-0" />
                <span>Topology (ASCII Map)</span>
              </button>
            </div>

            {leftSidebarTab === 'tree' ? (
              <div>
                <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-300">
                  <div className="flex items-center gap-1.5 text-xs font-mono font-bold text-slate-900 truncate">
                    <FolderOpen className="w-3.5 h-3.5 text-blue-900 shrink-0" />
                    <span className="truncate">{activeProject.id}/</span>
                  </div>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-900 border border-emerald-300 font-bold shrink-0">
                    {repoFiles.length} Files
                  </span>
                </div>

                <div className="space-y-0.5 text-xs font-mono">
                  {(['notebooks', 'github', 'configs', 'src', 'tests'] as const).map((folderKey) => {
                    const displayName = folderKey === 'github' ? '.github/workflows/' : `${folderKey}/`;
                    const folderFiles = folders[folderKey];
                    if (!folderFiles || folderFiles.length === 0) return null;

                    return (
                      <div key={folderKey} className="space-y-0.5">
                        <button
                          onClick={() => toggleFolder(folderKey)}
                          className="w-full flex items-center gap-1.5 px-1.5 py-1 rounded-md text-slate-800 hover:bg-slate-200 font-semibold transition-colors text-left"
                        >
                          {expandedFolders[folderKey] ? (
                            <ChevronDown className="w-3 h-3 text-slate-500 shrink-0" />
                          ) : (
                            <ChevronRight className="w-3 h-3 text-slate-500 shrink-0" />
                          )}
                          {expandedFolders[folderKey] ? (
                            <FolderOpen className="w-3.5 h-3.5 text-blue-900 shrink-0" />
                          ) : (
                            <Folder className="w-3.5 h-3.5 text-blue-900 shrink-0" />
                          )}
                          <span className="truncate">{displayName}</span>
                        </button>

                        {expandedFolders[folderKey] && (
                          <div className="pl-3 space-y-0.5 border-l border-slate-300 ml-2 my-0.5">
                            {folderFiles.map((file, fIdx) => {
                              const isSelected = activeFile?.path === file.path;
                              const isLast = fIdx === folderFiles.length - 1;
                              return (
                                <button
                                  key={file.path}
                                  onClick={() => setSelectedFilePath(file.path)}
                                  className={`w-full flex items-center gap-1.5 px-2 py-1 rounded-md text-left transition-colors ${
                                    isSelected
                                      ? 'bg-blue-900 text-white font-bold shadow-xs'
                                      : 'text-slate-700 hover:bg-slate-200'
                                  }`}
                                >
                                  <span className="text-slate-400 font-mono select-none text-[11px]">
                                    {isLast ? '└──' : '├──'}
                                  </span>
                                  <FileCode
                                    className={`w-3 h-3 shrink-0 ${
                                      isSelected ? 'text-white' : 'text-blue-900'
                                    }`}
                                  />
                                  <span className="truncate">{file.name}</span>
                                </button>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    );
                  })}

                  {/* Root Directory Files */}
                  <div className="pt-2 mt-2 border-t border-slate-300 space-y-0.5">
                    <span className="text-[10px] uppercase font-bold text-slate-500 px-1.5 block mb-1">
                      Root Files ({folders.root.length})
                    </span>
                    <div className="pl-1.5 space-y-0.5 border-l border-slate-300 ml-1.5">
                      {folders.root.map((file, rIdx) => {
                        const isSelected = activeFile?.path === file.path;
                        const isLast = rIdx === folders.root.length - 1;
                        return (
                          <button
                            key={file.path}
                            onClick={() => setSelectedFilePath(file.path)}
                            className={`w-full flex items-center gap-1.5 px-2 py-1 rounded-md text-left transition-colors ${
                              isSelected
                                ? 'bg-blue-900 text-white font-bold shadow-xs'
                                : 'text-slate-700 hover:bg-slate-200'
                            }`}
                          >
                            <span className="text-slate-400 font-mono select-none text-[11px]">
                              {isLast ? '└──' : '├──'}
                            </span>
                            <FileText
                              className={`w-3 h-3 shrink-0 ${
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
              </div>
            ) : (
              <div>
                <div className="flex items-center gap-2 text-xs font-bold text-slate-900 mb-2.5">
                  <Layers className="w-4 h-4 text-rose-900" />
                  <span>Repository Topology (ASCII Map)</span>
                </div>
                <pre className="p-3.5 bg-white border border-slate-300 rounded-xl text-[11px] font-mono text-slate-800 overflow-x-auto whitespace-pre-wrap break-words leading-relaxed">
                  {asciiTree}
                </pre>
              </div>
            )}
          </div>
        </div>

        {/* Right Main Panel: Active File Source Viewer */}
        <div className="lg:col-span-8 space-y-4">
          {/* Active File Metadata Bar */}
          {activeFile && (
            <div className="bg-slate-100 hover:bg-slate-200/80 border border-slate-300 rounded-2xl px-5 py-4 transition-colors shadow-xs">
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-2 text-xs font-mono text-blue-900 font-bold min-w-0">
                  <Terminal className="w-4 h-4 text-rose-900 shrink-0" />
                  <span className="truncate">{activeProject.id}/{activeFile.path}</span>
                  <span className="px-2 py-0.5 rounded bg-slate-200 text-slate-800 border border-slate-300 text-[10px] uppercase shrink-0">
                    {activeFile.language}
                  </span>
                </div>

                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-1.5">
                    {activeFile.language === 'ipynb' && (
                      <div className="inline-flex rounded-md bg-slate-200 p-0.5 border border-slate-300 text-[10px] font-semibold">
                        <button
                          type="button"
                          onClick={() => setNotebookViewMode('cells')}
                          className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${
                            notebookViewMode === 'cells'
                              ? 'bg-amber-600 text-white shadow-2xs'
                              : 'text-slate-700 hover:text-slate-900'
                          }`}
                        >
                          Colab / Jupyter Cells
                        </button>
                        <button
                          type="button"
                          onClick={() => setNotebookViewMode('json')}
                          className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${
                            notebookViewMode === 'json'
                              ? 'bg-blue-900 text-white shadow-2xs'
                              : 'text-slate-700 hover:text-slate-900'
                          }`}
                        >
                          Raw .ipynb JSON
                        </button>
                      </div>
                    )}

                    {activeFile.language === 'python' && (
                      <button
                        type="button"
                        onClick={() => handleCopyCell(`pct-${activeFile.path}`, buildColabPercentScript(activeProject))}
                        className="px-2 py-1 bg-amber-100 hover:bg-amber-200 text-amber-950 border border-amber-300 rounded-md text-[10px] font-semibold flex items-center gap-1 transition-colors cursor-pointer"
                        title="Copy full pipeline formatted with # %% Colab/Jupyter cells"
                      >
                        {copiedCellId === `pct-${activeFile.path}` ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-700" />
                            <span>Copied # %% Cells!</span>
                          </>
                        ) : (
                          <>
                            <BookOpen className="w-3 h-3 text-amber-800" />
                            <span>Copy # %% Colab Script</span>
                          </>
                        )}
                      </button>
                    )}
                  </div>

                  <div className="flex items-center gap-1.5 ml-auto">
                    {/* Font Size Adjusters (+ / -) */}
                    <div className="flex items-center bg-slate-200 border border-slate-300 rounded-md overflow-hidden shadow-xs">
                      <button
                        onClick={() => setCodeFontSize((prev) => Math.max(12, prev - 2))}
                        className="px-1.5 py-1 hover:bg-slate-300 text-slate-800 text-[10px] font-bold flex items-center transition-colors cursor-pointer"
                        title="Decrease font size (-)"
                      >
                        <Minus className="w-3 h-3" />
                      </button>
                      <span className="px-1.5 text-[10px] font-mono font-semibold text-slate-700 select-none">
                        {codeFontSize}px
                      </span>
                      <button
                        onClick={() => setCodeFontSize((prev) => Math.min(24, prev + 2))}
                        className="px-1.5 py-1 hover:bg-slate-300 text-slate-800 text-[10px] font-bold flex items-center transition-colors cursor-pointer"
                        title="Increase font size (+)"
                      >
                        <Plus className="w-3 h-3" />
                      </button>
                    </div>

                    <button
                      onClick={handleCopyFile}
                      className="px-2 py-1 bg-slate-200 hover:bg-slate-300 text-slate-900 border border-slate-300 rounded-md text-[10px] font-semibold flex items-center gap-1 transition-colors shadow-xs cursor-pointer"
                    >
                      {copiedFile ? (
                        <>
                          <Check className="w-3 h-3 text-emerald-700" />
                          <span className="text-emerald-800 font-bold">Copied File!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3 h-3" />
                          <span>Copy File</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Code Window: Colab / Jupyter Notebook Cell Renderer OR VS Code Syntax Highlighting */}
          {activeFile && activeFile.language === 'ipynb' && notebookViewMode === 'cells' ? (
            <div className="bg-slate-300/85 rounded-2xl border border-slate-400/70 overflow-hidden shadow-sm">
              <div className="bg-slate-300 px-5 py-3 border-b border-slate-400/70 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                  <span className="text-xs font-mono text-amber-950 font-semibold">
                    {activeFile.path}
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-100 text-amber-900 border border-amber-300 font-semibold">
                    Google Colab / Jupyter nbformat 4.5
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-[11px] font-mono text-slate-700 font-medium">
                    {notebookCells.length} cells
                  </span>
                  <button
                    type="button"
                    onClick={() => downloadProjectNotebook(activeProject)}
                    className="px-2.5 py-1 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-[11px] font-semibold flex items-center gap-1 transition-colors cursor-pointer"
                  >
                    <Download className="w-3 h-3" />
                    <span>Download .ipynb</span>
                  </button>
                </div>
              </div>
              <div className="p-4 space-y-4 max-h-[680px] overflow-y-auto bg-slate-200">
                {notebookCells.map((cell, cIdx) => (
                  <div
                    key={cell.id}
                    className="rounded-xl border border-slate-400/60 overflow-hidden shadow-2xs bg-slate-300/85"
                  >
                    <div className="px-3.5 py-2 bg-slate-300 border-b border-slate-400/60 flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2 text-[11px] font-mono">
                        {cell.cellType === 'code' ? (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-indigo-100 text-indigo-900 border border-indigo-300 font-bold">
                            <Play className="w-2.5 h-2.5 fill-current" />
                            <span>In [{cIdx + 1}]</span>
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 border border-emerald-300 font-bold">
                            Markdown [{cIdx + 1}]
                          </span>
                        )}
                        {cell.title && (
                          <span className="text-slate-900 font-semibold truncate">{cell.title}</span>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={() => handleCopyCell(cell.id, cell.source)}
                        className="text-[11px] font-mono text-slate-800 hover:text-slate-950 px-2 py-0.5 rounded bg-white hover:bg-slate-50 border border-slate-400/60 flex items-center gap-1 transition-colors cursor-pointer shrink-0"
                      >
                        {copiedCellId === cell.id ? (
                          <>
                            <Check className="w-3 h-3 text-emerald-700" />
                            <span className="text-emerald-800 font-semibold">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span>Copy Cell</span>
                          </>
                        )}
                      </button>
                    </div>
                    <div
                      className={`p-4 font-mono overflow-x-auto whitespace-pre-wrap break-words leading-relaxed bg-slate-300/80 ${
                        cell.cellType === 'markdown' ? 'text-slate-800' : 'text-slate-950'
                      }`}
                      style={{ fontSize: `${Math.max(13, codeFontSize - 1)}px` }}
                    >
                      {cell.cellType === 'code' ? <code>{highlightCodeToJSX(cell.source)}</code> : cell.source}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            activeFile && (
              <div className="bg-slate-300/85 rounded-2xl border border-slate-400/70 overflow-hidden shadow-sm">
                <div className="bg-slate-300 px-5 py-3 border-b border-slate-400/70 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500" />
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
                    <span className="ml-2 text-xs font-mono text-slate-900 font-semibold">
                      {activeFile.path}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-[11px] font-mono text-slate-700 font-medium">
                      {activeFile.content.split('\n').length} lines
                    </span>
                  </div>
                </div>
                <div
                  className="p-5 font-mono overflow-x-auto whitespace-pre-wrap break-words leading-relaxed max-h-[680px] bg-slate-300/80 text-slate-950"
                  style={{ fontSize: `${codeFontSize}px` }}
                >
                  <code>{highlightCodeToJSX(activeFile.content)}</code>
                </div>
              </div>
            )
          )}
        </div>
      </div>

      {/* Git & Google Colab / Jupyter Upload Modal */}
      {showGitModal && (
        <div className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full border border-slate-300 shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
            {/* Modal Header */}
            <div className="p-6 border-b border-slate-200 flex items-center justify-between bg-slate-50">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-blue-100 border border-blue-200 flex items-center justify-center text-blue-900">
                  <GitBranch className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-slate-900">
                    Upload &ldquo;{activeProject.title}&rdquo; to Git, Colab &amp; Jupyter
                  </h3>
                  <p className="text-xs text-slate-600">
                    Includes notebooks/{activeProject.id}_colab.ipynb (nbformat v4.5), .gitignore, Docker, and CI/CD
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowGitModal(false)}
                className="w-8 h-8 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-200/70 flex items-center justify-center transition-colors cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-6 space-y-5 max-h-[75vh] overflow-y-auto">
              {/* Google Colab & Jupyter Direct Upload */}
              <div className="bg-amber-50 border border-amber-300 rounded-xl p-4 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-amber-950 flex items-center gap-1.5">
                    <BookOpen className="w-4 h-4 text-amber-700" />
                    <span>Direct Google Colab &amp; Jupyter Upload (.ipynb)</span>
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-100 text-amber-950 font-bold border border-amber-300">
                    nbformat v4.5
                  </span>
                </div>
                <ol className="text-xs text-amber-950 space-y-1 list-decimal list-inside leading-relaxed">
                  <li>
                    Click <strong>Download .ipynb (Colab / Jupyter)</strong> to save <code className="font-mono font-semibold">{activeProject.id}_colab.ipynb</code>.
                  </li>
                  <li>
                    <strong>Google Colab:</strong> Open <code className="font-mono">colab.research.google.com</code> → <strong>File → Upload notebook</strong> → select the downloaded <code className="font-mono">.ipynb</code> file (or open directly after pushing to GitHub via <code className="font-mono">colab.research.google.com/github/&lt;user&gt;/{activeProject.id}/blob/main/notebooks/{activeProject.id}_colab.ipynb</code>).
                  </li>
                  <li>
                    <strong>Jupyter / JupyterLab:</strong> Upload <code className="font-mono">{activeProject.id}_colab.ipynb</code> in the Jupyter file browser or run <code className="font-mono">jupyter notebook notebooks/{activeProject.id}_colab.ipynb</code>.
                  </li>
                </ol>
              </div>

              {/* Option 1: Quick 1-Click Setup with setup_git.sh */}
              <div className="bg-slate-50 border border-slate-300 rounded-xl p-4 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                    <Zap className="w-4 h-4 text-amber-600" />
                    <span>Option 1: Automated 1-Step Git Setup (Fastest)</span>
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-100 text-emerald-900 font-bold border border-emerald-300">
                    Included in Zip
                  </span>
                </div>
                <p className="text-xs text-slate-600">
                  Unpack the downloaded zip (which also bundles <code className="font-mono">notebooks/{activeProject.id}_colab.ipynb</code>) and run the initializer script:
                </p>
                <div className="relative bg-slate-300/85 border border-slate-400/70 rounded-lg p-3 font-mono text-xs text-emerald-900 font-semibold overflow-x-auto">
                  <code>
                    unzip {activeProject.id}-git-repo.zip && cd {activeProject.id} && bash setup_git.sh
                  </code>
                </div>
              </div>

              {/* Option 2: Manual Git CLI Commands */}
              <div className="bg-slate-50 border border-slate-300 rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                    <Terminal className="w-4 h-4 text-blue-900" />
                    <span>Option 2: Direct Terminal Commands</span>
                  </span>
                  <button
                    onClick={handleCopyGitScript}
                    className="text-xs font-semibold text-blue-900 hover:underline flex items-center gap-1 cursor-pointer"
                  >
                    {copiedGitScript ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedGitScript ? 'Copied' : 'Copy Commands'}</span>
                  </button>
                </div>
                <div className="bg-slate-300/85 border border-slate-400/70 rounded-lg p-3.5 font-mono text-xs text-slate-900 overflow-x-auto whitespace-pre leading-relaxed">
                  {generateGitPushCommands(activeProject)}
                </div>
              </div>

              {/* Readiness Checklist */}
              <div className="space-y-2">
                <span className="text-xs font-bold text-slate-900 uppercase tracking-wider block">
                  Git, Colab &amp; Jupyter Readiness Checklist
                </span>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                  <div className="p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-950 flex items-center gap-2">
                    <Check className="w-4 h-4 text-amber-600 shrink-0" />
                    <span>notebooks/{activeProject.id}_colab.ipynb</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-900 flex items-center gap-2">
                    <Check className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>.gitignore (tracks .ipynb, skips caches)</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-900 flex items-center gap-2">
                    <Check className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>GitHub Actions CI (.github/workflows)</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-900 flex items-center gap-2">
                    <Check className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>requirements.txt &amp; pyproject.toml</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-900 flex items-center gap-2">
                    <Check className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>Dockerfile &amp; tests/test_pipeline.py</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-900 flex items-center gap-2">
                    <Check className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>src/main.py Orchestrator CLI</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-5 border-t border-slate-200 bg-slate-50 flex flex-wrap items-center justify-between gap-3">
              <button
                onClick={() => setShowGitModal(false)}
                className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-800 rounded-xl text-xs font-semibold transition-colors cursor-pointer"
              >
                Close
              </button>
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => downloadProjectNotebook(activeProject)}
                  className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs cursor-pointer"
                >
                  <BookOpen className="w-4 h-4" />
                  <span>Download .ipynb</span>
                </button>
                <button
                  onClick={handleDownloadZip}
                  disabled={isDownloadingZip}
                  className="px-5 py-2 bg-emerald-700 hover:bg-emerald-600 disabled:opacity-60 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs cursor-pointer"
                >
                  <Download className="w-4 h-4" />
                  <span>{isDownloadingZip ? 'Downloading...' : 'Download Git Repo (.zip)'}</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
