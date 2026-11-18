import React, { useState, useMemo, useEffect } from 'react';
import {
  Play,
  Youtube,
  Search,
  Filter,
  CheckCircle2,
  Clock,
  ExternalLink,
  GraduationCap,
  BookOpen,
  Sparkles,
  Layers,
  Compass,
  CheckSquare
} from 'lucide-react';

export type LearningPhase =
  | 'All Phases'
  | 'Phase 1: Foundations & Math'
  | 'Phase 2: Data Wrangling & SQL'
  | 'Phase 3: Classical Supervised ML'
  | 'Phase 4: Ensembles & Advanced ML'
  | 'Phase 5: Deep Learning Foundations'
  | 'Phase 6: Computer Vision & CNNs'
  | 'Phase 7: Transformers & LLMs'
  | 'Phase 8: RAG & GenAI Systems';

export interface CurriculumModule {
  id: string;
  youtubeId: string;
  title: string;
  channel: string;
  phase: Exclude<LearningPhase, 'All Phases'>;
  phaseNumber: number;
  level: 'Beginner' | 'Intermediate' | 'Advanced';
  duration: string;
  summary: string;
  keyTakeaways: string[];
  tags: string[];
}

const LEARNING_WATCHED_KEY = 'ml_studio_learning_watched_v2';

export const COMPREHENSIVE_CURRICULUM_MODULES: CurriculumModule[] = [
  // Phase 1: Foundations & Math
  {
    id: 'ds-foundations-python',
    youtubeId: 'LHBE6Q9XlzI',
    title: 'Python for Data Science - Course for Beginners (Pandas, NumPy, Matplotlib)',
    channel: 'freeCodeCamp.org',
    phase: 'Phase 1: Foundations & Math',
    phaseNumber: 1,
    level: 'Beginner',
    duration: '12h 19m',
    summary:
      'Complete foundational Python curriculum for data science covering vectorized NumPy operations, Pandas DataFrame indexing, data wrangling, cleaning, missing-value imputation, and Matplotlib exploratory charts.',
    keyTakeaways: [
      'High-speed vectorized array computing with NumPy',
      'Data cleaning, joining, GroupBy aggregations and missing data handling in Pandas',
      'Publication-grade exploratory visualization with Matplotlib'
    ],
    tags: ['Python', 'Pandas', 'NumPy', 'Matplotlib']
  },
  {
    id: 'ds-foundations-math',
    youtubeId: 'qYNweeDHiyU',
    title: 'AI, Machine Learning, Deep Learning and Generative AI Explained',
    channel: 'IBM Technology',
    phase: 'Phase 1: Foundations & Math',
    phaseNumber: 1,
    level: 'Beginner',
    duration: '10m',
    summary:
      'Executive-level roadmap clarifying the taxonomy of artificial intelligence, statistics, calculus, linear algebra, and data science foundations required before building predictive models.',
    keyTakeaways: [
      'Mathematical taxonomy of AI, ML, Deep Learning and GenAI',
      'Role of linear algebra and probability in predictive modeling',
      'How raw data transforms into actionable feature tensors'
    ],
    tags: ['AI Taxonomy', 'Statistics', 'Foundations']
  },

  // Phase 2: Data Wrangling & SQL
  {
    id: 'ds-data-science-full',
    youtubeId: 'ua-CiDNNj30',
    title: 'Learn Data Science Tutorial - Full Course for Beginners',
    channel: 'freeCodeCamp.org (Barton Poulson)',
    phase: 'Phase 2: Data Wrangling & SQL',
    phaseNumber: 2,
    level: 'Beginner',
    duration: '5h 52m',
    summary:
      'Comprehensive masterclass on exploratory data analysis (EDA), statistical probability, hypothesis testing, correlation analysis, and data storytelling for real-world tabular datasets.',
    keyTakeaways: [
      'End-to-end exploratory data analysis and distribution profiling',
      'Statistical testing, p-values, and correlation vs. causation',
      'Handling outliers, skewness and feature transformation'
    ],
    tags: ['EDA', 'Statistics', 'Data Wrangling', 'Hypothesis Testing']
  },

  // Phase 3: Classical Supervised ML
  {
    id: 'ds-ml-everybody',
    youtubeId: 'i_LwzRVP7bg',
    title: 'Machine Learning for Everybody – Full Course',
    channel: 'freeCodeCamp.org (Kylie Ying)',
    phase: 'Phase 3: Classical Supervised ML',
    phaseNumber: 3,
    level: 'Beginner',
    duration: '3h 53m',
    summary:
      'Gentle, intuitive introduction to supervised machine learning algorithms including Linear Regression, Logistic Regression, K-Nearest Neighbors, Naive Bayes, and Support Vector Machines.',
    keyTakeaways: [
      'Train/validation/test splits and bias-variance tradeoff',
      'Supervised regression and classification mechanics',
      'Scikit-Learn model training and evaluation metrics'
    ],
    tags: ['Scikit-Learn', 'Regression', 'Classification', 'SVM']
  },
  {
    id: 'ds-ml-beginners',
    youtubeId: 'NWONeJKn6kc',
    title: 'Machine Learning Course for Beginners',
    channel: 'freeCodeCamp.org',
    phase: 'Phase 3: Classical Supervised ML',
    phaseNumber: 3,
    level: 'Beginner',
    duration: '9h 50m',
    summary:
      'Extended applied bootcamp covering linear/logistic regression from scratch, gradient descent optimization, cost functions, decision trees, and regularization techniques (L1/L2).',
    keyTakeaways: [
      'Gradient descent optimization and loss minimization',
      'Decision tree entropy and information gain splits',
      'Preventing overfitting with Ridge and Lasso regularization'
    ],
    tags: ['Gradient Descent', 'Decision Trees', 'Regularization']
  },

  // Phase 4: Ensembles & Advanced ML
  {
    id: 'ds-sklearn-crash',
    youtubeId: '0B5eIE_1vpU',
    title: 'Scikit-learn Crash Course - Machine Learning Library for Python',
    channel: 'freeCodeCamp.org',
    phase: 'Phase 4: Ensembles & Advanced ML',
    phaseNumber: 4,
    level: 'Intermediate',
    duration: '1h 53m',
    summary:
      'Engineering masterclass on Scikit-Learn Pipelines, ColumnTransformers, hyperparameter tuning with GridSearchCV, and preventing data leakage in production ML workflows.',
    keyTakeaways: [
      'Building robust Scikit-Learn Pipelines & ColumnTransformers',
      'Automated hyperparameter tuning with GridSearchCV',
      'Stratified cross-validation and preventing data leakage'
    ],
    tags: ['Scikit-Learn', 'Pipelines', 'GridSearchCV', 'Feature Engineering']
  },
  {
    id: 'ds-statquest-rf',
    youtubeId: 'J4Wdy0Wc_xQ',
    title: 'StatQuest: Random Forests Part 1 - Building, Using and Evaluating',
    channel: 'StatQuest with Josh Starmer',
    phase: 'Phase 4: Ensembles & Advanced ML',
    phaseNumber: 4,
    level: 'Beginner',
    duration: '10m',
    summary:
      'Crystal-clear visual explanation of bootstrap aggregation (bagging), random feature subsetting, Out-of-Bag (OOB) error estimation, and why Random Forests outperform individual decision trees.',
    keyTakeaways: [
      'Bootstrap aggregating (bagging) and decorrelated trees',
      'Out-of-bag (OOB) error evaluation without holdout sets',
      'Feature importance scoring and ensemble voting'
    ],
    tags: ['StatQuest', 'Random Forest', 'Bagging', 'Ensembles']
  },

  // Phase 5: Deep Learning Foundations
  {
    id: 'ds-3b1b-neural-net',
    youtubeId: 'aircAruvnKk',
    title: 'But what is a neural network? | Deep learning chapter 1',
    channel: '3Blue1Brown',
    phase: 'Phase 5: Deep Learning Foundations',
    phaseNumber: 5,
    level: 'Beginner',
    duration: '19m',
    summary:
      'Visual geometric introduction to artificial neural networks, weights, biases, activation functions (sigmoid and ReLU), and hierarchical feature extraction.',
    keyTakeaways: [
      'How layers represent nested abstractions via matrix multiplications',
      'Role of nonlinear activation functions (ReLU, Sigmoid)',
      'Mapping raw pixels to output class probabilities'
    ],
    tags: ['Neural Networks', 'MLP', 'ReLU', 'Visual Math']
  },
  {
    id: 'ds-3b1b-backprop',
    youtubeId: 'Ilg3gGewQ5U',
    title: 'Backpropagation, intuitively | Deep Learning Chapter 3',
    channel: '3Blue1Brown',
    phase: 'Phase 5: Deep Learning Foundations',
    phaseNumber: 5,
    level: 'Intermediate',
    duration: '14m',
    summary:
      'Visual calculus intuition explaining how backpropagation computes gradients backward through network layers using the chain rule and mini-batch gradient descent.',
    keyTakeaways: [
      'Sensitivity of cost function to weights and biases',
      'Calculus chain-rule gradient accumulation across layers',
      'Mini-batch stochastic gradient descent mechanics'
    ],
    tags: ['Backpropagation', 'Chain Rule', 'Gradient Descent']
  },
  {
    id: 'ds-karpathy-micrograd',
    youtubeId: 'VMj-3S1tku0',
    title: 'The spelled-out intro to neural networks and backpropagation: building micrograd',
    channel: 'Andrej Karpathy',
    phase: 'Phase 5: Deep Learning Foundations',
    phaseNumber: 5,
    level: 'Intermediate',
    duration: '2h 25m',
    summary:
      'Build a tiny scalar-valued autograd engine and multi-layer perceptron completely from scratch in Python to demystify topological graph sorting and PyTorch `.backward()` internals.',
    keyTakeaways: [
      'Writing reverse-mode automatic differentiation from scratch',
      'Topological sorting of computational graphs',
      'Training a neural network classifier with L2 loss and weight decay'
    ],
    tags: ['Autograd', 'Python', 'PyTorch Internals', 'Backpropagation']
  },
  {
    id: 'ds-pytorch-full',
    youtubeId: 'V_xro1bcAuA',
    title: 'PyTorch for Deep Learning & Machine Learning – Full Course',
    channel: 'freeCodeCamp.org (Daniel Bourke)',
    phase: 'Phase 5: Deep Learning Foundations',
    phaseNumber: 5,
    level: 'Intermediate',
    duration: '25h 37m',
    summary:
      'Comprehensive PyTorch masterclass covering tensor manipulation, device-agnostic GPU workflows, custom `nn.Module` architectures, training loops, metrics tracking, and serialization.',
    keyTakeaways: [
      'Tensor broadcasting, autograd engine and CUDA optimization',
      'Writing clean train, validation and evaluation loops',
      'Model serialization, checkpointing and evaluation metrics'
    ],
    tags: ['PyTorch', 'Deep Learning', 'CUDA', 'Tensors']
  },

  // Phase 6: Computer Vision & CNNs
  {
    id: 'ds-tensorflow-full',
    youtubeId: 'tPYj3fFJGjk',
    title: 'TensorFlow 2.0 Complete Course - Python Neural Networks for Beginners Tutorial',
    channel: 'freeCodeCamp.org',
    phase: 'Phase 6: Computer Vision & CNNs',
    phaseNumber: 6,
    level: 'Beginner',
    duration: '6h 52m',
    summary:
      'Applied TensorFlow and Keras tutorial covering dense networks, convolutional neural networks (CNNs) for image classification, transfer learning, and computer vision pipelines.',
    keyTakeaways: [
      'Convolutional filters, pooling layers and spatial feature maps',
      'Transfer learning with pretrained MobileNet and ResNet models',
      'Data augmentation for robust image classification'
    ],
    tags: ['TensorFlow', 'Keras', 'CNN', 'Computer Vision']
  },

  // Phase 7: Transformers & LLMs
  {
    id: 'ds-karpathy-build-gpt',
    youtubeId: 'kCc8FmEb1nY',
    title: "Let's build GPT: from scratch, in code, spelled out.",
    channel: 'Andrej Karpathy',
    phase: 'Phase 7: Transformers & LLMs',
    phaseNumber: 7,
    level: 'Advanced',
    duration: '1h 56m',
    summary:
      'Step-by-step PyTorch walkthrough building a Generative Pre-trained Transformer (GPT) from scratch, starting with tokenization up to multi-head self-attention and autoregressive text generation.',
    keyTakeaways: [
      'Character and BPE tokenization pipelines',
      'Causal masked multi-head self-attention in PyTorch',
      'Transformer block composition, LayerNorm and residual connections'
    ],
    tags: ['PyTorch', 'GPT', 'Transformers', 'Self-Attention']
  },
  {
    id: 'ds-3b1b-transformers',
    youtubeId: 'wjZofJX0v4M',
    title: 'Transformers, the tech behind LLMs | Deep Learning Chapter 5',
    channel: '3Blue1Brown',
    phase: 'Phase 7: Transformers & LLMs',
    phaseNumber: 7,
    level: 'Beginner',
    duration: '27m',
    summary:
      'Visual geometric breakdown of Transformer embedding spaces, semantic directions, context vectors, and how language models predict the next token in sequence.',
    keyTakeaways: [
      'High-dimensional embedding geometry and semantic vector arithmetic',
      'Unembedding matrix projections and softmax temperature sampling',
      'Visual tour of the Transformer forward pass'
    ],
    tags: ['Visual Math', 'Embeddings', 'Transformers', 'LLMs']
  },
  {
    id: 'ds-3b1b-attention',
    youtubeId: 'eMlx5fFNoYc',
    title: 'Attention in transformers, step-by-step | Deep Learning Chapter 6',
    channel: '3Blue1Brown',
    phase: 'Phase 7: Transformers & LLMs',
    phaseNumber: 7,
    level: 'Intermediate',
    duration: '26m',
    summary:
      'Deep visual intuition for Query, Key, and Value matrices, scaled dot-product attention, causal masking, and parallel multi-head attention mechanisms.',
    keyTakeaways: [
      'Query-Key dot-product scoring and softmax normalization',
      'Causal masking for parallel autoregressive decoding',
      'Multi-head attention projections capturing diverse relationships'
    ],
    tags: ['Attention', 'QKV Matrices', 'Linear Algebra']
  },
  {
    id: 'ds-karpathy-intro-llms',
    youtubeId: 'zjkBMFhNj_g',
    title: '[1hr Talk] Intro to Large Language Models',
    channel: 'Andrej Karpathy',
    phase: 'Phase 7: Transformers & LLMs',
    phaseNumber: 7,
    level: 'Beginner',
    duration: '1h 00m',
    summary:
      'Executive conceptual overview of LLMs as an operating system kernel: pretraining internet data compression, fine-tuning assistants, scaling laws, and prompt-injection security.',
    keyTakeaways: [
      'Pretraining compression vs. assistant alignment tuning',
      'Scaling laws, compute limits and emerging capabilities',
      'Adversarial prompt injection and LLM security boundaries'
    ],
    tags: ['LLM OS', 'Scaling Laws', 'AI Security']
  },
  {
    id: 'ds-karpathy-deep-dive-llms',
    youtubeId: '7xTGNNLPyMI',
    title: 'Deep Dive into LLMs like ChatGPT',
    channel: 'Andrej Karpathy',
    phase: 'Phase 7: Transformers & LLMs',
    phaseNumber: 7,
    level: 'Intermediate',
    duration: '3h 31m',
    summary:
      'Comprehensive technical review of the 3-stage LLM training pipeline: pretraining, Supervised Fine-Tuning (SFT), RLHF/GRPO alignment, tool use, KV-cache optimization, and hallucination reduction.',
    keyTakeaways: [
      'Pretraining corpus preparation and tokenization details',
      'Supervised Fine-Tuning (SFT) and RLHF reward modeling',
      'KV-cache memory optimization and reasoning system architectures'
    ],
    tags: ['LLMs', 'RLHF', 'Fine-Tuning', 'Tokenization']
  },

  // Phase 8: RAG & GenAI Systems
  {
    id: 'ds-freecodecamp-rag',
    youtubeId: 'sVcwVQRHIc8',
    title: 'Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer',
    channel: 'freeCodeCamp.org (Lance Martin)',
    phase: 'Phase 8: RAG & GenAI Systems',
    phaseNumber: 8,
    level: 'Intermediate',
    duration: '2h 33m',
    summary:
      'Hands-on Python masterclass building production Retrieval-Augmented Generation (RAG) pipelines with LangChain, dense embeddings, vector indexing, query translation, and adaptive routing.',
    keyTakeaways: [
      'Document loading, semantic recursive chunking and embeddings',
      'Vector indexing and similarity search in FAISS / Chroma',
      'Advanced RAG: HyDE, RAG-Fusion, query rewriting and adaptive routing'
    ],
    tags: ['LangChain', 'RAG', 'Vector DB', 'Embeddings']
  },
  {
    id: 'ds-karpathy-state-of-gpt',
    youtubeId: 'bZQun8Y4L2A',
    title: 'State of GPT | BRK216HFS',
    channel: 'Microsoft Developer (Andrej Karpathy)',
    phase: 'Phase 8: RAG & GenAI Systems',
    phaseNumber: 8,
    level: 'Intermediate',
    duration: '42m',
    summary:
      'Engineering overview covering the GPT lifecycle, reinforcement learning from human feedback, prompt engineering best practices, RAG grounding, and agentic tool use.',
    keyTakeaways: [
      'Reinforcement Learning from Human Feedback (RLHF) mechanics',
      'Retrieval-Augmented Generation (RAG) context grounding',
      'System-1 prompt patterns vs. System-2 reasoning chains'
    ],
    tags: ['RLHF', 'RAG', 'Prompt Engineering', 'Agents']
  },
  {
    id: 'ds-google-genai',
    youtubeId: 'G2fqAlgmoPo',
    title: 'Introduction to Generative AI',
    channel: 'Google Cloud Tech',
    phase: 'Phase 8: RAG & GenAI Systems',
    phaseNumber: 8,
    level: 'Beginner',
    duration: '22m',
    summary:
      'Foundational module explaining generative foundation models, diffusion architectures, transformer models, and enterprise deployment best practices.',
    keyTakeaways: [
      'Discriminative vs. Generative artificial intelligence models',
      'Foundation model use cases across text, vision, and code',
      'Prompt design and enterprise tuning strategies'
    ],
    tags: ['Generative AI', 'Foundation Models', 'Google Cloud']
  }
];

export const CURRICULUM_PHASES: LearningPhase[] = [
  'All Phases',
  'Phase 1: Foundations & Math',
  'Phase 2: Data Wrangling & SQL',
  'Phase 3: Classical Supervised ML',
  'Phase 4: Ensembles & Advanced ML',
  'Phase 5: Deep Learning Foundations',
  'Phase 6: Computer Vision & CNNs',
  'Phase 7: Transformers & LLMs',
  'Phase 8: RAG & GenAI Systems'
];

export default function LearningSectionTab() {
  const [selectedPhase, setSelectedPhase] = useState<LearningPhase>('All Phases');
  const [selectedLevel, setSelectedLevel] = useState<'All' | 'Beginner' | 'Intermediate' | 'Advanced'>('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [activeVideo, setActiveVideo] = useState<CurriculumModule>(COMPREHENSIVE_CURRICULUM_MODULES[0]);
  const [watchedIds, setWatchedIds] = useState<string[]>(() => {
    try {
      const saved = window.localStorage.getItem(LEARNING_WATCHED_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) return parsed.filter((x): x is string => typeof x === 'string');
      }
    } catch {
      // Ignore storage read errors
    }
    return [];
  });

  useEffect(() => {
    try {
      window.localStorage.setItem(LEARNING_WATCHED_KEY, JSON.stringify(watchedIds));
    } catch {
      // Ignore storage write errors
    }
  }, [watchedIds]);

  const toggleWatched = (videoId: string) => {
    setWatchedIds((prev) =>
      prev.includes(videoId) ? prev.filter((id) => id !== videoId) : [...prev, videoId]
    );
  };

  const filteredModules = useMemo(() => {
    return COMPREHENSIVE_CURRICULUM_MODULES.filter((module) => {
      const matchesPhase = selectedPhase === 'All Phases' || module.phase === selectedPhase;
      const matchesLevel = selectedLevel === 'All' || module.level === selectedLevel;
      const q = searchQuery.trim().toLowerCase();
      const matchesSearch =
        !q ||
        module.title.toLowerCase().includes(q) ||
        module.channel.toLowerCase().includes(q) ||
        module.phase.toLowerCase().includes(q) ||
        module.summary.toLowerCase().includes(q) ||
        module.tags.some((t) => t.toLowerCase().includes(q));
      return matchesPhase && matchesLevel && matchesSearch;
    });
  }, [selectedPhase, selectedLevel, searchQuery]);

  const isActiveWatched = watchedIds.includes(activeVideo.id);
  const progressPercent = Math.round((watchedIds.length / COMPREHENSIVE_CURRICULUM_MODULES.length) * 100);

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="bg-slate-100 border border-slate-300 rounded-2xl p-6 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-blue-900 mb-2 font-semibold">
              <span className="bg-blue-100 px-2 py-0.5 rounded text-blue-900 font-bold border border-blue-200">
                [Complete End-to-End Syllabus]
              </span>
              <span>Data Science · Machine Learning · Deep Learning · LLMs &amp; RAG Mastery</span>
            </div>
            <h2 className="text-2xl font-bold text-slate-900 tracking-tight flex items-center gap-2.5">
              <GraduationCap className="w-6 h-6 text-indigo-600" />
              <span>Data Science &amp; AI/ML Masterclass Curriculum</span>
            </h2>
            <p className="text-slate-700 text-sm mt-1 max-w-4xl leading-relaxed">
              A structured 8-phase syllabus mapping out your learning journey from absolute beginner Python foundations and statistics up to production-grade Transformers, RAG systems, and MLOps.
            </p>
          </div>

          {/* Progress Tracker Pill */}
          <div className="bg-white border border-slate-300 rounded-xl px-4 py-3 shrink-0 min-w-[220px] shadow-2xs">
            <div className="flex items-center justify-between text-xs font-semibold text-slate-800 mb-1.5">
              <span>Syllabus Completion</span>
              <span className="font-mono text-indigo-700">
                {watchedIds.length}/{COMPREHENSIVE_CURRICULUM_MODULES.length} ({progressPercent}%)
              </span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-600 transition-all duration-300"
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>
        </div>

        {/* Phase Filter + Level + Search Controls */}
        <div className="mt-6 pt-5 border-t border-slate-300 space-y-4">
          <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-xs font-semibold text-slate-700 mr-1 flex items-center gap-1.5">
                <Compass className="w-3.5 h-3.5 text-blue-900" />
                <span>Curriculum Phase:</span>
              </span>
              {CURRICULUM_PHASES.map((phase) => (
                <button
                  key={phase}
                  onClick={() => setSelectedPhase(phase)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                    selectedPhase === phase
                      ? 'bg-blue-900 text-white shadow-xs'
                      : 'bg-slate-200 hover:bg-slate-300 text-slate-800 border border-slate-300'
                  }`}
                >
                  {phase === 'All Phases'
                    ? `All Phases (${COMPREHENSIVE_CURRICULUM_MODULES.length})`
                    : phase.replace(/^Phase \d+: /, '')}
                </button>
              ))}
            </div>

            <div className="flex flex-wrap items-center gap-3">
              <select
                value={selectedLevel}
                onChange={(e) => setSelectedLevel(e.target.value as typeof selectedLevel)}
                className="bg-white border border-slate-300 rounded-xl px-3 py-2 text-xs font-semibold text-slate-800 focus:outline-none focus:border-blue-900 shadow-2xs"
              >
                <option value="All">All Levels</option>
                <option value="Beginner">Beginner</option>
                <option value="Intermediate">Intermediate</option>
                <option value="Advanced">Advanced</option>
              </select>

              <div className="relative w-full sm:w-72">
                <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search modules, topics, tags..."
                  className="w-full bg-white border border-slate-300 rounded-xl pl-9 pr-4 py-2 text-xs text-slate-900 placeholder:text-slate-500 focus:outline-none focus:border-blue-900 shadow-2xs"
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Featured Embedded YouTube Player */}
      <div className="bg-slate-100 border border-slate-300 rounded-2xl p-6 shadow-md">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* 16:9 YouTube Player */}
          <div className="lg:col-span-7">
            <div className="aspect-video w-full rounded-xl overflow-hidden bg-slate-900 border border-slate-300 shadow-sm">
              <iframe
                key={activeVideo.youtubeId}
                src={`https://www.youtube.com/embed/${activeVideo.youtubeId}?rel=0`}
                title={activeVideo.title}
                className="w-full h-full"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                allowFullScreen
              />
            </div>
          </div>

          {/* Active Module Details */}
          <div className="lg:col-span-5 flex flex-col justify-between space-y-5">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="px-2.5 py-0.5 rounded-md bg-indigo-100 text-indigo-900 border border-indigo-300 font-bold font-mono">
                  Phase {activeVideo.phaseNumber}
                </span>
                <span className="px-2.5 py-0.5 rounded-md bg-blue-100 text-blue-900 border border-blue-300 font-semibold">
                  {activeVideo.phase}
                </span>
                <span className="px-2.5 py-0.5 rounded-md bg-slate-200 text-slate-800 border border-slate-300 font-semibold">
                  {activeVideo.level}
                </span>
                <span className="inline-flex items-center gap-1 text-slate-700 font-mono font-semibold">
                  <Clock className="w-3.5 h-3.5 text-slate-500" />
                  {activeVideo.duration}
                </span>
              </div>

              <h3 className="text-xl font-bold text-slate-900 leading-snug">{activeVideo.title}</h3>
              <p className="text-xs font-semibold text-indigo-700 flex items-center gap-1.5">
                <Youtube className="w-2.5 h-2.5 text-rose-600" />
                <span>Instructor / Channel: {activeVideo.channel}</span>
              </p>

              <p className="text-sm text-slate-700 leading-relaxed">{activeVideo.summary}</p>

              <div className="bg-white border border-slate-300 rounded-xl p-4 space-y-2">
                <div className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
                  <span>Key Learning Objectives</span>
                </div>
                <ul className="space-y-1.5">
                  {activeVideo.keyTakeaways.map((point, idx) => (
                    <li key={idx} className="text-xs text-slate-700 flex items-start gap-2">
                      <span className="text-indigo-600 font-bold mt-0.5">•</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-3 pt-2">
              <button
                type="button"
                onClick={() => toggleWatched(activeVideo.id)}
                className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-1.5 border transition-colors ${
                  isActiveWatched
                    ? 'bg-emerald-100 hover:bg-emerald-200 text-emerald-900 border-emerald-300'
                    : 'bg-white hover:bg-slate-200 text-slate-800 border-slate-300'
                }`}
              >
                <CheckCircle2 className={`w-4 h-4 ${isActiveWatched ? 'text-emerald-700' : 'text-slate-400'}`} />
                <span>{isActiveWatched ? 'Module Completed' : 'Mark Module Completed'}</span>
              </button>

              <a
                href={`https://www.youtube.com/watch?v=${activeVideo.youtubeId}`}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 rounded-xl text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white flex items-center gap-1.5 transition-colors shadow-xs"
              >
                <Youtube className="w-2.5 h-2.5" />
                <span>Open on YouTube</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Curriculum Grid */}
      {filteredModules.length === 0 ? (
        <div className="bg-slate-100 border border-slate-300 rounded-2xl p-10 text-center text-sm text-slate-600">
          No curriculum modules matched your filter criteria. Try resetting the phase filter or search terms.
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3.5">
          {filteredModules.map((module) => {
            const isSelected = activeVideo.id === module.id;
            const isWatched = watchedIds.includes(module.id);

            return (
              <div
                key={module.id}
                onClick={() => {
                  setActiveVideo(module);
                  window.scrollTo({ top: 0, behavior: 'smooth' });
                }}
                className={`cursor-pointer rounded-xl bg-slate-100 border p-2.5 transition-all duration-200 hover:bg-slate-200 hover:shadow-md flex flex-col justify-between ${
                  isSelected ? 'border-indigo-600 ring-2 ring-indigo-600/25 bg-slate-200' : 'border-slate-300'
                }`}
              >
                <div>
                  {/* Thumbnail */}
                  <div className="relative aspect-video rounded-lg overflow-hidden bg-slate-900 border border-slate-300 mb-2 group">
                    <img
                      src={`https://img.youtube.com/vi/${module.youtubeId}/hqdefault.jpg`}
                      alt={module.title}
                      loading="lazy"
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute inset-0 bg-slate-900/20 group-hover:bg-slate-900/10 transition-colors flex items-center justify-center">
                      <div className="w-5 h-5 rounded-full bg-white/95 shadow-xs flex items-center justify-center group-hover:scale-110 transition-transform">
                        <Play className="w-2.5 h-2.5 text-indigo-600 fill-indigo-600 ml-0.5" />
                      </div>
                    </div>
                    <span className="absolute bottom-1 right-1 px-1.5 py-0.5 rounded bg-slate-900/85 text-white text-[9px] font-mono font-semibold">
                      {module.duration}
                    </span>
                    <span className="absolute top-1 left-1 px-1.5 py-0.5 rounded bg-indigo-700 text-white text-[9px] font-mono font-bold shadow-xs">
                      P{module.phaseNumber}
                    </span>
                    {isWatched && (
                      <span className="absolute top-1 right-1 px-1.5 py-0.5 rounded bg-emerald-600 text-white text-[9px] font-semibold flex items-center gap-0.5 shadow-xs">
                        <CheckCircle2 className="w-2.5 h-2.5" />
                        <span>Done</span>
                      </span>
                    )}
                  </div>

                  {/* Phase Badge + Level */}
                  <div className="flex items-center justify-between gap-1 mb-1">
                    <span className="text-[10px] font-semibold text-blue-900 uppercase tracking-wider truncate">
                      {module.phase.replace(/^Phase \d+: /, '')}
                    </span>
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded font-medium border shrink-0 ${
                        module.level === 'Beginner'
                          ? 'bg-emerald-100 text-emerald-900 border-emerald-300'
                          : module.level === 'Intermediate'
                          ? 'bg-blue-100 text-blue-900 border-blue-300'
                          : 'bg-purple-100 text-purple-900 border-purple-300'
                      }`}
                    >
                      {module.level}
                    </span>
                  </div>

                  <h3 className="text-xs font-bold text-slate-900 leading-snug mb-1 line-clamp-2">
                    {module.title}
                  </h3>
                  <p className="text-[10px] font-semibold text-indigo-700 mb-1 truncate">{module.channel}</p>
                  <p className="text-[10px] text-slate-600 leading-relaxed line-clamp-1 mb-2">{module.summary}</p>
                </div>

                <div className="space-y-1.5 pt-1.5 border-t border-slate-300">
                  <div className="flex flex-wrap gap-1">
                    {module.tags.slice(0, 2).map((tag) => (
                      <span
                        key={tag}
                        className="text-[9px] px-1.5 py-0.5 rounded font-mono bg-slate-200 text-slate-800 border border-slate-300"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>

                  <div className="flex items-center justify-between text-[10px]">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleWatched(module.id);
                      }}
                      className={`inline-flex items-center gap-1 font-semibold transition-colors ${
                        isWatched ? 'text-emerald-800 hover:text-emerald-950' : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      <CheckSquare className="w-3 h-3" />
                      <span>{isWatched ? 'Done' : 'Mark'}</span>
                    </button>

                    <span className="text-blue-900 font-semibold flex items-center gap-0.5">
                      <BookOpen className="w-3 h-3" />
                      <span>{isSelected ? 'Playing' : 'Study'}</span>
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
