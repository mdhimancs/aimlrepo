import React, { useState, useEffect, useRef } from 'react';
import { 
  Cpu, 
  Layers, 
  Terminal, 
  Play, 
  Bot, 
  CheckCircle2, 
  Sparkles, 
  Code2, 
  Database, 
  BarChart3, 
  Server, 
  ArrowRight,
  ExternalLink,
  Search,
  Filter, 
  RefreshCw, 
  Copy, 
  Check, 
  Sliders, 
  Workflow, 
  FileText,
  ShieldCheck,
  Zap,
  X,
  BookOpen,
  LayoutDashboard,
  Wand2,
  LineChart,
  MessageSquareCode,
  MapPin,
  FlaskConical,
  Atom,
  Brain,
  Network,
  Gauge,
  TrendingUp,
  Activity,
  Lightbulb,
  AlertCircle,
  PieChart,
  FolderGit2,
  Newspaper,
  Archive,
  ChevronDown,
  Star,
  GraduationCap,
  Download,
  ArrowUp
} from 'lucide-react';
import { ProjectBlueprint, FlagshipProject, RagCitation, RagStatusResponse, SyntheticDataType, SyntheticVariant } from './types';
import WhitePaperBlogTab from './components/WhitePaperBlogTab';
import CodeRepoStructureTab from './components/CodeRepoStructureTab';
import LearningSectionTab from './components/LearningSectionTab';
import ProjectTelemetryCharts from './components/ProjectTelemetryCharts';
import AnnotatedCodePipeline from './components/AnnotatedCodePipeline';
import RepoRagIngestionPanel from './components/RepoRagIngestionPanel';
import ProjectSyntheticDatasetPanel from './components/ProjectSyntheticDatasetPanel';
import GoogleScholarWhitePaper from './components/GoogleScholarWhitePaper';
import ProjectEconomicsPanel from './components/ProjectEconomicsPanel';
import MLTrainingSimulatorTab, { SimConfigState, TrainingLogEntry } from './components/MLTrainingSimulatorTab';
import { buildColabPercentScript, downloadProjectGitZip, downloadProjectNotebook } from './utils/gitRepoBuilder';
import { computeProjectEconomics, formatRsOnly, formatUsdOnly } from './utils/projectEconomics';
import {
  ENCRYPTION_MECHANISM_META,
  LEARNING_RATE_PRESETS,
  ML_TECHNIQUE_META,
  PROGRAMMING_ARCHITECTURE_META,
  SYNTHETIC_DATA_TYPE_META,
  SYNTHETIC_VARIANT_META,
  computeLatencyP95P99,
  computeScale1B5Metrics,
  inferBestLearningRate,
  inferBestMlTechnique,
  inferBestProgrammingArch,
  inferOptimalEpochs,
  inferProjectDataType
} from './utils/syntheticDataset';
import { EXTERNAL_URL_PROJECTS } from './data/importedProjects';
import { RESEARCH_GRADE_PROJECTS } from './data/researchGradeProjects';
import { CYBERSECURITY_RESEARCH_PROJECTS } from './data/cybersecurityProjects';
import { UIDAI_DPI_PROJECTS } from './data/uidaiProjects';
import { WORLD_SCALE_COMMERCIAL_PROJECTS } from './data/worldScaleCommercialProjects';

const FLAGSHIP_PROJECTS: FlagshipProject[] = [
  {
    id: 'medical-swin-transformer',
    title: 'Multimodal Medical Imaging Diagnostic Swin Transformer',
    domain: 'Computer Vision & Healthcare AI',
    difficulty: 'Research Grade',
    description: 'Implement a Hierarchical Swin Transformer architecture combined with Vision-Language attention to diagnose pulmonary pathology from high-resolution 3D CT scans, accelerated via Rust and ONNX Runtime.',
    objective: 'Surpass radiologist consensus AUROC across 14 thoracic disease categories with sub-30ms edge inference.',
    metrics: 'Mean AUROC: 0.948 · Sensitivity: 96.2% · Specificity: 94.8%',
    trainingLoss: '0.089 (Focal Loss)',
    trainingAccuracy: '95.4% (Top-1 Recall)',
    problem: 'Radiological interpretation of 3D volumetric CT/MRI scans is bottlenecked by inter-observer variability, high cognitive fatigue, and the inability of traditional 2D CNNs to capture long-range multi-organ spatial dependencies.',
    breakingGround: 'First open-source hierarchical 3D Swin Transformer pretrained via Masked Volume Modeling (MVM) on 1.2M unlabelled scans, paired with zero-copy Rust DICOM parsing for deterministic sub-30ms clinical edge inference.',
    epochHistory: [
      { epoch: 1, loss: 1.85, accuracy: 0.45, valAccuracy: 0.42 },
      { epoch: 2, loss: 1.32, accuracy: 0.62, valAccuracy: 0.58 },
      { epoch: 3, loss: 0.94, accuracy: 0.74, valAccuracy: 0.71 },
      { epoch: 4, loss: 0.68, accuracy: 0.81, valAccuracy: 0.79 },
      { epoch: 5, loss: 0.49, accuracy: 0.87, valAccuracy: 0.85 },
      { epoch: 6, loss: 0.35, accuracy: 0.91, valAccuracy: 0.89 },
      { epoch: 7, loss: 0.24, accuracy: 0.93, valAccuracy: 0.92 },
      { epoch: 8, loss: 0.16, accuracy: 0.95, valAccuracy: 0.94 },
      { epoch: 9, loss: 0.11, accuracy: 0.96, valAccuracy: 0.95 },
      { epoch: 10, loss: 0.089, accuracy: 0.975, valAccuracy: 0.968 }
    ],
    sampleDistribution: [
      { label: 'Normal Lung Voxel Density', percentage: 54.2, count: '650,200 scans' },
      { label: 'Pulmonary Nodule (Benign)', percentage: 22.8, count: '273,600 scans' },
      { label: 'Consolidation / Pneumonia', percentage: 14.5, count: '174,000 scans' },
      { label: 'Pneumothorax / Effusion', percentage: 8.5, count: '102,200 scans' }
    ],
    performanceIndicators: {
      latency: '28ms / Scan (Rust ONNX Runtime)',
      throughput: '35 Volumetric Scans/s',
      memoryFootprint: '6.2 GB VRAM',
      hardwareTarget: 'NVIDIA Jetson AGX Orin / A100'
    },
    techStack: ['Rust', 'Python', 'PyTorch', 'MONAI', 'ONNX Runtime', 'FastAPI'],
    architectureFlow: [
      'DICOM 3D Volume Ingestion',
      'Rust Zero-Copy DICOM Header Parser',
      'HU Windowing & Isotropic Resampling',
      'Swin Transformer 3D Backbone',
      'Rust ONNX Runtime Inference Engine',
      'Multi-Label Pathology Classification'
    ],
    pipeline: [
      {
        title: '01. Zero-Copy DICOM Parsing in Rust',
        desc: 'Parse multi-gigabyte DICOM medical imaging files safely and concurrently using Rust memory-mapped file IO.',
        code: `// Rust High-Performance DICOM Parser
use dicom_object::open_file;

pub fn parse_dicom_metadata(file_path: &str) -> Result<String, Box<dyn std::error::Error>> {
    let obj = open_file(file_path)?;
    let patient_id = obj.element_by_name("PatientID")?.to_str()?;
    Ok(patient_id.to_string())
}`,
        language: 'rust'
      },
      {
        title: '02. Swin Transformer 3D Backbone (PyTorch)',
        desc: 'Construct hierarchical shifted window transformer blocks for multi-scale feature representation learning on volumetric medical inputs.',
        code: `import torch.nn as nn
from monai.networks.nets import SwinUNETR

model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=1,
    out_channels=14,
    feature_size=48,
    use_checkpoint=True
)`,
        language: 'python'
      }
    ],
    whitePaper: {
      problem: 'Radiological interpretation of 3D volumetric CT/MRI scans is bottlenecked by inter-observer variability, high cognitive fatigue, and the inability of traditional 2D CNNs to capture long-range multi-organ spatial dependencies.',
      breakingGround: 'First open-source hierarchical 3D Swin Transformer pretrained via Masked Volume Modeling (MVM) on 1.2M unlabelled scans, paired with zero-copy Rust DICOM parsing for deterministic sub-30ms clinical edge inference.',
      scope: 'Automated computer-aided diagnosis for 3D volumetric medical imaging (CT scans), integrating Rust for high-speed DICOM parsing and ONNX Runtime for clinical edge deployment.',
      functionality: '1. Memory-mapped zero-copy DICOM parsing in Rust.\n2. Isotropic voxel resampling and Hounsfield unit windowing.\n3. 3D Swin Transformer encoder-decoder segmentation.\n4. Rust-hosted ONNX Runtime execution for sub-30ms clinical inference.',
      abstract: 'This research white paper proposes a 3D Hierarchical Swin Transformer combined with Rust-accelerated DICOM preprocessing and open-source Apache TVM / OpenVINO / ONNX Runtime inference.',
      methodology: 'Volumetric data parsing is offloaded to Rust io_uring workers to eliminate Python memory overhead, feeding preprocessed tensors directly into GPU/CPU memory.',
      architecture: 'Rust preprocessing daemon communicating with C++20 Seastar / ONNX Runtime via Apache Arrow zero-copy shared memory buffers.',
      benchmarks: 'Achieved 35 volumetric scans per second throughput with 28ms P95 inference latency on NVIDIA Jetson Orin edge hardware.',
      productionConsiderations: 'Strict memory safety guarantees, zero data corruption during concurrent medical scan ingestion, and HIPAA-compliant audit logging.'
    }
  },
  {
    id: 'protein-folding-gnn',
    title: 'Protein Folding & Drug Discovery Graph Neural Network',
    domain: 'Biotech & Graph Representation Learning',
    difficulty: 'Research Grade',
    description: 'Construct an SE(3)-equivariant Graph Neural Network accelerated by Rust spatial indexing algorithms to predict 3D atomic coordinates of amino acid residues.',
    objective: 'Predict protein backbone torsion angles and RMSD < 1.5 Å with Rust k-d tree spatial graph acceleration.',
    metrics: 'Global Distance Test (GDT): 91.2 · RMSD: 1.24 Å',
    trainingLoss: '0.051 (Kabsch RMSD Loss)',
    trainingAccuracy: '93.8% (Backbone Precision)',
    problem: 'Determining 3D protein structures via X-ray crystallography or cryogenic electron microscopy is exceptionally slow, expensive, and fails to cover millions of sequenced proteins.',
    breakingGround: 'Combines SE(3)-equivariant message passing with $O(N \\log N)$ Rust k-d tree spatial graph construction, enabling real-time structural folding predictions for large macromolecular complexes.',
    epochHistory: [
      { epoch: 1, loss: 2.10, accuracy: 0.38, valAccuracy: 0.35 },
      { epoch: 2, loss: 1.55, accuracy: 0.52, valAccuracy: 0.49 },
      { epoch: 3, loss: 1.12, accuracy: 0.68, valAccuracy: 0.64 },
      { epoch: 4, loss: 0.81, accuracy: 0.76, valAccuracy: 0.73 },
      { epoch: 5, loss: 0.58, accuracy: 0.82, valAccuracy: 0.80 },
      { epoch: 6, loss: 0.41, accuracy: 0.87, valAccuracy: 0.85 },
      { epoch: 7, loss: 0.29, accuracy: 0.90, valAccuracy: 0.88 },
      { epoch: 8, loss: 0.19, accuracy: 0.93, valAccuracy: 0.91 },
      { epoch: 9, loss: 0.12, accuracy: 0.95, valAccuracy: 0.93 },
      { epoch: 10, loss: 0.051, accuracy: 0.965, valAccuracy: 0.952 }
    ],
    sampleDistribution: [
      { label: 'Alpha Helix Residues', percentage: 41.5, count: '4.2M residues' },
      { label: 'Beta Sheet Strands', percentage: 32.1, count: '3.2M residues' },
      { label: 'Coils & Turns', percentage: 18.4, count: '1.8M residues' },
      { label: 'Ligand Binding Pockets', percentage: 8.0, count: '0.8M residues' }
    ],
    performanceIndicators: {
      latency: '0.8s / Protein Monomer',
      throughput: '1,400 Residues/s',
      memoryFootprint: '16.2 GB VRAM',
      hardwareTarget: 'NVIDIA A100 Cluster'
    },
    techStack: ['Rust', 'Python', 'PyTorch Geometric', 'BioPython', 'Ray'],
    architectureFlow: [
      'PDB Crystal Structure Parser',
      'Rust k-d Tree Spatial Graph Builder',
      'Euclidean Distance Thresholding',
      'SE(3)-Equivariant Message Passing',
      'Iterative Coordinate Refinement',
      'HPC Distributed Docking Engine'
    ],
    pipeline: [
      {
        title: '01. Rust k-d Tree Spatial Graph Construction',
        desc: 'Compute C-alpha atom spatial proximity graphs in $O(N \\log N)$ time using Rust spatial indexing data structures.',
        code: `// Rust Spatial k-d Tree for Protein Residues
use kiddo::{KdTree, Ix3};

pub fn build_kd_tree_graph(coords: &[[f32; 3]]) -> Vec<(usize, usize)> {
    let mut tree: KdTree<f32, 3> = KdTree::new();
    for (i, &coord) in coords.iter().enumerate() {
        tree.add(&coord, i as u64);
    }
    // Find neighbors within 8.0 Angstrom threshold
    vec![]
}`,
        language: 'rust'
      },
      {
        title: '02. SE(3)-Equivariant Message Passing',
        desc: 'Implement rotation and translation invariant message passing layers in PyTorch that update 3D atomic coordinates.',
        code: `import torch
import torch.nn as nn

class EquivariantGNNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        
    def forward(self, x, edge_index, pos):
        return self.fc(x)`,
        language: 'python'
      }
    ],
    whitePaper: {
      problem: 'Determining 3D protein structures via X-ray crystallography or cryogenic electron microscopy is exceptionally slow, expensive, and fails to cover millions of sequenced proteins.',
      breakingGround: 'Combines SE(3)-equivariant message passing with $O(N \\log N)$ Rust k-d tree spatial graph construction, enabling real-time structural folding predictions for large macromolecular complexes.',
      scope: 'Computational drug discovery and structural biology modeling, utilizing Rust for $O(N \\log N)$ spatial graph construction and PyTorch for SE(3)-equivariant message passing.',
      functionality: '1. PDB file parser in Rust.\n2. k-d tree spatial neighbor indexing for fast atomic distance queries.\n3. SE(3)-equivariant graph neural network layers.\n4. Iterative coordinate refinement and binding free energy estimation.',
      abstract: 'This research paper presents an SE(3)-equivariant graph neural network paired with Rust spatial indexing algorithms for rapid ab initio protein folding.',
      methodology: 'Spatial graph construction is accelerated using Rust k-d trees, reducing CPU preprocessing bottlenecks by 85% for large macromolecular complexes.',
      architecture: 'Distributed Rust preprocessing workers feeding PyTorch Geometric GPU training pipelines.',
      benchmarks: 'Tested on CASP14 target sequences. Achieved average GDT score of 91.2 and backbone RMSD of 1.24 Å with 1,400 residues/sec throughput.',
      productionConsiderations: 'High-throughput HPC cluster execution with MPI parallelization for screening virtual chemical libraries valued at over $10.0 B / Rs 83,500 Cr.'
    }
  },
  {
    id: 'rlhf-dpo-llm',
    title: 'Direct Preference Optimization (DPO) & RLHF LLM Alignment',
    domain: 'Generative AI & LLM Safety Alignment',
    difficulty: 'Research Grade',
    description: 'Implement Direct Preference Optimization (DPO) with Rust-accelerated dataset shuffling and tokenization for aligning open-source LLMs.',
    objective: 'Maximize win-rate against unaligned baseline models with zero memory fragmentation.',
    metrics: 'AlpacaEval Win-Rate: 88.4% · Toxicity Reduction: 94.1%',
    trainingLoss: '0.182 (Bradley-Terry DPO Loss)',
    trainingAccuracy: '89.1% (Preference Match)',
    problem: 'Traditional PPO-based RLHF for large language models requires simultaneous training of actor, critic, reward, and reference models, leading to extreme training instability, GPU memory bloat, and hyperparameter sensitivity.',
    breakingGround: 'Eliminates the separate reward model entirely by formulating direct preference optimization (DPO) via closed-form mathematical derivation, accelerated by Rust Polars zero-copy data pipelines.',
    epochHistory: [
      { epoch: 1, loss: 1.45, accuracy: 0.51, valAccuracy: 0.48 },
      { epoch: 2, loss: 1.10, accuracy: 0.63, valAccuracy: 0.60 },
      { epoch: 3, loss: 0.85, accuracy: 0.72, valAccuracy: 0.69 },
      { epoch: 4, loss: 0.64, accuracy: 0.78, valAccuracy: 0.75 },
      { epoch: 5, loss: 0.48, accuracy: 0.83, valAccuracy: 0.81 },
      { epoch: 6, loss: 0.36, accuracy: 0.87, valAccuracy: 0.85 },
      { epoch: 7, loss: 0.27, accuracy: 0.90, valAccuracy: 0.88 },
      { epoch: 8, loss: 0.20, accuracy: 0.92, valAccuracy: 0.90 },
      { epoch: 9, loss: 0.14, accuracy: 0.94, valAccuracy: 0.92 },
      { epoch: 10, loss: 0.182, accuracy: 0.945, valAccuracy: 0.932 }
    ],
    sampleDistribution: [
      { label: 'Helpful & Constructive', percentage: 48.0, count: '1.2M pairs' },
      { label: 'Safe & Harmless Refusals', percentage: 31.2, count: '0.8M pairs' },
      { label: 'Factual & Grounded Q&A', percentage: 15.8, count: '0.4M pairs' },
      { label: 'Creative & Code Synthesis', percentage: 5.0, count: '0.1M pairs' }
    ],
    performanceIndicators: {
      latency: '14ms / Token (vLLM + Rust)',
      throughput: '68 tokens/s/GPU',
      memoryFootprint: '64 GB (ZeRO-3)',
      hardwareTarget: 'NVIDIA H100 80GB'
    },
    techStack: ['Rust', 'Python', 'PyTorch', 'Transformers', 'TRL', 'vLLM'],
    architectureFlow: [
      'Human Preference Dataset Ingestion',
      'Rust Polars High-Speed Dataset Shuffle',
      'Prompt, Chosen, Rejected Tokenization',
      'Llama-3 Base Policy Model',
      'Direct Preference Optimization Loss',
      'DeepSpeed ZeRO-3 Memory Offload',
      'vLLM Safety Guardrail Serving'
    ],
    pipeline: [
      {
        title: '01. High-Speed Data Shuffling with Polars & Rust',
        desc: 'Ingest and shuffle multi-gigabyte preference datasets instantly using Rust-backed Polars dataframe libraries.',
        code: `// Rust Polars High-Speed Dataset Ingestion
use polars::prelude::*;

pub fn process_preference_dataset(path: &str) -> Result<DataFrame, PolarsError> {
    LazyCsvReader::new(path)
        .finish()?
        .filter(col("score_diff").gt(1.5))
        .collect()
}`,
        language: 'rust'
      },
      {
        title: '02. DPO Trainer Initialization (Python / TRL)',
        desc: 'Optimize the language model policy directly on preference data using the DPO loss formulation.',
        code: `from trl import DPOTrainer
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b", device_map="auto")
ref_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b", device_map="auto")

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=0.1,
    train_dataset=dataset,
)`,
        language: 'python'
      }
    ],
    whitePaper: {
      problem: 'Traditional PPO-based RLHF for large language models requires simultaneous training of actor, critic, reward, and reference models, leading to extreme training instability, GPU memory bloat, and hyperparameter sensitivity.',
      breakingGround: 'Eliminates the separate reward model entirely by formulating direct preference optimization (DPO) via closed-form mathematical derivation, accelerated by Rust Polars zero-copy data pipelines.',
      scope: 'Post-training alignment of large language models for enterprise deployment, utilizing Rust-backed Polars data processing and vLLM inference engines.',
      functionality: '1. Multi-gigabyte preference dataset parsing with Rust Polars.\n2. Parameter-Efficient Fine-Tuning (PEFT) via LoRA adapters.\n3. Direct Preference Optimization (DPO) training loop.\n4. Rust-accelerated vLLM serving with PagedAttention.',
      abstract: 'This paper evaluates Direct Preference Optimization (DPO) combined with Rust-backed data pipelines for high-speed enterprise LLM alignment.',
      methodology: 'Dataset preparation and sharding are performed in Rust, eliminating memory leaks and accelerating epoch startup times by 4x.',
      architecture: 'Rust data preprocessing daemon paired with DeepSpeed ZeRO-3 distributed training nodes.',
      benchmarks: 'Achieved an 88.4% win-rate on AlpacaEval 2.0 and 68 tokens/sec inference generation throughput.',
      productionConsiderations: 'Memory-safe execution, zero GIL locking during data preprocessing, and enterprise-grade vLLM token filtering.'
    }
  },
  {
    id: 'hft-rl-trading',
    title: 'High-Frequency Algorithmic Trading with Deep Reinforcement Learning',
    domain: 'Quantitative Finance & Reinforcement Learning',
    difficulty: 'Research Grade',
    description: 'Design a PPO reinforcement learning agent featuring an ultra-low latency Rust limit order book matching and feature extraction engine.',
    objective: 'Execute optimal trade execution within 500 microseconds using hybrid Rust-Python execution bridges.',
    metrics: 'Sharpe Ratio: 3.42 · Max Drawdown: 4.1%',
    trainingLoss: '0.019 (PPO Clipped Surrogate Loss)',
    trainingAccuracy: '91.4% (Optimal Action Precision)',
    problem: 'Traditional Python-based quantitative trading backends suffer from garbage collection pauses and high latency jitter, missing microsecond arbitrage and market making execution windows.',
    breakingGround: 'Introduces a hybrid lock-free Rust Tokio async LOB engine coupled via shared memory IPC to a Ray PPO reinforcement learning policy network, guaranteeing sub-millisecond deterministic execution.',
    epochHistory: [
      { epoch: 1, loss: 1.62, accuracy: 0.41, valAccuracy: 0.39 },
      { epoch: 2, loss: 1.18, accuracy: 0.57, valAccuracy: 0.54 },
      { epoch: 3, loss: 0.84, accuracy: 0.69, valAccuracy: 0.66 },
      { epoch: 4, loss: 0.59, accuracy: 0.77, valAccuracy: 0.74 },
      { epoch: 5, loss: 0.42, accuracy: 0.83, valAccuracy: 0.81 },
      { epoch: 6, loss: 0.30, accuracy: 0.88, valAccuracy: 0.86 },
      { epoch: 7, loss: 0.21, accuracy: 0.91, valAccuracy: 0.89 },
      { epoch: 8, loss: 0.14, accuracy: 0.93, valAccuracy: 0.91 },
      { epoch: 9, loss: 0.08, accuracy: 0.95, valAccuracy: 0.93 },
      { epoch: 10, loss: 0.019, accuracy: 0.962, valAccuracy: 0.951 }
    ],
    sampleDistribution: [
      { label: 'Limit Order Book Bids', percentage: 42.0, count: '45M ticks' },
      { label: 'Limit Order Book Asks', percentage: 41.5, count: '44M ticks' },
      { label: 'Market Execution Swaps', percentage: 11.5, count: '12M ticks' },
      { label: 'Cancellation / Amend Events', percentage: 5.0, count: '5M ticks' }
    ],
    performanceIndicators: {
      latency: '380 microseconds / Tick',
      throughput: '120,000 Ticks/s',
      memoryFootprint: '1.8 GB RAM',
      hardwareTarget: 'Co-Located FPGA / Rust Daemon'
    },
    techStack: ['Rust', 'Python', 'PyTorch', 'Ray / RLlib', 'Tokio Async'],
    architectureFlow: [
      'Limit Order Book (LOB) Tick Stream',
      'Rust Tokio Async LOB Engine',
      'Order Flow Imbalance Feature Calculator',
      'Zero-Copy FFI Shared Memory Bridge',
      'PPO Actor-Critic Neural Network',
      'Co-Located FIX Protocol Execution'
    ],
    pipeline: [
      {
        title: '01. Ultra-Low Latency LOB Engine in Rust',
        desc: 'Process incoming limit order book tick events in sub-millisecond time using Rust async Tokio runtimes and lock-free rings.',
        code: `// Rust High-Frequency LOB Event Processor
use tokio::sync::mpsc;

#[derive(Debug)]
pub struct TickEvent {
    pub price: f64,
    pub volume: u64,
    pub is_bid: bool,
}

pub async fn process_tick_stream(mut rx: mpsc::Receiver<TickEvent>) {
    while let Some(tick) = rx.recv().await {
        // Execute low-latency feature calculations
        let _imbalance = tick.price * tick.volume as f64;
    }
}`,
        language: 'rust'
      },
      {
        title: '02. PPO Agent Actor-Critic (PyTorch)',
        desc: 'Train an actor-critic neural network with Generalized Advantage Estimation (GAE) for optimal order execution.',
        code: `import torch
import torch.nn as nn

class ActorCriticLOB(nn.Module):
    def __init__(self, input_dim=40, action_dim=3):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)`,
        language: 'python'
      }
    ],
    whitePaper: {
      problem: 'Traditional Python-based quantitative trading backends suffer from garbage collection pauses and high latency jitter, missing microsecond arbitrage and market making execution windows.',
      breakingGround: 'Introduces a hybrid lock-free Rust Tokio async LOB engine coupled via shared memory IPC to a Ray PPO reinforcement learning policy network, guaranteeing sub-millisecond deterministic execution.',
      scope: 'Automated high-frequency market making and optimal order execution for quantitative trading firms, combining Rust for microsecond-level tick processing and PyTorch for deep reinforcement learning.',
      functionality: '1. High-frequency limit order book event processing in Rust Tokio.\n2. Lock-free ring buffer FFI bridge to Python neural policy networks.\n3. PPO actor-critic reinforcement learning for optimal trade routing.\n4. Risk-adjusted reward calculation incorporating Sharpe ratio and inventory penalties.',
      abstract: 'This research paper investigates hybrid Rust-Python architectures for high-frequency reinforcement learning trading systems, achieving sub-millisecond execution latency.',
      methodology: 'Tick data ingestion and feature calculation are implemented in Rust to guarantee deterministic latency under 500 microseconds, passing normalized state tensors to PyTorch via zero-copy shared memory.',
      architecture: 'Tokio async Rust event loop communicating with PyTorch GPU inference workers via shared memory IPC.',
      benchmarks: 'Processed 120,000 order book ticks/sec with P99 latency of 380 microseconds and out-of-sample Sharpe ratio of 3.42.',
      productionConsiderations: 'Deterministic latency guarantees, garbage-collection-free Rust execution loops, and direct exchange FIX protocol co-location.'
    }
  }
];

const DATACAMP_REPOSITORY_PROJECTS: FlagshipProject[] = [
  {
    "id": "dc-blood-donations",
    "title": "Give Life: Predict Blood Donations with AutoML & TPOT",
    "domain": "Healthcare & Binary Classification",
    "difficulty": "Beginner",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Predict whether a donor will return to donate blood using Recency, Frequency, Monetary, Time (RFMTC) metrics and automated machine learning pipelines.",
    "objective": "Maximize ROC-AUC on blood donation prediction using Scikit-Learn and TPOT genetic algorithm pipelines.",
    "metrics": "ROC-AUC: 0.794 \u00b7 Accuracy: 82.4% \u00b7 F1: 0.762",
    "trainingLoss": "0.384 (Binary Cross-Entropy)",
    "trainingAccuracy": "82.4%",
    "problem": "Blood banks face critical supply shortages during emergency surges, needing predictive models to schedule repeat donation reminders efficiently.",
    "breakingGround": "Applies automated machine learning (AutoML via TPOT) to explore thousands of preprocessing and estimator combinations automatically.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 0.68,
        "accuracy": 0.58,
        "valAccuracy": 0.55
      },
      {
        "epoch": 3,
        "loss": 0.52,
        "accuracy": 0.71,
        "valAccuracy": 0.69
      },
      {
        "epoch": 6,
        "loss": 0.43,
        "accuracy": 0.78,
        "valAccuracy": 0.76
      },
      {
        "epoch": 10,
        "loss": 0.384,
        "accuracy": 0.824,
        "valAccuracy": 0.812
      }
    ],
    "sampleDistribution": [
      {
        "label": "Non-Donors (Current Month)",
        "percentage": 76.2,
        "count": "570 individuals"
      },
      {
        "label": "Repeat Donors (Current Month)",
        "percentage": 23.8,
        "count": "178 individuals"
      }
    ],
    "performanceIndicators": {
      "latency": "1.2ms / Donor",
      "throughput": "8,000 queries/s",
      "memoryFootprint": "65 MB RAM",
      "hardwareTarget": "Standard CPU"
    },
    "techStack": [
      "Python",
      "TPOT",
      "Scikit-Learn",
      "Pandas"
    ],
    "architectureFlow": [
      "Blood Transfusion Service CSV Ingestion",
      "RFMTC Feature Extraction",
      "StandardScaler Normalization",
      "TPOT Genetic Pipeline Search",
      "Calibrated Logistic Regression"
    ],
    "pipeline": [
      {
        "title": "01. Data Cleaning & RFMTC Feature Engineering",
        "desc": "Load transfusion records, rename target columns, and analyze donor Recency and Frequency distributions.",
        "code": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler\n\n# Ingest UCI Blood Transfusion Service Center dataset\nurl = \"transfusion.data\"\ndf = pd.read_csv(url)\n\n# Rename columns to standard RFMTC terminology\ndf.rename(columns={\n    'Recency (months)': 'recency',\n    'Frequency (times)': 'frequency',\n    'Monetary (c.c. blood)': 'monetary',\n    'Time (months)': 'time',\n    'whether he/she donated blood in March 2007': 'target'\n}, inplace=True)\n\n# Inspect target class distribution\nprint(\"Target balance:\\n\", df['target'].value_counts(normalize=True))\n\n# Train / validation / test split (80/20)\nX = df.drop(columns=['target'])\ny = df['target']\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y\n)",
        "language": "python"
      },
      {
        "title": "02. TPOT Genetic Algorithm Pipeline Search",
        "desc": "Run genetic programming search over thousands of classifier and transformer combinations.",
        "code": "from tpot import TPOTClassifier\n\n# Configure genetic search parameters\ntpot = TPOTClassifier(\n    generations=5,\n    population_size=20,\n    scoring='roc_auc',\n    cv=5,\n    verbosity=2,\n    random_state=42,\n    n_jobs=-1\n)\n\n# Fit AutoML search on training split\nprint(\"Starting TPOT genetic algorithm search...\")\ntpot.fit(X_train, y_train)\n\n# Evaluate on holdout test set\ntest_score = tpot.score(X_test, y_test)\nprint(f\"Optimal Pipeline Holdout ROC-AUC: {test_score:.4f}\")",
        "language": "python"
      },
      {
        "title": "03. Model Evaluation & Pipeline Export",
        "desc": "Generate precision-recall metrics, export best Python script, and serialize model artifact.",
        "code": "from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix\nimport joblib\n\n# Predict probabilities for ROC-AUC\ny_probs = tpot.predict_proba(X_test)[:, 1]\ny_preds = tpot.predict(X_test)\n\nprint(\"Confusion Matrix:\\n\", confusion_matrix(y_test, y_preds))\nprint(\"\\nClassification Report:\\n\", classification_report(y_test, y_preds))\nprint(f\"Holdout ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}\")\n\n# Export full standalone Python code generated by TPOT\ntpot.export('tpot_blood_pipeline.py')\n\n# Serialize fitted pipeline for production API deployment\njoblib.dump(tpot.fitted_pipeline_, 'blood_donor_model.joblib')\nprint(\"Model saved to blood_donor_model.joblib\")",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Blood transfusion services require deterministic scheduling of prospective donors to maintain optimal regional inventory.",
      "breakingGround": "End-to-end beginner AutoML pipeline utilizing genetic programming to discover optimal classifier ensembles.",
      "scope": "Blood transfusion donor propensity scoring and outreach automation.",
      "functionality": "1. Ingestion of historical donor frequency records.\\n2. Feature scaling and log transforms.\\n3. Genetic pipeline search.\\n4. Precision-recall threshold optimization.",
      "abstract": "DataCamp benchmark project modeling donor retention using UCI Blood Transfusion Service Center data.",
      "methodology": "Genetic algorithm-driven pipeline synthesis evaluated via holdout cross-validation.",
      "architecture": "Modular Python Scikit-Learn & TPOT classification workflow.",
      "benchmarks": "Achieved 0.794 ROC-AUC on holdout validation data.",
      "productionConsiderations": "Lightweight model artifact deployable on low-cost serverless functions."
    }
  },
  {
    "id": "dc-energy-consumption",
    "title": "Predicting Energy Consumption with Time-Series Regression",
    "domain": "Clean Energy & Tabular Regression",
    "difficulty": "Beginner",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Forecast hourly megawatt electricity consumption using calendar features, cyclical encodings, and Ridge regression.",
    "objective": "Predict power grid load demands with Mean Absolute Percentage Error (MAPE) < 4.5%.",
    "metrics": "R\u00b2 Score: 0.887 \u00b7 RMSE: 142.5 MW \u00b7 MAPE: 3.8%",
    "trainingLoss": "0.112 (MSE Loss)",
    "trainingAccuracy": "88.7% (Explained Variance)",
    "problem": "Power utility grid operators must balance energy generation with consumption in real-time to prevent blackouts and optimize turbine efficiency.",
    "breakingGround": "Applies cyclical sine/cosine time transformations and temperature lag indicators to capture strong diurnal and seasonal electricity usage cycles.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 1.1,
        "accuracy": 0.45,
        "valAccuracy": 0.42
      },
      {
        "epoch": 3,
        "loss": 0.48,
        "accuracy": 0.72,
        "valAccuracy": 0.7
      },
      {
        "epoch": 6,
        "loss": 0.22,
        "accuracy": 0.83,
        "valAccuracy": 0.81
      },
      {
        "epoch": 10,
        "loss": 0.112,
        "accuracy": 0.887,
        "valAccuracy": 0.879
      }
    ],
    "sampleDistribution": [
      {
        "label": "Off-Peak Night Hours",
        "percentage": 41.5,
        "count": "36,400 hours"
      },
      {
        "label": "Standard Daytime Hours",
        "percentage": 38.2,
        "count": "33,500 hours"
      },
      {
        "label": "Peak Summer/Winter Surge",
        "percentage": 20.3,
        "count": "17,800 hours"
      }
    ],
    "performanceIndicators": {
      "latency": "3ms / Prediction",
      "throughput": "4,500 req/s",
      "memoryFootprint": "95 MB RAM",
      "hardwareTarget": "Standard CPU"
    },
    "techStack": [
      "Python",
      "Scikit-Learn",
      "Statsmodels",
      "Pandas"
    ],
    "architectureFlow": [
      "Hourly PJM Grid Ingestion",
      "Cyclical Sine/Cosine Feature Engineering",
      "Temperature Lag Correlation",
      "Ridge & Lasso Linear Models",
      "Grid Load Forecast"
    ],
    "pipeline": [
      {
        "title": "01. Time Series Cyclical Decomposition & Lag Features",
        "desc": "Extract harmonic sine/cosine features and lag variables from hourly timestamps.",
        "code": "import numpy as np\nimport pandas as pd\n\n# Load hourly megawatt consumption records\ndf = pd.read_csv('PJME_hourly.csv', parse_dates=['Datetime'], index_col='Datetime')\ndf = df.sort_index()\n\n# Engineer cyclical diurnal and seasonal features\ndf['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24.0)\ndf['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24.0)\ndf['month_sin'] = np.sin(2 * np.pi * df.index.month / 12.0)\ndf['month_cos'] = np.cos(2 * np.pi * df.index.month / 12.0)\ndf['dayofweek'] = df.index.dayofweek\n\n# Lagged load indicators (24h prior, 168h prior week)\ndf['lag_24h'] = df['PJME_MW'].shift(24)\ndf['lag_168h'] = df['PJME_MW'].shift(168)\ndf.dropna(inplace=True)",
        "language": "python"
      },
      {
        "title": "02. Regularized Ridge Regression & Cross-Validation",
        "desc": "Fit L2 regularized Ridge model with alpha search over chronological splits.",
        "code": "from sklearn.linear_model import RidgeCV\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import Pipeline\n\nfeatures = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dayofweek', 'lag_24h', 'lag_168h']\ntarget = 'PJME_MW'\n\n# Chronological train/test split to avoid data leakage\nsplit_date = '2016-01-01'\ntrain = df.loc[df.index < split_date]\ntest = df.loc[df.index >= split_date]\n\npipeline = Pipeline([\n    ('scaler', StandardScaler()),\n    ('ridge', RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5))\n])\n\npipeline.fit(train[features], train[target])\nprint(f\"Optimal Alpha: {pipeline.named_steps['ridge'].alpha_:.3f}\")",
        "language": "python"
      },
      {
        "title": "03. Residual Diagnostics & Model Serialization",
        "desc": "Calculate RMSE, MAE, R-squared, and persist model artifact for inference.",
        "code": "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\nimport joblib\n\npreds = pipeline.predict(test[features])\nactuals = test[target]\n\nrmse = np.sqrt(mean_squared_error(actuals, preds))\nmae = mean_absolute_error(actuals, preds)\nr2 = r2_score(actuals, preds)\nmape = np.mean(np.abs((actuals - preds) / actuals)) * 100\n\nprint(f\"Test R\u00b2 Score: {r2:.4f}\")\nprint(f\"Test RMSE: {rmse:.2f} MW\")\nprint(f\"Test MAPE: {mape:.2f}%\")\n\njoblib.dump(pipeline, 'energy_grid_forecast.joblib')\nprint(\"Model serialized to energy_grid_forecast.joblib\")",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Power distribution grids suffer from volatility due to fluctuating weather patterns and diurnal human activity.",
      "breakingGround": "Demonstrates lightweight feature engineering transforming non-stationary timestamps into cyclical harmonic regressors.",
      "scope": "Municipal grid load prediction and renewable energy dispatch scheduling.",
      "functionality": "1. Time-series timestamp decomposition.\\n2. Harmonic feature transformation.\\n3. Regularized linear regression.\\n4. Residual error decomposition.",
      "abstract": "DataCamp foundational regression project modeling regional energy consumption patterns.",
      "methodology": "Supervised regression with chronological time-series splitting.",
      "architecture": "Clean Python Statsmodels & Scikit-Learn forecasting pipeline.",
      "benchmarks": "Achieved 0.887 R\u00b2 score and 3.8% MAPE on multi-year holdout data.",
      "productionConsiderations": "Extremely fast inference suitable for IoT edge smart meters."
    }
  },
  {
    "id": "dc-customer-churn",
    "title": "Customer Churn Prediction with Logistic Regression & Random Forests",
    "domain": "Predictive Analytics & Telecom",
    "difficulty": "Beginner",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Predict customer churn using exploratory data analysis, feature encoding, and classification models.",
    "objective": "Maximize ROC-AUC and identify key churn indicators for proactive customer retention.",
    "metrics": "ROC-AUC: 0.892 \u00b7 Accuracy: 87.5% \u00b7 Recall: 84.1%",
    "trainingLoss": "0.241 (Binary Cross-Entropy)",
    "trainingAccuracy": "88.1%",
    "problem": "Subscription businesses suffer from high customer acquisition costs offset by rapid churn, requiring predictive identification of at-risk accounts.",
    "breakingGround": "End-to-end beginner-to-intermediate pipeline demonstrating feature engineering, handling class imbalance with SMOTE, and hyperparameter tuning.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 0.65,
        "accuracy": 0.62,
        "valAccuracy": 0.6
      },
      {
        "epoch": 3,
        "loss": 0.45,
        "accuracy": 0.75,
        "valAccuracy": 0.73
      },
      {
        "epoch": 5,
        "loss": 0.32,
        "accuracy": 0.82,
        "valAccuracy": 0.8
      },
      {
        "epoch": 8,
        "loss": 0.241,
        "accuracy": 0.881,
        "valAccuracy": 0.875
      }
    ],
    "sampleDistribution": [
      {
        "label": "Retained Customers",
        "percentage": 73.5,
        "count": "5,880 accounts"
      },
      {
        "label": "Churned Customers",
        "percentage": 26.5,
        "count": "2,120 accounts"
      }
    ],
    "performanceIndicators": {
      "latency": "4ms / Inference",
      "throughput": "2,500 req/s",
      "memoryFootprint": "120 MB RAM",
      "hardwareTarget": "CPU / Edge"
    },
    "techStack": [
      "Python",
      "Scikit-Learn",
      "Pandas",
      "Seaborn"
    ],
    "architectureFlow": [
      "Raw CSV Data Ingestion",
      "Missing Value Imputation",
      "One-Hot Encoding",
      "Random Forest Classifier",
      "Model Evaluation & ROC Curve"
    ],
    "pipeline": [
      {
        "title": "01. Preprocessing, One-Hot Encoding & Missing Values",
        "desc": "Handle numeric coercion on TotalCharges, impute zeros, and encode categorical subscription attributes.",
        "code": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n\ndf = pd.read_csv('telecom_churn.csv')\n\n# Handle whitespace in numeric TotalCharges column\ndf['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip(), errors='coerce').fillna(0)\ndf.drop(columns=['customerID'], inplace=True)\n\n# Encode binary target\ndf['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})\n\n# One-hot encode remaining categorical features\nX = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)\ny = df['Churn']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y\n)",
        "language": "python"
      },
      {
        "title": "02. Random Forest Classifier & Hyperparameter Grid Search",
        "desc": "Tune tree depth, minimum samples per leaf, and estimators using 5-fold cross-validation.",
        "code": "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.model_selection import GridSearchCV\n\nparam_grid = {\n    'n_estimators': [100, 200],\n    'max_depth': [6, 8, 10],\n    'min_samples_leaf': [2, 4],\n    'class_weight': ['balanced']\n}\n\nrf = RandomForestClassifier(random_state=42)\ngrid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)\ngrid.fit(X_train, y_train)\n\nbest_rf = grid.best_estimator_\nprint(f\"Best Parameters: {grid.best_params_}\")",
        "language": "python"
      },
      {
        "title": "03. ROC-AUC Evaluation & Feature Importance Analysis",
        "desc": "Assess holdout performance, plot feature importances, and serialize model artifact.",
        "code": "from sklearn.metrics import roc_auc_score, classification_report\nimport joblib\n\ny_pred_proba = best_rf.predict_proba(X_test)[:, 1]\ny_pred = best_rf.predict(X_test)\n\nprint(f\"Holdout ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}\")\nprint(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n\n# Extract top driving features for customer churn\nfeature_importance = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)\nprint(\"Top 5 Churn Drivers:\\n\", feature_importance.head(5))\n\njoblib.dump(best_rf, 'telecom_churn_model.joblib')",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Subscription businesses suffer from high customer acquisition costs offset by rapid churn.",
      "breakingGround": "Imported DataCamp reference implementation for tabular classification.",
      "scope": "Beginner-to-intermediate churn prediction pipeline.",
      "functionality": "1. Exploratory data analysis.\\n2. Logistic regression and random forest classification.\\n3. Evaluation via ROC-AUC and precision-recall.",
      "abstract": "DataCamp benchmark project for predicting customer churn using Scikit-Learn.",
      "methodology": "Supervised classification with cross-validation.",
      "architecture": "Standard Python data science notebook pipeline.",
      "benchmarks": "Achieved 0.892 ROC-AUC on holdout test set.",
      "productionConsiderations": "Easy containerization and batch scoring via cron."
    }
  },
  {
    "id": "dc-mnist-digits",
    "title": "Handwritten Digit Classification with MNIST & Neural Networks",
    "domain": "Computer Vision & Deep Learning",
    "difficulty": "Beginner",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Build your first deep neural network classifier to recognize handwritten digits from the MNIST benchmark dataset.",
    "objective": "Achieve >97% test accuracy using multi-layer perceptrons in PyTorch/Keras.",
    "metrics": "Test Accuracy: 98.2% \u00b7 Test Loss: 0.058",
    "trainingLoss": "0.058 (Categorical Cross-Entropy)",
    "trainingAccuracy": "98.2%",
    "problem": "Recognizing optical character digits automatically is the foundational benchmark task for computer vision and neural networks.",
    "breakingGround": "Introduces neural network fundamentals, activation functions, backpropagation, and stochastic gradient descent.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 0.65,
        "accuracy": 0.84,
        "valAccuracy": 0.83
      },
      {
        "epoch": 3,
        "loss": 0.22,
        "accuracy": 0.93,
        "valAccuracy": 0.92
      },
      {
        "epoch": 5,
        "loss": 0.058,
        "accuracy": 0.982,
        "valAccuracy": 0.979
      }
    ],
    "sampleDistribution": [
      {
        "label": "Digit Classes (0-9)",
        "percentage": 100.0,
        "count": "70,000 images"
      }
    ],
    "performanceIndicators": {
      "latency": "0.5ms / Image",
      "throughput": "15,000 images/s",
      "memoryFootprint": "300 MB RAM",
      "hardwareTarget": "CPU / GPU"
    },
    "techStack": [
      "Python",
      "PyTorch",
      "Torchvision",
      "NumPy"
    ],
    "architectureFlow": [
      "28x28 Image Ingestion",
      "Flattening Layer",
      "Linear Hidden Layers (ReLU)",
      "Softmax Output"
    ],
    "pipeline": [
      {
        "title": "01. PyTorch Dataset Loader & Preprocessing Pipeline",
        "desc": "Download MNIST, normalize grayscale tensors, and instantiate batched DataLoaders.",
        "code": "import torch\nimport torchvision\nimport torchvision.transforms as transforms\nfrom torch.utils.data import DataLoader\n\ntransform = transforms.Compose([\n    transforms.ToTensor(),\n    transforms.Normalize((0.1307,), (0.3081,))\n])\n\ntrain_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)\ntest_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)\n\ntrain_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)\ntest_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)",
        "language": "python"
      },
      {
        "title": "02. Deep Neural Network Architecture & Training Loop",
        "desc": "Construct a multi-layer perceptron with dropout and train using Adam optimizer.",
        "code": "import torch.nn as nn\nimport torch.optim as optim\n\nclass MNISTMLP(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.net = nn.Sequential(\n            nn.Flatten(),\n            nn.Linear(28*28, 256),\n            nn.ReLU(),\n            nn.Dropout(0.2),\n            nn.Linear(256, 128),\n            nn.ReLU(),\n            nn.Linear(128, 10)\n        )\n    def forward(self, x):\n        return self.net(x)\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel = MNISTMLP().to(device)\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.Adam(model.parameters(), lr=0.001)\n\n# Training loop across 5 epochs\nfor epoch in range(5):\n    model.train()\n    for images, labels in train_loader:\n        images, labels = images.to(device), labels.to(device)\n        optimizer.zero_grad()\n        loss = criterion(model(images), labels)\n        loss.backward()\n        optimizer.step()",
        "language": "python"
      },
      {
        "title": "03. Test Evaluation & ONNX Model Serialization",
        "desc": "Evaluate top-1 test accuracy and export model to standard ONNX runtime format.",
        "code": "model.eval()\ncorrect = 0\ntotal = 0\n\nwith torch.no_grad():\n    for images, labels in test_loader:\n        images, labels = images.to(device), labels.to(device)\n        outputs = model(images)\n        _, predicted = torch.max(outputs.data, 1)\n        total += labels.size(0)\n        correct += (predicted == labels).sum().item()\n\naccuracy = 100 * correct / total\nprint(f\"Holdout Test Accuracy: {accuracy:.2f}%\")\n\n# Export to ONNX format for cross-platform low-latency execution\ndummy_input = torch.randn(1, 1, 28, 28, device=device)\ntorch.onnx.export(model, dummy_input, \"mnist_mlp.onnx\", input_names=['input'], output_names=['output'])\nprint(\"Model exported to mnist_mlp.onnx\")",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Optical character recognition of handwritten digits.",
      "breakingGround": "DataCamp introductory deep learning project.",
      "scope": "Image classification using Multi-Layer Perceptrons.",
      "functionality": "1. MNIST dataset loading.\\n2. Neural network forward and backward passes.",
      "abstract": "DataCamp benchmark deep learning tutorial.",
      "methodology": "Feedforward neural network trained with SGD.",
      "architecture": "PyTorch modular neural network.",
      "benchmarks": "Achieved 98.2% accuracy on MNIST test set.",
      "productionConsiderations": "Ideal educational starting point for deep learning frameworks."
    }
  },
  {
    "id": "dc-penguins-clustering",
    "title": "Clustering Antarctic Penguin Species with K-Means & PCA",
    "domain": "Bioinformatics & Unsupervised Learning",
    "difficulty": "Intermediate",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Group Palmer Archipelago penguin species by physical bill length, flipper length, and body mass using PCA and K-Means clustering.",
    "objective": "Discover natural taxonomic clusters with Silhouette Score > 0.65 without ground-truth species labels.",
    "metrics": "Silhouette Score: 0.684 \u00b7 Adjusted Rand Index: 0.871 \u00b7 Inertia SSE: 42.1",
    "trainingLoss": "0.042 (K-Means Inertia SSE)",
    "trainingAccuracy": "93.6% (Cluster Purity)",
    "problem": "Field biologists require automated morphometric clustering to classify species variations without invasive genetic testing.",
    "breakingGround": "Combines Principal Component Analysis (PCA) dimensionality reduction with the elbow method to discover true biological species boundaries.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 0.89,
        "accuracy": 0.6,
        "valAccuracy": 0.58
      },
      {
        "epoch": 3,
        "loss": 0.35,
        "accuracy": 0.81,
        "valAccuracy": 0.79
      },
      {
        "epoch": 6,
        "loss": 0.12,
        "accuracy": 0.9,
        "valAccuracy": 0.88
      },
      {
        "epoch": 10,
        "loss": 0.042,
        "accuracy": 0.936,
        "valAccuracy": 0.925
      }
    ],
    "sampleDistribution": [
      {
        "label": "Ad\u00e9lie Penguins",
        "percentage": 44.1,
        "count": "152 specimens"
      },
      {
        "label": "Gentoo Penguins",
        "percentage": 36.0,
        "count": "124 specimens"
      },
      {
        "label": "Chinstrap Penguins",
        "percentage": 19.9,
        "count": "68 specimens"
      }
    ],
    "performanceIndicators": {
      "latency": "0.8ms / Sample",
      "throughput": "12,000 samples/s",
      "memoryFootprint": "45 MB RAM",
      "hardwareTarget": "Standard CPU"
    },
    "techStack": [
      "Python",
      "Scikit-Learn",
      "PCA",
      "Seaborn"
    ],
    "architectureFlow": [
      "Palmer Penguins CSV Ingestion",
      "StandardScaler Feature Scaling",
      "Principal Component Analysis (2D)",
      "Elbow Method & Silhouette Analysis",
      "K-Means Cluster Assignment"
    ],
    "pipeline": [
      {
        "title": "01. Morphometric Scaling & PCA Dimensionality Reduction",
        "desc": "Scale 4 physical dimensions and project into 2 orthogonal principal components.",
        "code": "import pandas as pd\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\ndf = pd.read_csv('penguins.csv').dropna()\nfeatures = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']\nX = df[features]\ny_true = df['species']\n\n# Standardize to zero mean and unit variance\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n\npca = PCA(n_components=2)\nX_pca = pca.fit_transform(X_scaled)\nprint(f\"Explained Variance Ratio: {pca.explained_variance_ratio_.sum():.2%}\")",
        "language": "python"
      },
      {
        "title": "02. Elbow Curve & Silhouette Score Analysis",
        "desc": "Search across cluster counts (k=2 to 7) to identify the optimal taxonomic cluster count.",
        "code": "from sklearn.cluster import KMeans\nfrom sklearn.metrics import silhouette_score\n\ninertias = []\nsilhouettes = []\nk_range = range(2, 7)\n\nfor k in k_range:\n    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n    labels = kmeans.fit_predict(X_pca)\n    inertias.append(kmeans.inertia_)\n    silhouettes.append(silhouette_score(X_pca, labels))\n    print(f\"k={k} -> Silhouette Score: {silhouettes[-1]:.3f}\")\n\n# Best cluster count is k=3\nbest_k = 3",
        "language": "python"
      },
      {
        "title": "03. K-Means Clustering & Taxonomic Contingency Matrix",
        "desc": "Assign samples to clusters and cross-tabulate with biological ground-truth labels.",
        "code": "from sklearn.metrics import adjusted_rand_score\nimport joblib\n\nfinal_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)\ncluster_labels = final_kmeans.fit_predict(X_pca)\n\ndf['cluster'] = cluster_labels\ncontingency = pd.crosstab(df['species'], df['cluster'])\nprint(\"Taxonomic Contingency Matrix:\\n\", contingency)\n\nari = adjusted_rand_score(y_true, cluster_labels)\nprint(f\"Adjusted Rand Index (ARI): {ari:.4f}\")\n\njoblib.dump(final_kmeans, 'penguin_kmeans_model.joblib')",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Taxonomic separation of morphologically similar avian populations in remote Antarctic habitats.",
      "breakingGround": "Demonstrates unsupervised feature projection isolating biological clusters without supervision.",
      "scope": "Biological morphometric clustering and phenotypic analysis.",
      "functionality": "1. Multi-feature scaling and outlier detection.\\n2. Eigenvector decomposition via PCA.\\n3. K-Means optimization.\\n4. Silhouette validation.",
      "abstract": "DataCamp intermediate unsupervised learning reference implementation.",
      "methodology": "Principal component projection followed by expectation-maximization K-Means clustering.",
      "architecture": "Python Scikit-Learn unsupervised pipeline.",
      "benchmarks": "Achieved 0.684 silhouette coefficient and 0.871 ARI.",
      "productionConsiderations": "Deterministic offline execution with zero training overhead."
    }
  },
  {
    "id": "dc-fraud-detection",
    "title": "Credit Card Fraud Detection with Imbalanced Data Techniques",
    "domain": "Financial Security & Anomaly Detection",
    "difficulty": "Intermediate",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Detect fraudulent credit card transactions amidst extreme class imbalance using SMOTE and Isolation Forests.",
    "objective": "Maximize Precision-Recall AUC on highly imbalanced financial transaction streams.",
    "metrics": "PR-AUC: 0.865 \u00b7 Specificity: 99.8% \u00b7 Recall: 86.4%",
    "trainingLoss": "0.092 (Focal Loss)",
    "trainingAccuracy": "96.2% (Specificity)",
    "problem": "Credit card fraud represents less than 0.1% of transactions, causing standard classifiers to predict all transactions as legitimate.",
    "breakingGround": "Demonstrates synthetic minority oversampling technique (SMOTE) combined with ensemble gradient boosting and cost-sensitive learning.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 0.85,
        "accuracy": 0.99,
        "valAccuracy": 0.99
      },
      {
        "epoch": 5,
        "loss": 0.28,
        "accuracy": 0.96,
        "valAccuracy": 0.95
      },
      {
        "epoch": 10,
        "loss": 0.092,
        "accuracy": 0.962,
        "valAccuracy": 0.958
      }
    ],
    "sampleDistribution": [
      {
        "label": "Legitimate Transactions",
        "percentage": 99.8,
        "count": "284,315 txns"
      },
      {
        "label": "Fraudulent Transactions",
        "percentage": 0.2,
        "count": "492 txns"
      }
    ],
    "performanceIndicators": {
      "latency": "2ms / Transaction",
      "throughput": "10,000 txns/s",
      "memoryFootprint": "512 MB RAM",
      "hardwareTarget": "Fraud Scoring Gateway"
    },
    "techStack": [
      "Python",
      "Imbalanced-learn",
      "LightGBM",
      "Scikit-Learn"
    ],
    "architectureFlow": [
      "Transaction Stream Ingestion",
      "SMOTE Oversampling",
      "LightGBM Classifier",
      "Real-time Alert Gateway"
    ],
    "pipeline": [
      {
        "title": "01. Robust Scaling & SMOTE Minority Oversampling",
        "desc": "Scale skewed transaction Amount and Time features and synthesize minority fraud examples.",
        "code": "import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import RobustScaler\nfrom imblearn.over_sampling import SMOTE\n\ndf = pd.read_csv('creditcard.csv')\nrob_scaler = RobustScaler()\ndf['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))\ndf['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1, 1))\ndf.drop(['Time', 'Amount'], axis=1, inplace=True)\n\nX = df.drop('Class', axis=1)\ny = df['Class']\n\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y\n)\n\n# Apply SMOTE exclusively on training data\nsmote = SMOTE(sampling_strategy=0.2, random_state=42)\nX_train_res, y_train_res = smote.fit_resample(X_train, y_train)",
        "language": "python"
      },
      {
        "title": "02. Cost-Sensitive LightGBM Classifier Training",
        "desc": "Train gradient boosted decision trees with focal loss weighting to penalize missed frauds.",
        "code": "from lightgbm import LGBMClassifier\n\nclf = LGBMClassifier(\n    n_estimators=200,\n    learning_rate=0.05,\n    max_depth=6,\n    scale_pos_weight=5.0,\n    random_state=42,\n    n_jobs=-1\n)\n\nclf.fit(X_train_res, y_train_res)",
        "language": "python"
      },
      {
        "title": "03. Precision-Recall AUC & Decision Threshold Tuning",
        "desc": "Calculate average precision score and determine the optimal probability cutoff threshold.",
        "code": "from sklearn.metrics import average_precision_score, precision_recall_curve, classification_report\nimport joblib\n\ny_prob = clf.predict_proba(X_test)[:, 1]\npr_auc = average_precision_score(y_test, y_prob)\nprint(f\"Precision-Recall AUC (PR-AUC): {pr_auc:.4f}\")\n\nprecisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)\n# Select decision threshold maximizing F1 score\nfscores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)\noptimal_idx = fscores.argmax()\noptimal_threshold = thresholds[optimal_idx]\n\nprint(f\"Optimal Threshold: {optimal_threshold:.4f}\")\ny_pred_opt = (y_prob >= optimal_threshold).astype(int)\nprint(classification_report(y_test, y_pred_opt))\n\njoblib.dump(clf, 'fraud_detection_lgbm.joblib')",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Extreme class imbalance in financial fraud detection.",
      "breakingGround": "DataCamp project implementing SMOTE and cost-sensitive boosting.",
      "scope": "Credit card fraud anomaly detection.",
      "functionality": "1. SMOTE oversampling.\\n2. LightGBM probability calibration.",
      "abstract": "DataCamp reference project for handling imbalanced datasets.",
      "methodology": "Ensemble classification with oversampling.",
      "architecture": "Low-latency scoring service.",
      "benchmarks": "Achieved 0.865 PR-AUC on skewed transaction logs.",
      "productionConsiderations": "Real-time scoring within 5ms latency threshold."
    }
  },
  {
    "id": "dc-sentiment-imdb",
    "title": "Sentiment Analysis of Financial News & Customer Reviews",
    "domain": "Natural Language Processing & Text Classification",
    "difficulty": "Intermediate",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Classify sentiment polarity in financial headlines and customer reviews using VADER, TF-IDF, and Naive Bayes.",
    "objective": "Achieve >88% classification accuracy on unstructured text reviews and headlines.",
    "metrics": "Accuracy: 88.6% \u00b7 F1-Score: 0.884 \u00b7 AUC: 0.925",
    "trainingLoss": "0.210 (Cross-Entropy)",
    "trainingAccuracy": "88.6%",
    "problem": "Understanding market sentiment from thousands of unstructured news articles requires automated semantic vectorization and classification.",
    "breakingGround": "Combines lexicon-based VADER sentiment intensity scoring with statistical TF-IDF n-gram vectorization and Naive Bayes classifiers.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 0.69,
        "accuracy": 0.55,
        "valAccuracy": 0.54
      },
      {
        "epoch": 3,
        "loss": 0.42,
        "accuracy": 0.78,
        "valAccuracy": 0.76
      },
      {
        "epoch": 5,
        "loss": 0.21,
        "accuracy": 0.886,
        "valAccuracy": 0.88
      }
    ],
    "sampleDistribution": [
      {
        "label": "Positive / Bullish Reviews",
        "percentage": 50.0,
        "count": "25,000 reviews"
      },
      {
        "label": "Negative / Bearish Reviews",
        "percentage": 50.0,
        "count": "25,000 reviews"
      }
    ],
    "performanceIndicators": {
      "latency": "1ms / Review",
      "throughput": "5,000 reviews/s",
      "memoryFootprint": "180 MB RAM",
      "hardwareTarget": "CPU Server"
    },
    "techStack": [
      "Python",
      "Scikit-Learn",
      "NLTK",
      "Pandas"
    ],
    "architectureFlow": [
      "Raw Review Corpus Ingestion",
      "TF-IDF Vectorizer",
      "Multinomial Naive Bayes",
      "Sentiment Prediction"
    ],
    "pipeline": [
      {
        "title": "01. Text Tokenization & Sublinear TF-IDF Vectorization",
        "desc": "Clean text, filter English stopwords, and extract unigram/bigram term matrices.",
        "code": "import pandas as pd\nimport re\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.model_selection import train_test_split\n\ndf = pd.read_csv('IMDB_Dataset.csv')\n\ndef clean_text(text):\n    text = re.sub(r'<.*?>', '', text)  # remove HTML\n    text = re.sub(r'[^a-zA-Z\\s]', '', text)  # remove non-alphanumeric\n    return text.lower().strip()\n\ndf['cleaned'] = df['review'].apply(clean_text)\ny = df['sentiment'].map({'positive': 1, 'negative': 0})\n\nX_train, X_test, y_train, y_test = train_test_split(df['cleaned'], y, test_size=0.2, random_state=42)\n\nvectorizer = TfidfVectorizer(\n    max_features=10000,\n    ngram_range=(1, 2),\n    sublinear_tf=True,\n    stop_words='english'\n)\nX_train_tfidf = vectorizer.fit_transform(X_train)\nX_test_tfidf = vectorizer.transform(X_test)",
        "language": "python"
      },
      {
        "title": "02. Calibrated Naive Bayes Classifier Fitting",
        "desc": "Fit MultinomialNB and calibrate probability estimates with Platt scaling.",
        "code": "from sklearn.naive_bayes import MultinomialNB\nfrom sklearn.calibration import CalibratedClassifierCV\n\nbase_nb = MultinomialNB(alpha=0.5)\ncalibrated_nb = CalibratedClassifierCV(base_nb, method='sigmoid', cv=5)\ncalibrated_nb.fit(X_train_tfidf, y_train)",
        "language": "python"
      },
      {
        "title": "03. Classification Report & Real-Time Inference Function",
        "desc": "Evaluate test metrics and construct a sub-millisecond inference prediction function.",
        "code": "from sklearn.metrics import classification_report, roc_auc_score\nimport joblib\n\ny_pred = calibrated_nb.predict(X_test_tfidf)\ny_prob = calibrated_nb.predict_proba(X_test_tfidf)[:, 1]\n\nprint(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))\nprint(f\"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\")\n\ndef predict_sentiment(text):\n    cleaned = clean_text(text)\n    vec = vectorizer.transform([cleaned])\n    prob = calibrated_nb.predict_proba(vec)[0][1]\n    return {'sentiment': 'Positive' if prob >= 0.5 else 'Negative', 'confidence': prob}\n\n# Test sample inference\nprint(predict_sentiment(\"This film had breathtaking cinematography and phenomenal acting.\"))\n\njoblib.dump((vectorizer, calibrated_nb), 'sentiment_pipeline.joblib')",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Automated sentiment extraction from unstructured textual corpora.",
      "breakingGround": "DataCamp reference NLP pipeline using TF-IDF and Naive Bayes.",
      "scope": "Text sentiment classification.",
      "functionality": "1. Text tokenization and stopword filtering.\\n2. TF-IDF feature extraction.\\n3. Naive Bayes classification.",
      "abstract": "DataCamp intermediate NLP project.",
      "methodology": "Probabilistic bag-of-words classification.",
      "architecture": "Python Scikit-Learn text processing pipeline.",
      "benchmarks": "Achieved 88.6% accuracy on IMDb review test set.",
      "productionConsiderations": "Lightweight model with sub-millisecond inference time."
    }
  },
  {
    "id": "dc-artwork-resnet",
    "title": "Artwork & Image Classification with Transfer Learning (ResNet)",
    "domain": "Computer Vision & Deep Learning",
    "difficulty": "Intermediate",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Classify fine art artistic movements (Baroque, Impressionism, Cubism) using a pretrained PyTorch ResNet-50 backbone.",
    "objective": "Reach >89% classification accuracy on fine art imagery using transfer learning.",
    "metrics": "Top-1 Accuracy: 89.4% \u00b7 Top-5 Accuracy: 96.8% \u00b7 F1: 0.891",
    "trainingLoss": "0.134 (Cross-Entropy Loss)",
    "trainingAccuracy": "91.2%",
    "problem": "Training high-accuracy deep convolutional vision models from scratch requires millions of labeled images and massive GPU compute clusters.",
    "breakingGround": "Leverages ImageNet pre-trained representations, freezing early feature extractor layers and fine-tuning dense classification heads with differential learning rates.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 1.45,
        "accuracy": 0.61,
        "valAccuracy": 0.58
      },
      {
        "epoch": 3,
        "loss": 0.72,
        "accuracy": 0.79,
        "valAccuracy": 0.77
      },
      {
        "epoch": 6,
        "loss": 0.31,
        "accuracy": 0.88,
        "valAccuracy": 0.86
      },
      {
        "epoch": 10,
        "loss": 0.134,
        "accuracy": 0.912,
        "valAccuracy": 0.894
      }
    ],
    "sampleDistribution": [
      {
        "label": "Impressionism & Realism",
        "percentage": 42.0,
        "count": "8,400 paintings"
      },
      {
        "label": "Baroque & Renaissance",
        "percentage": 35.5,
        "count": "7,100 paintings"
      },
      {
        "label": "Modern & Abstract Expressionism",
        "percentage": 22.5,
        "count": "4,500 paintings"
      }
    ],
    "performanceIndicators": {
      "latency": "18ms / Image (GPU)",
      "throughput": "60 images/s",
      "memoryFootprint": "2.4 GB VRAM",
      "hardwareTarget": "NVIDIA T4 / RTX 3080"
    },
    "techStack": [
      "Python",
      "PyTorch",
      "Torchvision",
      "Pillow"
    ],
    "architectureFlow": [
      "Artwork Image Ingestion (224x224)",
      "Random Crop & Color Jitter Augmentation",
      "ResNet-50 Pretrained Backbone",
      "Adaptive Average Pooling",
      "Softmax Genre Classification"
    ],
    "pipeline": [
      {
        "title": "01. PyTorch Image Transforms & Data Augmentation",
        "desc": "Define ImageNet normalization, geometric rotation, and color jitter augmentations.",
        "code": "import torch\nimport torchvision.transforms as transforms\nfrom torchvision.datasets import ImageFolder\nfrom torch.utils.data import DataLoader\n\ntrain_transforms = transforms.Compose([\n    transforms.Resize((256, 256)),\n    transforms.RandomCrop(224),\n    transforms.RandomHorizontalFlip(),\n    transforms.ColorJitter(brightness=0.2, contrast=0.2),\n    transforms.ToTensor(),\n    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n])\n\ntrain_data = ImageFolder(root='wikiart/train', transform=train_transforms)\ntrain_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)",
        "language": "python"
      },
      {
        "title": "02. ResNet-50 Transfer Learning Architecture",
        "desc": "Load pre-trained convolutional backbone, freeze early weights, and attach custom multi-class head.",
        "code": "import torch.nn as nn\nfrom torchvision import models\n\nmodel = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)\n\n# Freeze convolutional feature extraction parameters\nfor param in model.parameters():\n    param.requires_grad = False\n\n# Replace final fully connected layer for 10 art genres\nnum_features = model.fc.in_features\nmodel.fc = nn.Sequential(\n    nn.Linear(num_features, 512),\n    nn.ReLU(),\n    nn.Dropout(0.3),\n    nn.Linear(512, 10)\n)",
        "language": "python"
      },
      {
        "title": "03. Fine-Tuning Loop with Cosine Learning Rate Schedule",
        "desc": "Train custom classification layers with AdamW and save model checkpoint.",
        "code": "import torch.optim as optim\n\ndevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nmodel = model.to(device)\n\ncriterion = nn.CrossEntropyLoss()\noptimizer = optim.AdamW(model.fc.parameters(), lr=0.001, weight_decay=0.01)\nscheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)\n\n# 10 Epochs Fine-Tuning\nfor epoch in range(10):\n    model.train()\n    running_loss = 0.0\n    for images, labels in train_loader:\n        images, labels = images.to(device), labels.to(device)\n        optimizer.zero_grad()\n        outputs = model(images)\n        loss = criterion(outputs, labels)\n        loss.backward()\n        optimizer.step()\n        running_loss += loss.item()\n    scheduler.step()\n\ntorch.save(model.state_dict(), 'resnet_art_classifier.pth')\nprint(\"Model checkpoint saved to resnet_art_classifier.pth\")",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Automated curation and archival categorization of historical artwork collections.",
      "breakingGround": "Demonstrates modern deep transfer learning, achieving production vision accuracy in under 10 epochs.",
      "scope": "Computer vision classification for cultural artifacts and art history.",
      "functionality": "1. PyTorch DataLoader with geometric augmentations.\\n2. Feature-frozen residual networks.\\n3. Fine-tuning classification head.",
      "abstract": "DataCamp intermediate deep learning reference project.",
      "methodology": "Transfer learning from ImageNet-1k checkpoints fine-tuned with AdamW.",
      "architecture": "PyTorch Torchvision modular convolutional pipeline.",
      "benchmarks": "Achieved 89.4% top-1 accuracy on WikiArt genre validation set.",
      "productionConsiderations": "Deployable on Triton Inference Server or TorchScript."
    }
  },
  {
    "id": "dc-house-prices",
    "title": "House Price Prediction & Advanced Feature Engineering",
    "domain": "Real Estate & Tabular Regression",
    "difficulty": "Advanced",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Predict residential property sale prices using 79 structural variables, Box-Cox transformations, and XGBoost ensembles.",
    "objective": "Minimize Root Mean Squared Logarithmic Error (RMSLE) below 0.12 on housing evaluation benchmarks.",
    "metrics": "RMSLE: 0.114 \u00b7 R\u00b2 Score: 0.928 \u00b7 MAE: $13,400",
    "trainingLoss": "0.076 (MSE Loss)",
    "trainingAccuracy": "92.8% (R\u00b2 Variance)",
    "problem": "Property valuation is subjective and influenced by dozens of nonlinear structural and neighborhood attributes.",
    "breakingGround": "Comprehensive feature engineering combining Box-Cox skewness corrections, neighborhood target encodings, and Gradient Boosting regressors.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 1.25,
        "accuracy": 0.4,
        "valAccuracy": 0.38
      },
      {
        "epoch": 5,
        "loss": 0.38,
        "accuracy": 0.82,
        "valAccuracy": 0.8
      },
      {
        "epoch": 10,
        "loss": 0.076,
        "accuracy": 0.928,
        "valAccuracy": 0.916
      }
    ],
    "sampleDistribution": [
      {
        "label": "Single Family Residential",
        "percentage": 78.0,
        "count": "1,140 homes"
      },
      {
        "label": "Townhouse / Condos",
        "percentage": 22.0,
        "count": "320 homes"
      }
    ],
    "performanceIndicators": {
      "latency": "8ms",
      "throughput": "1,200 predictions/s",
      "memoryFootprint": "250 MB RAM",
      "hardwareTarget": "Standard CPU"
    },
    "techStack": [
      "Python",
      "XGBoost",
      "LightGBM",
      "Scikit-Learn"
    ],
    "architectureFlow": [
      "Property Feature Ingestion",
      "Box-Cox Log Transformation",
      "Neighborhood Target Encoding",
      "XGBoost Regressor",
      "Ensemble Averaging"
    ],
    "pipeline": [
      {
        "title": "01. Advanced Feature Engineering & Target Transformation",
        "desc": "Impute missing values, apply log1p to skewed sale prices, and engineer total square footage.",
        "code": "import pandas as pd\nimport numpy as np\nfrom sklearn.model_selection import train_test_split\n\ndf = pd.read_csv('housing.csv')\n\n# Log-transform target variable to achieve normality\ny = np.log1p(df['SalePrice'])\nX = df.drop(columns=['SalePrice', 'Id'])\n\n# Impute missing values\nnum_cols = X.select_dtypes(include=[np.number]).columns\ncat_cols = X.select_dtypes(include=['object']).columns\n\nX[num_cols] = X[num_cols].fillna(X[num_cols].median())\nX[cat_cols] = X[cat_cols].fillna('None')\n\n# Composite square footage feature\nX['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']\nX = pd.get_dummies(X, drop_first=True)\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
        "language": "python"
      },
      {
        "title": "02. XGBoost Regressor with Early Stopping",
        "desc": "Train gradient boosted decision trees with subsampling and regularization.",
        "code": "import xgboost as xgb\n\nmodel_xgb = xgb.XGBRegressor(\n    n_estimators=1000,\n    learning_rate=0.03,\n    max_depth=4,\n    subsample=0.8,\n    colsample_bytree=0.8,\n    reg_alpha=0.1,\n    reg_lambda=1.0,\n    random_state=42,\n    n_jobs=-1\n)\n\nmodel_xgb.fit(\n    X_train, y_train,\n    eval_set=[(X_test, y_test)],\n    verbose=False\n)",
        "language": "python"
      },
      {
        "title": "03. Ensembling & Root Mean Squared Log Error (RMSLE)",
        "desc": "Calculate holdout RMSLE, inverse-transform predictions, and serialize model.",
        "code": "from sklearn.metrics import mean_squared_error\nimport joblib\n\npreds_log = model_xgb.predict(X_test)\nrmsle = np.sqrt(mean_squared_error(y_test, preds_log))\nprint(f\"Holdout RMSLE: {rmsle:.4f}\")\n\n# Convert back to real USD pricing\npreds_usd = np.expm1(preds_log)\nactuals_usd = np.expm1(y_test)\nmae_usd = np.mean(np.abs(actuals_usd - preds_usd))\nprint(f\"Mean Absolute Error in USD: ${mae_usd:,.2f}\")\n\njoblib.dump(model_xgb, 'xgboost_house_prices.joblib')",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Property valuation is complex, non-linear, and prone to regional price distortions.",
      "breakingGround": "DataCamp advanced feature engineering and gradient boosting workflow.",
      "scope": "Real estate price prediction using tabular regression.",
      "functionality": "1. Missing value handling and skewness correction.\\n2. XGBoost regression modeling.",
      "abstract": "DataCamp benchmark for regression modeling on Ames Housing dataset.",
      "methodology": "Gradient boosted decision trees with cross-validated hyperparameter tuning.",
      "architecture": "Python Scikit-Learn & XGBoost pipeline.",
      "benchmarks": "Achieved 0.114 RMSLE and 0.928 R\u00b2 score on test data.",
      "productionConsiderations": "Regular model retraining on quarterly real estate transactions."
    }
  },
  {
    "id": "dc-stock-lstm",
    "title": "Stock Price Movement Prediction using LSTM Recurrent Neural Networks",
    "domain": "Quantitative Finance & Time Series Forecasting",
    "difficulty": "Advanced",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Model sequential stock market volatility and price direction using Long Short-Term Memory (LSTM) networks.",
    "objective": "Predict directional price movement with >63% accuracy using historical OHLCV time series.",
    "metrics": "Directional Accuracy: 63.8% \u00b7 Sharpe Ratio: 2.15 \u00b7 MAE: 0.84%",
    "trainingLoss": "0.031 (MSE Loss)",
    "trainingAccuracy": "63.8% Directional",
    "problem": "Time series forecasting of financial assets is notoriously difficult due to non-stationarity, noise, and sequential dependency.",
    "breakingGround": "Applies LSTM recurrent cells to capture long-term temporal dependencies in historical stock ticker data.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 0.25,
        "accuracy": 0.51,
        "valAccuracy": 0.5
      },
      {
        "epoch": 5,
        "loss": 0.09,
        "accuracy": 0.58,
        "valAccuracy": 0.57
      },
      {
        "epoch": 10,
        "loss": 0.031,
        "accuracy": 0.638,
        "valAccuracy": 0.625
      }
    ],
    "sampleDistribution": [
      {
        "label": "Bullish Up Ticks",
        "percentage": 52.3,
        "count": "1,250 days"
      },
      {
        "label": "Bearish Down Ticks",
        "percentage": 47.7,
        "count": "1,140 days"
      }
    ],
    "performanceIndicators": {
      "latency": "12ms",
      "throughput": "800 tickers/s",
      "memoryFootprint": "1.2 GB VRAM",
      "hardwareTarget": "NVIDIA GPU"
    },
    "techStack": [
      "Python",
      "PyTorch",
      "Pandas",
      "YFinance"
    ],
    "architectureFlow": [
      "OHLCV Time Series Ingestion",
      "MinMax Normalization",
      "LSTM Recurrent Layers",
      "Linear Regression Head"
    ],
    "pipeline": [
      {
        "title": "01. Sliding Window Sequence Generation from OHLCV",
        "desc": "Scale closing price sequences into 60-day historical window arrays.",
        "code": "import numpy as np\nimport pandas as pd\nimport torch\nfrom sklearn.preprocessing import MinMaxScaler\n\n# Load OHLCV price series\ndf = pd.read_csv('SP500_historical.csv')\ndata = df['Close'].values.reshape(-1, 1)\n\nscaler = MinMaxScaler(feature_range=(0, 1))\nscaled_data = scaler.fit_transform(data)\n\n# Create 60-day sliding window sequences\ndef create_sequences(data, seq_length=60):\n    xs, ys = [], []\n    for i in range(len(data) - seq_length):\n        xs.append(data[i:i+seq_length])\n        ys.append(data[i+seq_length])\n    return np.array(xs), np.array(ys)\n\nX, y = create_sequences(scaled_data)\ntrain_size = int(len(X) * 0.8)\n\nX_train = torch.tensor(X[:train_size], dtype=torch.float32)\ny_train = torch.tensor(y[:train_size], dtype=torch.float32)\nX_test = torch.tensor(X[train_size:], dtype=torch.float32)\ny_test = torch.tensor(y[train_size:], dtype=torch.float32)",
        "language": "python"
      },
      {
        "title": "02. PyTorch Multi-Layer LSTM Recurrent Architecture",
        "desc": "Construct a two-layer LSTM with recurrent dropout and dense linear head.",
        "code": "import torch.nn as nn\n\nclass StockLSTM(nn.Module):\n    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):\n        super().__init__()\n        self.lstm = nn.LSTM(\n            input_dim, hidden_dim, num_layers,\n            batch_first=True, dropout=0.2\n        )\n        self.fc = nn.Linear(hidden_dim, 1)\n\n    def forward(self, x):\n        out, _ = self.lstm(x)\n        return self.fc(out[:, -1, :])\n\nmodel = StockLSTM()\ncriterion = nn.MSELoss()\noptimizer = torch.optim.Adam(model.parameters(), lr=0.001)",
        "language": "python"
      },
      {
        "title": "03. Out-of-Sample Backtesting & Directional Accuracy",
        "desc": "Forecast forward prices, inverse-scale coordinates, and calculate directional trading win rate.",
        "code": "model.eval()\nwith torch.no_grad():\n    test_preds = model(X_test).numpy()\n\n# Invert MinMax scaling\npreds_actual = scaler.inverse_transform(test_preds)\ntargets_actual = scaler.inverse_transform(y_test.numpy())\n\n# Calculate directional movement accuracy (Up vs Down)\npred_direction = (preds_actual[1:] > preds_actual[:-1]).astype(int)\ntrue_direction = (targets_actual[1:] > targets_actual[:-1]).astype(int)\ndirectional_acc = (pred_direction == true_direction).mean()\n\nprint(f\"Directional Prediction Accuracy: {directional_acc * 100:.2f}%\")\ntorch.save(model.state_dict(), 'stock_lstm_model.pth')",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Time series forecasting of non-stationary financial data.",
      "breakingGround": "DataCamp advanced time series modeling project.",
      "scope": "Stock price direction forecasting using LSTMs.",
      "functionality": "1. Sequential data windowing.\\n2. LSTM recurrent cell training.",
      "abstract": "DataCamp advanced finance project.",
      "methodology": "Recurrent neural networks with sliding window sequences.",
      "architecture": "PyTorch LSTM sequence model.",
      "benchmarks": "Achieved 63.8% directional accuracy on out-of-sample data.",
      "productionConsiderations": "Requires continuous retraining with sliding window validation."
    }
  },
  {
    "id": "dc-pyspark-bigdata",
    "title": "Scalable Big Data Machine Learning with PySpark & MLlib",
    "domain": "Big Data & Distributed Systems",
    "difficulty": "Advanced",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Train distributed gradient-boosted trees and random forests across 100M flight delay records using Apache Spark.",
    "objective": "Achieve sub-minute distributed model training on over 100 million flight arrival records.",
    "metrics": "Distributed ROC-AUC: 0.842 \u00b7 Training Time: 48s / 100M rows",
    "trainingLoss": "0.145 (Logistic Loss)",
    "trainingAccuracy": "84.2%",
    "problem": "Single-node Python environments crash due to out-of-memory errors when processing hundreds of gigabytes of flight transaction data.",
    "breakingGround": "Leverages PySpark DataFrame pipelines, distributed Resilient Distributed Datasets (RDDs), and Spark MLlib parallel tree algorithms across multi-worker clusters.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 0.95,
        "accuracy": 0.58,
        "valAccuracy": 0.56
      },
      {
        "epoch": 3,
        "loss": 0.44,
        "accuracy": 0.74,
        "valAccuracy": 0.72
      },
      {
        "epoch": 6,
        "loss": 0.23,
        "accuracy": 0.81,
        "valAccuracy": 0.79
      },
      {
        "epoch": 10,
        "loss": 0.145,
        "accuracy": 0.842,
        "valAccuracy": 0.838
      }
    ],
    "sampleDistribution": [
      {
        "label": "On-Time Flight Arrivals",
        "percentage": 78.4,
        "count": "78.4M records"
      },
      {
        "label": "Delayed (>15 min) Arrivals",
        "percentage": 21.6,
        "count": "21.6M records"
      }
    ],
    "performanceIndicators": {
      "latency": "15ms (Distributed Batch)",
      "throughput": "250,000 rows/s",
      "memoryFootprint": "64 GB RAM Cluster",
      "hardwareTarget": "PySpark Standalone / EMR Cluster"
    },
    "techStack": [
      "Python",
      "PySpark",
      "Apache Spark",
      "HDFS",
      "Parquet"
    ],
    "architectureFlow": [
      "Distributed Parquet Data Ingestion",
      "StringIndexer & VectorAssembler",
      "CrossValidator Pipeline",
      "Spark MLlib GBTClassifier",
      "Cluster Parquet Checkpointing"
    ],
    "pipeline": [
      {
        "title": "01. Distributed Spark Session & Parquet Ingestion",
        "desc": "Initialize cluster SparkSession and read partitioned multi-gigabyte Parquet tables.",
        "code": "from pyspark.sql import SparkSession\nfrom pyspark.sql.functions import col, when\n\nspark = SparkSession.builder \\\n    .appName(\"FlightDelayMLlib\") \\\n    .config(\"spark.executor.memory\", \"8g\") \\\n    .config(\"spark.driver.memory\", \"4g\") \\\n    .getOrCreate()\n\n# Load partitioned flight delay records\ndf = spark.read.parquet(\"hdfs:///data/flights/*.parquet\")\ndf = df.filter(col(\"Cancelled\") == 0)\n\n# Binary target: 1 if delayed > 15 minutes, else 0\ndf = df.withColumn(\"Delayed\", when(col(\"ArrDelay\") > 15, 1.0).otherwise(0.0))",
        "language": "python"
      },
      {
        "title": "02. Feature Transformation Pipeline (VectorAssembler)",
        "desc": "Index categorical airline carriers and assemble dense feature vectors.",
        "code": "from pyspark.ml.feature import StringIndexer, VectorAssembler\nfrom pyspark.ml import Pipeline\n\nindexer = StringIndexer(inputCol=\"Carrier\", outputCol=\"CarrierIndex\", handleInvalid=\"keep\")\nassembler = VectorAssembler(\n    inputCols=[\"CarrierIndex\", \"Distance\", \"AirTime\", \"DepDelay\", \"DayOfWeek\"],\n    outputCol=\"features\"\n)\n\ntrain_df, test_df = df.randomSplit([0.8, 0.2], seed=42)",
        "language": "python"
      },
      {
        "title": "03. Distributed GBTClassifier & Evaluator",
        "desc": "Train distributed gradient boosted trees across executors and evaluate cluster ROC-AUC.",
        "code": "from pyspark.ml.classification import GBTClassifier\nfrom pyspark.ml.evaluation import BinaryClassificationEvaluator\n\ngbt = GBTClassifier(featuresCol=\"features\", labelCol=\"Delayed\", maxIter=20, maxDepth=5)\npipeline = Pipeline(stages=[indexer, assembler, gbt])\n\n# Train distributed model across Spark cluster\nmodel = pipeline.fit(train_df)\n\n# Evaluate on distributed test DataFrame\npredictions = model.transform(test_df)\nevaluator = BinaryClassificationEvaluator(labelCol=\"Delayed\", metricName=\"areaUnderROC\")\nauc = evaluator.evaluate(predictions)\nprint(f\"Distributed Test ROC-AUC: {auc:.4f}\")\n\nmodel.save(\"hdfs:///models/flight_delay_gbt\")",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Scalability bottlenecks in machine learning workflows when dataset volumes exceed single-machine memory capacity.",
      "breakingGround": "Demonstrates distributed machine learning architecture using Apache Spark MLlib with linear scaling across cluster nodes.",
      "scope": "Big data enterprise predictive analytics and high-throughput batch scoring.",
      "functionality": "1. Partitioned Parquet file loading.\\n2. Distributed vector assembly.\\n3. Gradient boosted tree fitting across worker nodes.\\n4. Distributed metric evaluation.",
      "abstract": "DataCamp advanced big data ML reference project applying PySpark to US DOT aviation records.",
      "methodology": "Distributed data-parallel tree boosting with in-memory RDD caching.",
      "architecture": "Apache Spark cluster architecture with master-worker topologies.",
      "benchmarks": "Trained on 100M rows in 48 seconds on an 8-node cluster with 0.842 ROC-AUC.",
      "productionConsiderations": "Automated cluster autoscaling and fault-tolerant executor recovery."
    }
  },
  {
    "id": "dc-rag-langchain",
    "title": "Retrieval-Augmented Generation (RAG) QA Assistant with LangChain",
    "domain": "Generative AI & LLM Systems",
    "difficulty": "Advanced",
    "description": "Imported from DataCamp Machine Learning Projects for All Levels. Construct an enterprise RAG assistant that semantically indexes technical documentation using dense vector embeddings and LangChain.",
    "objective": "Provide factual, zero-hallucination question answering with Mean Reciprocal Rank (MRR@5) > 90%.",
    "metrics": "Retrieval MRR@5: 92.4% \u00b7 Answer Faithfulness: 95.1% \u00b7 Latency: 420ms",
    "trainingLoss": "0.052 (Contrastive Ranking Loss)",
    "trainingAccuracy": "94.6% (Top-3 Retrieval Accuracy)",
    "problem": "Standard foundation LLMs hallucinate inaccurate information and lack access to proprietary, private company knowledge bases.",
    "breakingGround": "Combines recursive chunking with dense vector embeddings (FAISS), cross-encoder neural reranking, and grounded prompt synthesis.",
    "epochHistory": [
      {
        "epoch": 1,
        "loss": 1.15,
        "accuracy": 0.62,
        "valAccuracy": 0.6
      },
      {
        "epoch": 3,
        "loss": 0.58,
        "accuracy": 0.8,
        "valAccuracy": 0.78
      },
      {
        "epoch": 6,
        "loss": 0.18,
        "accuracy": 0.91,
        "valAccuracy": 0.89
      },
      {
        "epoch": 10,
        "loss": 0.052,
        "accuracy": 0.946,
        "valAccuracy": 0.938
      }
    ],
    "sampleDistribution": [
      {
        "label": "Technical API Documentation",
        "percentage": 45.0,
        "count": "18,000 chunks"
      },
      {
        "label": "Customer Support Tickets",
        "percentage": 35.0,
        "count": "14,000 chunks"
      },
      {
        "label": "Engineering Architecture RFCs",
        "percentage": 20.0,
        "count": "8,000 chunks"
      }
    ],
    "performanceIndicators": {
      "latency": "420ms / Query",
      "throughput": "85 concurrent Q&A/s",
      "memoryFootprint": "3.8 GB RAM",
      "hardwareTarget": "Vector Database + API Endpoint"
    },
    "techStack": [
      "Python",
      "LangChain",
      "FAISS",
      "Sentence-Transformers",
      "HuggingFace"
    ],
    "architectureFlow": [
      "PDF & Markdown Document Loader",
      "RecursiveCharacterTextSplitter",
      "HuggingFace BGE Dense Embeddings",
      "FAISS Vector Index",
      "Context Injection Prompt",
      "Gemini LLM Generation"
    ],
    "pipeline": [
      {
        "title": "01. Document Chunking & Dense Embeddings",
        "desc": "Split enterprise documentation into semantic overlapping chunks with Sentence-Transformers.",
        "code": "from langchain_community.document_loaders import DirectoryLoader, TextLoader\nfrom langchain_text_splitters import RecursiveCharacterTextSplitter\nfrom langchain_community.embeddings import HuggingFaceEmbeddings\n\n# Load documentation corpus\nloader = DirectoryLoader('docs/', glob=\"**/*.md\", loader_cls=TextLoader)\ndocuments = loader.load()\n\n# Split into 500-token chunks with 50-token overlap\ntext_splitter = RecursiveCharacterTextSplitter(\n    chunk_size=500,\n    chunk_overlap=50,\n    separators=[\"\\n\\n\", \"\\n\", \" \", \"\"]\n)\nchunks = text_splitter.split_documents(documents)\nprint(f\"Total Chunks Generated: {len(chunks)}\")\n\n# Dense embeddings model\nembeddings = HuggingFaceEmbeddings(model_name=\"BAAI/bge-small-en-v1.5\")",
        "language": "python"
      },
      {
        "title": "02. FAISS Vector Indexing & Similarity Search",
        "desc": "Index vectors in FAISS for sub-millisecond nearest neighbor semantic search.",
        "code": "from langchain_community.vectorstores import FAISS\n\n# Ingest and build in-memory vector index\nvector_db = FAISS.from_documents(chunks, embeddings)\n\n# Persist vector index to local storage\nvector_db.save_local(\"faiss_index_store\")\n\n# Execute test similarity search\nquery = \"How to configure distributed cluster timeouts?\"\nmatching_docs = vector_db.similarity_search_with_score(query, k=3)\nfor doc, score in matching_docs:\n    print(f\"Score {score:.3f} | {doc.page_content[:100]}...\")",
        "language": "python"
      },
      {
        "title": "03. Grounded RetrievalQA Chain with Context Injection",
        "desc": "Build contextual question-answering chain enforcing citation constraints.",
        "code": "from langchain.prompts import PromptTemplate\nfrom langchain.chains import RetrievalQA\nfrom langchain_community.llms import HuggingFacePipeline\n\ntemplate = \"\"\"You are an expert technical assistant. Answer the question strictly using the provided context. If the answer is not contained in the context, say 'I do not know'.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:\"\"\"\n\nprompt = PromptTemplate(template=template, input_variables=['context', 'question'])\n\nretriever = vector_db.as_retriever(search_kwargs={\"k\": 4})\n# In production, pass LLM of choice (e.g., Gemini or local Llama-3)\nprint(\"RAG Retrieval pipeline active and ready for incoming queries.\")",
        "language": "python"
      }
    ],
    "whitePaper": {
      "problem": "Information retrieval bottlenecks and hallucination vulnerabilities in enterprise generative AI.",
      "breakingGround": "DataCamp state-of-the-art Generative AI project implementing end-to-end dense RAG.",
      "scope": "Enterprise conversational search and grounded knowledge base assistant.",
      "functionality": "1. Document parsing and recursive chunking.\\n2. Dense vector indexing in FAISS.\\n3. Semantic k-nearest neighbor retrieval.\\n4. Grounded answer generation.",
      "abstract": "DataCamp advanced AI project combining vector search with LangChain RAG workflows.",
      "methodology": "Contrastive bi-encoder embeddings paired with approximate nearest neighbor indexing.",
      "architecture": "LangChain retrieval-augmented generative microservice.",
      "benchmarks": "Achieved 92.4% MRR@5 retrieval precision and 95.1% grounded faithfulness.",
      "productionConsiderations": "Continuous vector index synchronization and metadata security filtering."
    }
  },
  ...WORLD_SCALE_COMMERCIAL_PROJECTS
];

const ALL_REPOSITORY_PROJECTS: FlagshipProject[] = [
  ...FLAGSHIP_PROJECTS,
  ...UIDAI_DPI_PROJECTS,
  ...CYBERSECURITY_RESEARCH_PROJECTS,
  ...RESEARCH_GRADE_PROJECTS,
  ...EXTERNAL_URL_PROJECTS,
  ...DATACAMP_REPOSITORY_PROJECTS.map((p) => ({
    ...p,
    sourceName: p.sourceName || 'DataCamp',
    sourceUrl: p.sourceUrl || 'https://www.datacamp.com/blog/machine-learning-projects-for-all-levels'
  }))
];

const STARRED_REPO_STORAGE_KEY = 'ml_studio_starred_repo_ids_v1';

// Top 5 standout research-grade projects selected from ML Case Studies across UIDAI DPI, Cybersecurity & Multimodal Diffusion
const DEFAULT_STARRED_REPO_IDS: string[] = [
  'uidai-multimodal-biometric-pad',
  'uidai-ekyc-fraud-prevention',
  'cyber-cagn-gat-ids',
  'cyber-rust-pe-malware',
  'research-diffusioncom-mkgc'
];

export default function App() {
  const [activeTab, setActiveTab] = useState<'showcase' | 'repository' | 'whitepapers' | 'learning' | 'coderepo' | 'training' | 'archives' | 'builder' | 'copilot' | 'roadmap'>('showcase');
  const [archivesOpen, setArchivesOpen] = useState(false);

  const [starredRepoIds, setStarredRepoIds] = useState<string[]>(() => {
    try {
      const saved = window.localStorage.getItem(STARRED_REPO_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          const validIds = parsed.filter(
            (id): id is string => typeof id === 'string' && ALL_REPOSITORY_PROJECTS.some(p => p.id === id)
          );
          return validIds;
        }
      }
    } catch {
      // Ignore storage read errors
    }
    return DEFAULT_STARRED_REPO_IDS;
  });
  const [starredOnlyFilter, setStarredOnlyFilter] = useState(false);

  useEffect(() => {
    try {
      window.localStorage.setItem(STARRED_REPO_STORAGE_KEY, JSON.stringify(starredRepoIds));
    } catch {
      // Ignore storage write errors
    }
  }, [starredRepoIds]);

  const starredRepoProjects = ALL_REPOSITORY_PROJECTS.filter(p => starredRepoIds.includes(p.id));
  const researchProjects: FlagshipProject[] = starredRepoProjects;
  const nonStarredRepoProjects = ALL_REPOSITORY_PROJECTS.filter(p => !starredRepoIds.includes(p.id));

  useEffect(() => {
    if (activeTab === 'archives' || activeTab === 'builder' || activeTab === 'copilot' || activeTab === 'roadmap') {
      setArchivesOpen(true);
    }
  }, [activeTab]);
  const [selectedProject, setSelectedProject] = useState<FlagshipProject | null>(() => starredRepoProjects[0] || null);
  const [repoFilter, setRepoFilter] = useState<'All' | 'Beginner' | 'Intermediate' | 'Advanced' | 'Research Grade'>('All');
  const [repoSearch, setRepoSearch] = useState('');
  const [selectedRepoProject, setSelectedRepoProject] = useState<FlagshipProject | null>(ALL_REPOSITORY_PROJECTS[0]);
  const [codeRepoTargetProjectId, setCodeRepoTargetProjectId] = useState<string>((starredRepoProjects[0] || ALL_REPOSITORY_PROJECTS[0])?.id || '');
  const [copiedCodeIndex, setCopiedCodeIndex] = useState<number | null>(null);
  const repoSpecsRef = useRef<HTMLDivElement | null>(null);
  const storyboardSpecsRef = useRef<HTMLDivElement | null>(null);

  const handleViewRepoSpecs = (proj: FlagshipProject) => {
    setSelectedRepoProject(proj);
    requestAnimationFrame(() => {
      setTimeout(() => {
        repoSpecsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 30);
    });
  };

  const handleViewStoryboardSpecs = (proj: FlagshipProject) => {
    setSelectedProject(proj);
    requestAnimationFrame(() => {
      setTimeout(() => {
        storyboardSpecsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 30);
    });
  };

  const activeStoryboardProject =
    selectedProject && starredRepoIds.includes(selectedProject.id)
      ? selectedProject
      : (researchProjects[0] || null);

  const toggleStarRepoProject = (projectId: string) => {
    const isCurrentlyStarred = starredRepoIds.includes(projectId);
    const nextStarredIds = isCurrentlyStarred
      ? starredRepoIds.filter(id => id !== projectId)
      : [...starredRepoIds, projectId];

    setStarredRepoIds(nextStarredIds);

    const nextResearchList = ALL_REPOSITORY_PROJECTS.filter(p => nextStarredIds.includes(p.id));
    if (isCurrentlyStarred && selectedProject?.id === projectId) {
      setSelectedProject(nextResearchList[0] || null);
    } else if (!isCurrentlyStarred && !selectedProject) {
      const newlyStarred = ALL_REPOSITORY_PROJECTS.find(p => p.id === projectId) || nextResearchList[0] || null;
      setSelectedProject(newlyStarred);
    }
  };
  
  // White Paper Modal State
  const [whitePaperModalProject, setWhitePaperModalProject] = useState<FlagshipProject | null>(null);

  // Project Builder State
  const [builderForm, setBuilderForm] = useState({
    projectName: 'Autonomous Vision SLAM & 3D Occupancy Network',
    domain: 'Computer Vision & Robotics',
    objective: 'Reconstruct dense 3D voxel occupancy from multi-camera surround video for autonomous robot navigation.',
    dataType: 'NuScenes Multi-Camera Video & LiDAR Point Clouds',
    targetMetric: 'mAP > 0.68, Ray IoU > 0.55',
    deploymentTarget: 'NVIDIA Drive Orin Edge ECU'
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedBlueprint, setGeneratedBlueprint] = useState<ProjectBlueprint | null>(null);
  const [builderError, setBuilderError] = useState<string | null>(null);
  const [showBuilderWhitePaper, setShowBuilderWhitePaper] = useState(false);

  // ML Training Simulator State (1.50B Cohort + 20-Year Horizon + Architecture, ML & Encryption)
  const allProjectsCatalog: FlagshipProject[] = ALL_REPOSITORY_PROJECTS;
  const [simConfig, setSimConfig] = useState<SimConfigState>({
    projectId: FLAGSHIP_PROJECTS[0].id,
    dataType: inferProjectDataType(FLAGSHIP_PROJECTS[0]),
    syntheticVariant: 'balanced',
    sampleCount: 1500000000,
    annualDeltaPct: 2.4,
    horizonYears: 20,
    seed: 42,
    learningRate: inferBestLearningRate(FLAGSHIP_PROJECTS[0]).lrValue,
    batchSize: '64',
    epochs: inferOptimalEpochs(FLAGSHIP_PROJECTS[0]),
    optimizer: 'AdamW',
    architecture: FLAGSHIP_PROJECTS[0].techStack.slice(0, 2).join(' + '),
    programmingArch: inferBestProgrammingArch(FLAGSHIP_PROJECTS[0]),
    mlTechnique: inferBestMlTechnique(FLAGSHIP_PROJECTS[0]),
    encryptionProfile: 'pqc_hybrid_vault'
  });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingEpoch, setTrainingEpoch] = useState(0);
  const [trainingLogs, setTrainingLogs] = useState<TrainingLogEntry[]>([]);
  const trainingIntervalRef = useRef<number | null>(null);

  // RAG Knowledge Base & AI Copilot Chat State
  const [ragStatus, setRagStatus] = useState<RagStatusResponse | null>(null);
  const [useRagGrounding, setUseRagGrounding] = useState(true);
  const [copilotSearchMode, setCopilotSearchMode] = useState<'hybrid' | 'dense' | 'bm25'>('hybrid');
  const [chatMessages, setChatMessages] = useState<{ sender: 'user' | 'ai'; text: string; citations?: RagCitation[] }[]>([
    { sender: 'ai', text: 'Hello! I am your Hybrid RAG-Enabled AI/ML Principal Copilot. Ingest PDFs or raw text in the Repository tab to index chunks into FAISS + BM25, then ask questions here grounded in your documents.' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const [copilotMonitor, setCopilotMonitor] = useState({
    totalRequests: 0,
    successCount: 0,
    lastLatencyMs: 0,
    avgLatencyMs: 0,
    totalInputTokens: 0,
    totalOutputTokens: 0,
    lastPromptTokens: 0,
    lastCompletionTokens: 0
  });

  const fetchRagStatus = async () => {
    try {
      const res = await fetch('/api/rag/status');
      if (res.ok) {
        const data = (await res.json()) as RagStatusResponse;
        setRagStatus(data);
      }
    } catch {
      // Ignore transient startup polling errors
    }
  };

  useEffect(() => {
    fetchRagStatus();
  }, []);

  // Roadmap Checklist State
  const [checklist, setChecklist] = useState([
    { id: 1, phase: '01. Problem Scope & Metric Definition', done: true, desc: 'Define business objective, baseline metrics, and success criteria.' },
    { id: 2, phase: '02. Data Collection & Exploratory Analysis (EDA)', done: true, desc: 'Gather data sources, check for missing values, imbalances, and distribution shifts.' },
    { id: 3, phase: '03. Feature Engineering & Preprocessing', done: true, desc: 'Normalization, tokenization, encoding categorical variables, and train/val/test splits.' },
    { id: 4, phase: '04. Model Architecture & Baseline', done: false, desc: 'Select baseline model (e.g. XGBoost or ResNet50) and establish benchmark performance.' },
    { id: 5, phase: '05. Training & Hyperparameter Optimization', done: false, desc: 'Run training loop with learning rate scheduler, early stopping, and weight decay.' },
    { id: 6, phase: '06. Evaluation & Error Analysis', done: false, desc: 'Analyze confusion matrix, precision-recall curves, and edge-case failure modes.' },
    { id: 7, phase: '07. Model Serialization & ONNX Export', done: false, desc: 'Export model weights to ONNX or torchscript for optimized low-latency inference.' },
    { id: 8, phase: '08. Production Deployment & MLOps Monitoring', done: false, desc: 'Deploy via FastAPI / Docker / Cloud Run, and set up Prometheus metrics & data drift alarms.' }
  ]);

  const [copiedAllId, setCopiedAllId] = useState<string | null>(null);
  const [copiedCitationId, setCopiedCitationId] = useState<string | null>(null);

  const handleCopyCode = (code: string, index: number) => {
    navigator.clipboard.writeText(code);
    setCopiedCodeIndex(index);
    setTimeout(() => setCopiedCodeIndex(null), 2000);
  };

  const handleCopyAllPipeline = (project: FlagshipProject) => {
    const fullCode = buildColabPercentScript(project);
    navigator.clipboard.writeText(fullCode);
    setCopiedAllId(project.id);
    setTimeout(() => setCopiedAllId(null), 2500);
  };

  const handleCopyCitation = (project: FlagshipProject) => {
    const bibtex = `@article{datacamp_ml_${project.id.replace(/-/g, '_')}_2024,
  title = {${project.title}},
  author = {Applied Machine Learning Research Group},
  journal = {DataCamp Machine Learning Projects Catalog & Google Scholar Benchmark},
  year = {2024},
  keywords = {${project.techStack.join(', ')}},
  note = {Loss: ${project.trainingLoss}, Accuracy: ${project.trainingAccuracy}, Metrics: ${project.metrics}}
}`;
    navigator.clipboard.writeText(bibtex);
    setCopiedCitationId(project.id);
    setTimeout(() => setCopiedCitationId(null), 2500);
  };

  const handleGenerateBlueprint = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsGenerating(true);
    setBuilderError(null);
    try {
      const res = await fetch('/api/generate-project', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(builderForm)
      });
      if (!res.ok) throw new Error('Failed to generate project blueprint');
      const data = await res.json();
      if (!data.whitePaper) {
        data.whitePaper = {
          problem: `Current limitations in ${builderForm.domain} include high latency, lack of deterministic performance, and inadequate handling of complex input distributions.`,
          breakingGround: `Introduces a novel hybrid neural architecture optimized for ${builderForm.objective} with real-time throughput.`,
          scope: `Full technical scope for ${builderForm.projectName}, addressing ${builderForm.objective} within ${builderForm.domain}.`,
          functionality: `1. Ingestion of ${builderForm.dataType} using Rust asynchronous runtimes.\n2. Deep neural network processing.\n3. Validation against ${builderForm.targetMetric}.\n4. Low-latency deployment on ${builderForm.deploymentTarget}.`,
          abstract: `Technical white paper for ${builderForm.projectName} incorporating Rust for high-performance data preprocessing and inference.`,
          methodology: `Rigorous data pipelines processing ${builderForm.dataType} with Rust memory-safe concurrency and PyTorch neural networks.`,
          architecture: `End-to-end containerized microservice architecture featuring Rust tokenizers/parsers and Python GPU training workers.`,
          benchmarks: `Exceeded target metric ${builderForm.targetMetric} with sub-millisecond preprocessing latency.`,
          productionConsiderations: `High availability, zero-copy memory management, and robust drift detection.`
        };
      }
      setGeneratedBlueprint(data);
    } catch (err: any) {
      setBuilderError(err.message || 'Error generating blueprint');
    } finally {
      setIsGenerating(false);
    }
  };

  const runTrainingSimulation = (overrideConfig?: SimConfigState) => {
    const cfg = overrideConfig || simConfig;
    if (trainingIntervalRef.current !== null) {
      window.clearInterval(trainingIntervalRef.current);
    }

    setIsTraining(true);
    setTrainingEpoch(0);
    setTrainingLogs([]);

    const targetProj =
      allProjectsCatalog.find((p) => p.id === cfg.projectId) || FLAGSHIP_PROJECTS[0];
    const nativeDataType = inferProjectDataType(targetProj);
    const isCrossModality = cfg.dataType !== nativeDataType;

    const varMeta = SYNTHETIC_VARIANT_META[cfg.syntheticVariant];
    const typeMeta = SYNTHETIC_DATA_TYPE_META[cfg.dataType];
    const progMeta = PROGRAMMING_ARCHITECTURE_META[cfg.programmingArch];
    const mlMeta = ML_TECHNIQUE_META[cfg.mlTechnique];

    const maxEpochs = Math.max(6, Number(cfg.epochs) || 18);
    const lrNum = Math.max(1e-5, Math.min(0.05, parseFloat(cfg.learningRate) || 0.0001));
    const matchedLrPreset = LEARNING_RATE_PRESETS.find(
      (p) => Math.abs(p.numericLr - lrNum) < p.numericLr * 0.2
    );
    const lrSigmaMult = matchedLrPreset ? matchedLrPreset.sigmaMult : lrNum > 0.005 ? 1.35 : 1.0;

    // Unified 1.50B-scale + 20-year horizon optimized convergence targets
    const scaleTargets = computeScale1B5Metrics({
      project: targetProj,
      dataType: cfg.dataType,
      variant: cfg.syntheticVariant,
      progArch: cfg.programmingArch,
      mlTech: cfg.mlTechnique,
      encProfile: cfg.encryptionProfile,
      learningRate: cfg.learningRate,
      batchSize: cfg.batchSize,
      epochs: maxEpochs,
      optimizer: cfg.optimizer,
      sampleCount: cfg.sampleCount,
      annualDeltaPct: cfg.annualDeltaPct,
      horizonYears: cfg.horizonYears
    });

    const startLoss = Math.max(
      0.85,
      (targetProj.epochHistory[0]?.loss || 1.45) *
        varMeta.lossPenalty *
        typeMeta.lossMult *
        (isCrossModality ? 1.16 : 1.0)
    );
    const rawFirstAcc = targetProj.epochHistory[0]?.accuracy || 0.68;
    const startAcc = Math.max(
      0.48,
      Math.min(0.86, (rawFirstAcc > 1 ? rawFirstAcc / 100 : rawFirstAcc) - (isCrossModality ? 0.05 : 0))
    );

    const latStats = computeLatencyP95P99(
      cfg.programmingArch,
      cfg.dataType,
      cfg.mlTechnique,
      cfg.encryptionProfile,
      cfg.sampleCount
    );

    let currentEpoch = 0;

    const interval = window.setInterval(() => {
      currentEpoch += 1;
      const t = currentEpoch / maxEpochs; // 0..1 progress along Cosine WSD schedule
      const cosineDecay = 0.5 * (1 + Math.cos(Math.PI * t)); // 1 -> 0 smooth cosine
      const fastDrop = Math.pow(1 - t, 1.85);
      const blend = 0.55 * fastDrop + 0.45 * cosineDecay;
      const noiseScale = (1 - t * 0.85) * 0.012 * varMeta.noiseMultiplier * lrSigmaMult;

      const loss = Math.max(
        scaleTargets.optimizedLoss,
        scaleTargets.optimizedLoss + (startLoss - scaleTargets.optimizedLoss) * blend + Math.random() * noiseScale
      );
      const accuracy = Math.min(
        scaleTargets.optimizedTrainAcc,
        startAcc + (scaleTargets.optimizedTrainAcc - startAcc) * (1 - blend) - Math.random() * 0.002
      );
      const valLoss = Math.max(loss * 1.03, loss * 1.06 + Math.random() * noiseScale * 0.7);
      const gap = Math.max(0.002, (scaleTargets.generalizationGapPct / 100) * (0.75 + 0.5 * (1 - t)));
      const valAcc = Math.min(
        scaleTargets.optimizedValAcc,
        Math.max(0.42, accuracy - gap + (Math.random() - 0.5) * 0.0018 * (1 - t * 0.7))
      );

      const sigma = Number(
        (varMeta.sigmaBase * lrSigmaMult * (0.45 + 0.55 * (1 - t))).toFixed(2)
      );
      const throughput = Math.round(
        scaleTargets.optimizedThroughput * (0.96 + Math.random() * 0.07)
      );
      const jitterRand = 0.96 + Math.random() * 0.06;
      const latencyMs = Number((latStats.p95Ms * jitterRand).toFixed(2));
      const p99LatencyMs = Number((latStats.p99Ms * (0.97 + Math.random() * 0.08)).toFixed(2));

      setTrainingEpoch(currentEpoch);
      setTrainingLogs((prev) => [
        ...prev,
        {
          epoch: currentEpoch,
          loss: Number(loss.toFixed(4)),
          accuracy: Number(accuracy.toFixed(4)),
          valLoss: Number(valLoss.toFixed(4)),
          valAcc: Number(valAcc.toFixed(4)),
          sigma,
          throughput,
          latencyMs,
          p99LatencyMs,
          dataTypeLabel: typeMeta.shortLabel,
          progArchLabel: progMeta.shortName,
          mlTechLabel: mlMeta.shortName
        }
      ]);

      if (currentEpoch >= maxEpochs) {
        window.clearInterval(interval);
        trainingIntervalRef.current = null;
        setIsTraining(false);
      }
    }, 240);

    trainingIntervalRef.current = interval;
  };

  const handleLaunchTrainingFromProject = (
    project: FlagshipProject,
    dataType: SyntheticDataType,
    variant: SyntheticVariant
  ) => {
    const autoMlTech = inferBestMlTechnique(project);
    const nextConfig: SimConfigState = {
      ...simConfig,
      projectId: project.id,
      architecture: project.techStack.slice(0, 2).join(' + '),
      dataType,
      syntheticVariant: variant,
      sampleCount: variant === 'scaled_100k' ? 2410000000 : 1500000000,
      batchSize: '64',
      epochs: inferOptimalEpochs(project, dataType, autoMlTech, simConfig.annualDeltaPct),
      programmingArch: inferBestProgrammingArch(project),
      mlTechnique: autoMlTech,
      learningRate: inferBestLearningRate(project).lrValue
    };
    setSimConfig(nextConfig);
    setActiveTab('training');
    runTrainingSimulation(nextConfig);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || isChatLoading) return;

    const userText = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { sender: 'user', text: userText }]);
    setIsChatLoading(true);
    const startTime = performance.now();

    try {
      const res = await fetch('/api/ai-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userText,
          useRag: useRagGrounding,
          searchMode: copilotSearchMode,
          alpha: 0.55
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Chat failed');

      const elapsed = Math.round(performance.now() - startTime);
      const promptTok = Math.round(userText.split(/\s+/).length * 1.3);
      const replyTok = Math.round((data.reply || '').split(/\s+/).length * 1.3);

      setCopilotMonitor(prev => {
        const totalReqs = prev.totalRequests + 1;
        const successes = prev.successCount + 1;
        const newAvg = Math.round(((prev.avgLatencyMs * (totalReqs - 1)) + elapsed) / totalReqs);
        return {
          totalRequests: totalReqs,
          successCount: successes,
          lastLatencyMs: elapsed,
          avgLatencyMs: newAvg,
          totalInputTokens: prev.totalInputTokens + promptTok,
          totalOutputTokens: prev.totalOutputTokens + replyTok,
          lastPromptTokens: promptTok,
          lastCompletionTokens: replyTok
        };
      });

      setChatMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: data.reply,
          citations: Array.isArray(data.citations) ? data.citations : []
        }
      ]);
      fetchRagStatus();
    } catch (err: any) {
      const elapsed = Math.round(performance.now() - startTime);
      setCopilotMonitor(prev => ({
        ...prev,
        totalRequests: prev.totalRequests + 1,
        lastLatencyMs: elapsed
      }));
      setChatMessages(prev => [
        ...prev,
        {
          sender: 'ai',
          text: `Error communicating with the RAG Copilot: ${err.message || 'Check Settings > Secrets for GEMINI_API_KEY.'}`
        }
      ]);
    } finally {
      setIsChatLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex font-sans selection:bg-indigo-500 selection:text-white">
      
      {/* Left Hand Side Vertical Sidebar Menu */}
      <aside className="w-64 bg-white border-r border-slate-200 flex flex-col shrink-0 fixed inset-y-0 left-0 z-40 shadow-sm">
        {/* Brand Header */}
        <div className="p-6 border-b border-slate-200 flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-600 to-violet-700 flex items-center justify-center shadow-md shadow-indigo-600/20 shrink-0">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <div>
            <span className="text-sm font-bold tracking-tight text-slate-900 block">ScholarML Studio</span>
            <span className="text-xs text-indigo-600 font-medium">Enterprise AI &amp; Research</span>
          </div>
        </div>

        {/* Vertical Navigation Links */}
        <nav className="flex-1 px-5 py-6 space-y-2 overflow-y-auto">
          <button 
            onClick={() => setActiveTab('showcase')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${activeTab === 'showcase' ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/20 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
          >
            <LayoutDashboard className="w-4 h-4 shrink-0" />
            <span>Storyboard</span>
          </button>

          <button 
            onClick={() => setActiveTab('coderepo')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${activeTab === 'coderepo' ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/20 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
          >
            <FolderGit2 className="w-4 h-4 shrink-0" />
            <span>Code Repo</span>
          </button>

          <button 
            onClick={() => setActiveTab('training')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${activeTab === 'training' ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/20 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
          >
            <LineChart className="w-4 h-4 shrink-0" />
            <span>ML Simulator</span>
          </button>

          <button 
            onClick={() => setActiveTab('repository')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${activeTab === 'repository' ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/20 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
          >
            <Database className="w-4 h-4 shrink-0" />
            <span>ML Case Studies</span>
          </button>

          <button 
            onClick={() => setActiveTab('whitepapers')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${activeTab === 'whitepapers' ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/20 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
          >
            <Newspaper className="w-4 h-4 shrink-0" />
            <span>White Papers</span>
          </button>

          <button 
            onClick={() => setActiveTab('learning')}
            className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${activeTab === 'learning' ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/20 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
          >
            <GraduationCap className="w-4 h-4 shrink-0" />
            <span>Learning</span>
          </button>

          {/* Archives Section (Project Builder, AI ML Copilot, MLOps Roadmap) */}
          <div className="pt-4 mt-4 border-t border-slate-200 space-y-2 pb-4">
            <button
              onClick={() => {
                if (activeTab === 'archives') {
                  setArchivesOpen(prev => !prev);
                } else {
                  setActiveTab('archives');
                  setArchivesOpen(true);
                }
              }}
              className={`w-full flex items-center justify-between px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                activeTab === 'archives'
                  ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/20 font-semibold'
                  : (activeTab === 'builder' || activeTab === 'copilot' || activeTab === 'roadmap')
                    ? 'bg-slate-100 text-slate-900 font-semibold'
                    : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <span className="flex items-center gap-3">
                <Archive className="w-4 h-4 shrink-0" />
                <span>Archives</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-md ${
                  activeTab === 'archives' ? 'bg-indigo-500 text-white' : 'bg-slate-200 text-slate-700'
                }`}>
                  3
                </span>
                <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-200 ${archivesOpen ? 'rotate-180' : ''}`} />
              </span>
            </button>

            {archivesOpen && (
              <div className="pl-3 pr-1 space-y-1.5 pt-1.5 pb-1">
                <button 
                  onClick={() => setActiveTab('builder')}
                  className={`w-full flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all ${activeTab === 'builder' ? 'bg-indigo-600 text-white shadow-xs shadow-indigo-600/20 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
                >
                  <Wand2 className="w-3.5 h-3.5 shrink-0" />
                  <span>Project Builder Wizard</span>
                </button>

                <button 
                  onClick={() => setActiveTab('copilot')}
                  className={`w-full flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all ${activeTab === 'copilot' ? 'bg-indigo-600 text-white shadow-xs shadow-indigo-600/20 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
                >
                  <MessageSquareCode className="w-3.5 h-3.5 shrink-0" />
                  <span>AI ML Copilot</span>
                </button>

                <button 
                  onClick={() => setActiveTab('roadmap')}
                  className={`w-full flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all ${activeTab === 'roadmap' ? 'bg-indigo-600 text-white shadow-xs shadow-indigo-600/20 font-semibold' : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'}`}
                >
                  <Workflow className="w-3.5 h-3.5 shrink-0" />
                  <span>MLOps Roadmap</span>
                </button>
              </div>
            )}
          </div>
        </nav>

      </aside>

      {/* Main Content Area (Offset by Sidebar Width pl-64) */}
      <div className="flex-1 pl-64 flex flex-col min-h-screen">
        
        {/* Top Header Bar */}
        <header className="sticky top-0 z-30 bg-white/90 backdrop-blur-md border-b border-slate-200 px-8 py-4 flex items-center justify-between shadow-xs">
          <div className="flex items-center gap-2">
            <span className="text-xs uppercase tracking-wider font-semibold text-slate-500">Workspace /</span>
            {(activeTab === 'builder' || activeTab === 'copilot' || activeTab === 'roadmap') && (
              <>
                <button
                  onClick={() => setActiveTab('archives')}
                  className="text-xs uppercase tracking-wider font-semibold text-slate-500 hover:text-slate-900 transition-colors"
                >
                  Archives /
                </button>
              </>
            )}
            <span className="text-xs font-bold text-slate-900 capitalize">
              {activeTab === 'showcase'
                ? 'Storyboard'
                : activeTab === 'repository'
                  ? 'ML Case Studies'
                  : activeTab === 'builder'
                    ? 'Project Builder Wizard'
                    : activeTab === 'copilot'
                      ? 'AI ML Copilot'
                      : activeTab === 'roadmap'
                        ? 'MLOps Roadmap'
                        : activeTab === 'coderepo'
                          ? 'Code Repo'
                          : activeTab === 'training'
                            ? 'ML Simulator'
                            : activeTab === 'whitepapers'
                              ? 'White Papers'
                              : activeTab === 'learning'
                                ? 'Learning'
                                : 'Archives'}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs text-slate-600 font-mono">Gemini 4 Hydrogen Online</span>
          </div>
        </header>

        {/* Main Body */}
        <main className="flex-1 max-w-[1440px] w-full mx-auto px-8 py-10">
          
          {/* HERO BANNER */}
          {activeTab === 'showcase' && (
            <div className="space-y-6 mb-10">
              <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-indigo-50 via-white to-slate-50 border border-slate-200 p-6 md:p-8 w-full shadow-sm">
                <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/5 rounded-full blur-3xl pointer-events-none" />
                <div className="relative z-10 flex flex-col gap-4">
                  <div className="space-y-2 max-w-4xl">
                    <h1 className="text-2xl md:text-3xl font-bold text-slate-900 tracking-tight">
                      AI/ML Research &amp; Cases Studies
                    </h1>
                    <p className="text-slate-600 text-base leading-relaxed">
                      Production-ready reference implementations featuring Training Convergence Charts, Sample Data Distributions, Architecture Topologies, and Rust-accelerated pipelines.
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2.5">
                    <button
                      onClick={() => setActiveTab('builder')}
                      className="px-3.5 py-2 bg-indigo-600 text-white font-semibold rounded-xl hover:bg-indigo-500 transition-colors flex items-center gap-1.5 text-xs shadow-sm shadow-indigo-600/20 cursor-pointer"
                    >
                      <span>Start Custom Project</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setActiveTab('training')}
                      className="px-3.5 py-2 bg-white text-slate-700 font-semibold rounded-xl hover:bg-slate-50 transition-colors flex items-center gap-1.5 text-xs border border-slate-300 shadow-xs cursor-pointer"
                    >
                      <Play className="w-3.5 h-3.5 text-indigo-600" />
                      <span>Training Sandbox</span>
                    </button>
                  </div>
                </div>
              </div>

            </div>
          )}

          {/* TAB: REPOSITORY (DATACAMP IMPORT) */}
          {activeTab === 'repository' && (
            <div className="space-y-8">
              {/* Header Banner */}
              <div className="bg-slate-100 hover:bg-slate-200/80 p-6 rounded-2xl border border-slate-300 shadow-xs transition-all duration-200">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div>
                    <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Machine Learning & AI Case Studies Repository</h2>
                    <p className="text-slate-700 text-sm mt-1 max-w-4xl leading-relaxed">
                      End-to-end ML &amp; research-grade architectures complete with convergence &amp; ±σ deviation graphs, model comparisons, and annotated executable code.
                    </p>
                  </div>
                </div>

                {/* Level Filter + Search Bar */}
                <div className="mt-6 pt-5 border-t border-slate-300 space-y-4">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    {/* Difficulty Filters + Starred Filter */}
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs font-semibold text-slate-700 mr-1 flex items-center gap-1.5">
                        <Filter className="w-3.5 h-3.5 text-blue-900" />
                        <span>Level:</span>
                      </span>
                      {(['All', 'Beginner', 'Intermediate', 'Advanced', 'Research Grade'] as const).map(lvl => (
                        <button
                          key={lvl}
                          onClick={() => {
                            setRepoFilter(lvl);
                            if (lvl === 'All') {
                              setStarredOnlyFilter(false);
                            }
                          }}
                          className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 ${
                            repoFilter === lvl && !(lvl === 'All' && starredOnlyFilter)
                              ? 'bg-blue-900 text-white shadow-xs' 
                              : 'bg-slate-200 hover:bg-slate-300 text-slate-800 border border-slate-300'
                          }`}
                        >
                          {lvl === 'All'
                            ? `All Projects (${ALL_REPOSITORY_PROJECTS.length})`
                            : `${lvl} (${ALL_REPOSITORY_PROJECTS.filter(p => p.difficulty === lvl).length})`}
                        </button>
                      ))}
                      <button
                        onClick={() => setStarredOnlyFilter(prev => !prev)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-200 flex items-center gap-1.5 border ${
                          starredOnlyFilter
                            ? 'bg-amber-500 text-white border-amber-600 shadow-xs'
                            : 'bg-amber-50 hover:bg-amber-100 text-amber-900 border-amber-300'
                        }`}
                        title="Filter projects starred & pinned to Storyboard"
                      >
                        <Star className={`w-3.5 h-3.5 ${starredOnlyFilter ? 'fill-white text-white' : 'fill-amber-400 text-amber-500'}`} />
                        <span>Pinned to Storyboard ({starredRepoIds.length})</span>
                      </button>
                    </div>

                    {/* Search Input */}
                    <div className="relative w-full md:w-72">
                      <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
                      <input 
                        type="text"
                        value={repoSearch}
                        onChange={e => setRepoSearch(e.target.value)}
                        placeholder="Search projects, sources, tech..."
                        className="w-full bg-white border border-slate-300 rounded-xl pl-9 pr-4 py-2 text-xs text-slate-900 placeholder:text-slate-500 focus:outline-none focus:border-blue-900 transition-colors shadow-xs"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* LangChain + FAISS RAG Document Ingestion Section (Collapsed by default so all projects are immediately visible) */}
              <RepoRagIngestionPanel
                ragStatus={ragStatus}
                onRefreshStatus={fetchRagStatus}
                onOpenCopilot={() => setActiveTab('copilot')}
              />

              {/* Explicit Visible Catalog Counter + Reset Filters */}
              {(() => {
                const filteredRepoProjects = ALL_REPOSITORY_PROJECTS.filter(proj => {
                  const matchesLevel = repoFilter === 'All' || proj.difficulty === repoFilter;
                  const matchesStarred = !starredOnlyFilter || starredRepoIds.includes(proj.id);
                  const q = repoSearch.trim().toLowerCase();
                  const matchesSearch = !q || 
                    proj.title.toLowerCase().includes(q) ||
                    proj.domain.toLowerCase().includes(q) ||
                    (proj.sourceName || '').toLowerCase().includes(q) ||
                    proj.techStack.some(t => t.toLowerCase().includes(q));
                  return matchesLevel && matchesStarred && matchesSearch;
                });
                const hasActiveFilter = repoFilter !== 'All' || starredOnlyFilter || repoSearch.trim().length > 0;

                return (
                  <div className="space-y-5">
                    <div className="flex flex-wrap items-center justify-between gap-3 bg-white border border-slate-300 rounded-xl px-4 py-3 shadow-2xs">
                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        <span className="font-bold text-slate-900">
                          Showing {filteredRepoProjects.length} of {ALL_REPOSITORY_PROJECTS.length} ML &amp; AI Case Studies
                        </span>
                        <span className="text-slate-400">·</span>
                        <span className="text-slate-600">
                          Click the <strong className="text-amber-700">Star</strong> button on any project to pin it to <strong className="text-indigo-700">Storyboard</strong>
                        </span>
                      </div>
                      {hasActiveFilter && (
                        <button
                          type="button"
                          onClick={() => {
                            setRepoFilter('All');
                            setStarredOnlyFilter(false);
                            setRepoSearch('');
                          }}
                          className="px-3 py-1.5 rounded-lg bg-blue-900 hover:bg-blue-800 text-white text-xs font-semibold transition-colors"
                        >
                          Show All {ALL_REPOSITORY_PROJECTS.length} Projects
                        </button>
                      )}
                    </div>

                    {/* Projects Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                      {filteredRepoProjects.map(proj => {
                        const isStarred = starredRepoIds.includes(proj.id);
                        return (
                    <div 
                      key={proj.id}
                      onClick={() => handleViewRepoSpecs(proj)}
                      className={`cursor-pointer rounded-2xl bg-slate-100 border border-slate-300 p-6 transition-all duration-200 hover:bg-slate-200 hover:shadow-lg flex flex-col justify-between ${
                        selectedRepoProject?.id === proj.id ? 'ring-2 ring-indigo-600/30 bg-slate-200 border-indigo-600' : ''
                      }`}
                    >
                      <div>
                        <div className="flex items-start justify-between gap-2 mb-3">
                          <span className="text-xs font-semibold text-blue-900 uppercase tracking-wider leading-snug">{proj.domain}</span>
                          <div className="flex items-center gap-1.5 shrink-0">
                            <span className={`text-xs px-2.5 py-0.5 rounded-md font-medium border ${
                              proj.difficulty === 'Beginner' 
                                ? 'bg-emerald-100 text-emerald-900 border-emerald-300'
                                : proj.difficulty === 'Intermediate'
                                ? 'bg-blue-100 text-blue-900 border-blue-300'
                                : proj.difficulty === 'Research Grade'
                                ? 'bg-rose-100 text-rose-900 border-rose-300 font-semibold'
                                : 'bg-purple-100 text-purple-900 border-purple-300'
                            }`}>
                              {proj.difficulty}
                            </span>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleStarRepoProject(proj.id);
                              }}
                              aria-label={isStarred ? `Unpin ${proj.title} from Storyboard` : `Star to pin ${proj.title} to Storyboard`}
                              title={isStarred ? 'Pinned to Storyboard (Click to unstar)' : 'Star to pin to Storyboard'}
                              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-lg border text-[11px] font-sans font-semibold transition-all ${
                                isStarred
                                  ? 'bg-amber-50 hover:bg-amber-100 text-amber-900 border-amber-300 shadow-2xs'
                                  : 'bg-white hover:bg-amber-50 text-slate-600 hover:text-amber-800 border-slate-300'
                              }`}
                            >
                              <Star className={`w-3.5 h-3.5 ${isStarred ? 'fill-amber-400 text-amber-500' : 'text-slate-400'}`} />
                              <span>{isStarred ? 'Pinned' : 'Star'}</span>
                            </button>
                          </div>
                        </div>
                        <h3 className="text-lg font-bold text-slate-900 mb-2 leading-snug">{proj.title}</h3>
                        <p className="text-slate-700 text-sm mb-4 line-clamp-3 leading-relaxed">{proj.description}</p>
                      </div>
                      
                      <div className="space-y-4">
                        {/* Training Loss & Accuracy Preview in Silver Grey Box */}
                        <div className="grid grid-cols-2 gap-2 bg-slate-200/80 hover:bg-slate-300 p-2.5 rounded-xl border border-slate-300 text-xs font-mono transition-all duration-200">
                          <div>
                            <span className="text-slate-600 block text-[10px]">Train Loss</span>
                            <span className="text-blue-900 font-semibold">{proj.trainingLoss.split(' ')[0]}</span>
                          </div>
                          <div>
                            <span className="text-slate-600 block text-[10px]">Train Accuracy</span>
                            <span className="text-emerald-900 font-semibold">{proj.trainingAccuracy.split(' ')[0]}</span>
                          </div>
                        </div>

                        {/* CapEx & 20-Yr YoY OpEx Financial Preview */}
                        {(() => {
                          const econ = computeProjectEconomics(proj);
                          return (
                            <div className="grid grid-cols-3 gap-1.5 bg-white p-2 rounded-xl border border-slate-300 text-[11px] font-mono tabular-nums">
                              <div>
                                <span className="text-slate-500 block text-[9.5px]">Total CapEx</span>
                                <span className="text-blue-950 font-bold block">{formatUsdOnly(econ.totalCapExUsd)}</span>
                                <span className="text-slate-500 text-[9px] block">{formatRsOnly(econ.totalCapExUsd)}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block text-[9.5px]">OpEx YoY (Y1)</span>
                                <span className="text-emerald-900 font-bold block">{formatUsdOnly(econ.opExYear1Usd)}</span>
                                <span className="text-slate-500 text-[9px] block">{formatRsOnly(econ.opExYear1Usd)}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block text-[9.5px]">20-Yr OpEx</span>
                                <span className="text-indigo-950 font-bold block">{formatUsdOnly(econ.totalOpEx20YrUsd)}</span>
                                <span className="text-slate-500 text-[9px] block">{formatRsOnly(econ.totalOpEx20YrUsd)}</span>
                              </div>
                            </div>
                          );
                        })()}

                        <div className="flex flex-wrap gap-1.5">
                          {proj.techStack.map((tech, i) => (
                            <span key={i} className="text-xs px-2 py-0.5 rounded font-mono bg-slate-200 text-slate-800 border border-slate-300">
                              {tech}
                            </span>
                          ))}
                        </div>
                        
                        <div className="pt-3 border-t border-slate-300 flex items-center justify-between">
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              setWhitePaperModalProject(proj);
                            }}
                            className="px-3 py-1.5 bg-indigo-100 hover:bg-indigo-200 text-indigo-900 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors border border-indigo-300"
                          >
                            <BookOpen className="w-3.5 h-3.5" />
                            <span>White Paper</span>
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleViewRepoSpecs(proj);
                            }}
                            className="px-3 py-1.5 bg-blue-900 hover:bg-blue-800 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-2xs"
                          >
                            <span>View Specs &amp; Code</span>
                            <ArrowRight className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
                    </div>
                  </div>
                );
              })()}

              {/* Selected Repository Project Detailed View */}
              {selectedRepoProject && (
                <div
                  id="repo-specs-panel"
                  ref={repoSpecsRef}
                  className="scroll-mt-24 rounded-2xl bg-slate-100 border border-slate-300 p-8 space-y-10 shadow-md"
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-300">
                    <div>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-700 font-medium mb-1">
                        <span className="font-semibold text-slate-900">{selectedRepoProject.domain}</span>
                        <span>·</span>
                        <span className="text-blue-900 font-bold">{selectedRepoProject.metrics}</span>
                      </div>
                      <h2 className="text-2xl font-bold text-slate-900">{selectedRepoProject.title}</h2>
                    </div>
                    <div className="flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                        className="px-3 py-2 bg-white hover:bg-slate-200 text-slate-700 border border-slate-300 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
                      >
                        <ArrowUp className="w-3.5 h-3.5" />
                        <span>Back to Projects</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => toggleStarRepoProject(selectedRepoProject.id)}
                        className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors border shadow-xs ${
                          starredRepoIds.includes(selectedRepoProject.id)
                            ? 'bg-amber-50 hover:bg-amber-100 text-amber-900 border-amber-300'
                            : 'bg-white hover:bg-amber-50 text-slate-800 border-slate-300'
                        }`}
                      >
                        <Star
                          className={`w-4 h-4 ${
                            starredRepoIds.includes(selectedRepoProject.id)
                              ? 'fill-amber-400 text-amber-500'
                              : 'text-slate-500'
                          }`}
                        />
                        <span>
                          {starredRepoIds.includes(selectedRepoProject.id)
                            ? 'Pinned on Storyboard'
                            : 'Star to Pin on Storyboard'}
                        </span>
                      </button>
                      <button 
                        onClick={() => {
                          setCodeRepoTargetProjectId(selectedRepoProject.id);
                          setActiveTab('coderepo');
                        }}
                        className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-900 border border-slate-300 rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs"
                      >
                        <FolderGit2 className="w-4 h-4 text-blue-900" />
                        <span>Code Repo</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => downloadProjectNotebook(selectedRepoProject)}
                        className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs cursor-pointer"
                      >
                        <BookOpen className="w-4 h-4" />
                        <span>Download .ipynb (Colab / Jupyter)</span>
                      </button>
                      <button 
                        onClick={async () => {
                          await downloadProjectGitZip(selectedRepoProject);
                        }}
                        className="px-4 py-2 bg-emerald-700 hover:bg-emerald-600 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs cursor-pointer"
                      >
                        <Download className="w-4 h-4" />
                        <span>Download Git Repo (.zip)</span>
                      </button>
                      <button 
                        onClick={() => setWhitePaperModalProject(selectedRepoProject)}
                        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-md"
                      >
                        <BookOpen className="w-4 h-4" />
                        <span>Read Full Technical White Paper</span>
                      </button>
                    </div>
                  </div>

                  {/* PROBLEM STATEMENT & BREAKING GROUND INNOVATION */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-slate-100 hover:bg-slate-200 p-6 rounded-xl border border-slate-300 space-y-3 transition-all duration-200 shadow-xs">
                      <div className="flex items-center gap-2 text-rose-900 font-bold text-sm">
                        <AlertCircle className="w-4 h-4" />
                        <span>Problem Statement (Google Scholar Spec)</span>
                      </div>
                      <p className="text-slate-800 text-sm leading-relaxed">{selectedRepoProject.problem}</p>
                    </div>

                    <div className="bg-slate-100 hover:bg-slate-200 p-6 rounded-xl border border-slate-300 space-y-3 transition-all duration-200 shadow-xs">
                      <div className="flex items-center gap-2 text-blue-900 font-bold text-sm">
                        <Lightbulb className="w-4 h-4" />
                        <span>Breaking Ground & Innovation</span>
                      </div>
                      <p className="text-slate-800 text-sm leading-relaxed">{selectedRepoProject.breakingGround}</p>
                    </div>
                  </div>

                  {/* TRAINING LOSS, ACCURACY, DEVIATION, EFFICIENCY & MODEL COMPARISON GRAPHS */}
                  <ProjectTelemetryCharts project={selectedRepoProject} />

                  {/* TOTAL CAPEX, 20-YEAR YOY OPEX & OPERATIONAL COST ARCHITECTURE */}
                  <ProjectEconomicsPanel project={selectedRepoProject} />

                  {/* SYNTHETIC DATASET & SELECTED DATA TYPE TRAINING LAUNCHER */}
                  <ProjectSyntheticDatasetPanel
                    project={selectedRepoProject}
                    onRunTrainingWithDataType={handleLaunchTrainingFromProject}
                  />

                  {/* Architecture Flow Diagram Section */}
                  <div className="bg-slate-100 hover:bg-slate-200 p-6 rounded-xl border border-slate-300 transition-all duration-200 shadow-xs">
                    <h3 className="text-sm font-bold text-slate-900 mb-4 flex items-center gap-2">
                      <Network className="w-4 h-4 text-blue-900" />
                      <span>System Architecture & Data Flow Topology (Google Scholar Spec)</span>
                    </h3>
                    <div className="flex flex-wrap items-center gap-3">
                      {selectedRepoProject.architectureFlow.map((node, i) => (
                        <React.Fragment key={i}>
                          <div className="px-3.5 py-2 rounded-lg bg-white border border-slate-300 text-blue-950 text-xs font-mono font-semibold shadow-xs">
                            <span className="text-blue-900 font-bold mr-1.5">0{i+1}.</span>
                            {node}
                          </div>
                          {i < selectedRepoProject.architectureFlow.length - 1 && (
                            <ArrowRight className="w-4 h-4 text-slate-400 shrink-0" />
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                  </div>

                  {/* Annotated Pipeline Steps & Code */}
                  <AnnotatedCodePipeline
                    project={selectedRepoProject}
                    bannerTitle={`${selectedRepoProject.sourceName || 'DataCamp'} Repository Executable Pipeline (Annotated)`}
                    bannerSubtitle="Annotated production pipeline with inline module comments, breakdown descriptions, feature engineering, training loop, and artifact serialization."
                    copiedCodeIndex={copiedCodeIndex}
                    copiedAllId={copiedAllId}
                    indexOffset={200}
                    onCopyCode={handleCopyCode}
                    onCopyAll={handleCopyAllPipeline}
                  />
                </div>
              )}
            </div>
          )}

          {/* TAB 1: STORYBOARD (STARRED / PINNED PROJECTS ONLY) */}
          {activeTab === 'showcase' && (
            <div>
              {researchProjects.length === 0 ? (
                <div className="rounded-2xl bg-slate-100 border border-slate-300 p-12 text-center space-y-4 mb-12 shadow-xs">
                  <div className="w-12 h-12 rounded-2xl bg-amber-50 border border-amber-200 flex items-center justify-center mx-auto">
                    <Star className="w-6 h-6 text-amber-500" />
                  </div>
                  <div className="space-y-1 max-w-md mx-auto">
                    <h3 className="text-lg font-bold text-slate-900">No Starred Projects on Storyboard</h3>
                    <p className="text-sm text-slate-600">
                      Select the star icon on any project in ML Case Studies to pin it to your Storyboard.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setActiveTab('repository')}
                    className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-xl text-xs inline-flex items-center gap-2 transition-colors shadow-sm"
                  >
                    <Database className="w-4 h-4" />
                    <span>Browse ML Case Studies</span>
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
                  {researchProjects.map(proj => (
                    <div 
                      key={proj.id}
                      onClick={() => handleViewStoryboardSpecs(proj)}
                      className={`cursor-pointer rounded-2xl bg-slate-100 border border-slate-300 p-6 transition-all duration-200 hover:bg-slate-200 hover:shadow-lg flex flex-col justify-between ${activeStoryboardProject?.id === proj.id ? 'ring-2 ring-indigo-600/30 bg-slate-200 border-indigo-600' : ''}`}
                    >
                      <div>
                        <div className="flex items-start justify-between gap-2 mb-3">
                          <span className="text-xs font-semibold text-blue-900 uppercase tracking-wider leading-snug">{proj.domain}</span>
                          <div className="flex items-center gap-1.5 shrink-0">
                            <span className="text-xs px-2.5 py-0.5 rounded-md font-medium bg-emerald-100 text-emerald-900 border border-emerald-300">
                              {proj.difficulty}
                            </span>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                toggleStarRepoProject(proj.id);
                              }}
                              title="Unstar to remove from Storyboard"
                              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg bg-amber-50 hover:bg-amber-100 text-amber-900 border border-amber-300 text-[11px] font-sans font-semibold transition-colors"
                            >
                              <Star className="w-3 h-3 fill-amber-400 text-amber-500" />
                              <span>Pinned</span>
                            </button>
                          </div>
                        </div>
                        <h3 className="text-lg font-bold text-slate-900 mb-2 leading-snug">{proj.title}</h3>
                        <p className="text-slate-700 text-sm mb-4 line-clamp-3 leading-relaxed">{proj.description}</p>
                      </div>
                      
                      <div className="space-y-4">
                        {/* Training Loss & Accuracy Preview in Silver Grey Box */}
                        <div className="grid grid-cols-2 gap-2 bg-slate-200/80 hover:bg-slate-300 p-2.5 rounded-xl border border-slate-300 text-xs font-mono transition-all duration-200">
                          <div>
                            <span className="text-slate-600 block text-[10px]">Train Loss</span>
                            <span className="text-blue-900 font-semibold">{proj.trainingLoss.split(' ')[0]}</span>
                          </div>
                          <div>
                            <span className="text-slate-600 block text-[10px]">Train Accuracy</span>
                            <span className="text-emerald-900 font-semibold">{proj.trainingAccuracy.split(' ')[0]}</span>
                          </div>
                        </div>

                        {/* CapEx & 20-Yr YoY OpEx Financial Preview */}
                        {(() => {
                          const econ = computeProjectEconomics(proj);
                          return (
                            <div className="grid grid-cols-3 gap-1.5 bg-white p-2 rounded-xl border border-slate-300 text-[11px] font-mono tabular-nums">
                              <div>
                                <span className="text-slate-500 block text-[9.5px]">Total CapEx</span>
                                <span className="text-blue-950 font-bold block">{formatUsdOnly(econ.totalCapExUsd)}</span>
                                <span className="text-slate-500 text-[9px] block">{formatRsOnly(econ.totalCapExUsd)}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block text-[9.5px]">OpEx YoY (Y1)</span>
                                <span className="text-emerald-900 font-bold block">{formatUsdOnly(econ.opExYear1Usd)}</span>
                                <span className="text-slate-500 text-[9px] block">{formatRsOnly(econ.opExYear1Usd)}</span>
                              </div>
                              <div>
                                <span className="text-slate-500 block text-[9.5px]">20-Yr OpEx</span>
                                <span className="text-indigo-950 font-bold block">{formatUsdOnly(econ.totalOpEx20YrUsd)}</span>
                                <span className="text-slate-500 text-[9px] block">{formatRsOnly(econ.totalOpEx20YrUsd)}</span>
                              </div>
                            </div>
                          );
                        })()}

                        <div className="flex flex-wrap gap-1.5">
                          {proj.techStack.slice(0, 3).map((tech, i) => (
                            <span key={i} className={`text-xs px-2 py-0.5 rounded font-mono ${tech === 'Rust' ? 'bg-amber-100 text-amber-900 border border-amber-300 font-semibold' : 'bg-slate-200 text-slate-800 border border-slate-300'}`}>
                              {tech}
                            </span>
                          ))}
                        </div>
                        
                        <div className="pt-3 border-t border-slate-300 flex items-center justify-between">
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              setWhitePaperModalProject(proj);
                            }}
                            className="px-3 py-1.5 bg-indigo-100 hover:bg-indigo-200 text-indigo-900 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors border border-indigo-300"
                          >
                            <BookOpen className="w-3.5 h-3.5" />
                            <span>Technical White Paper</span>
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              handleViewStoryboardSpecs(proj);
                            }}
                            className="px-3 py-1.5 bg-blue-900 hover:bg-blue-800 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer shadow-2xs"
                          >
                            <span>View Specs</span>
                            <ArrowRight className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Selected Project Detailed View: Training Convergence Chart, Sample Distribution, Problem, Breaking Ground, Architecture, & Code */}
              {activeStoryboardProject && (
                <div
                  id="storyboard-specs-panel"
                  ref={storyboardSpecsRef}
                  className="scroll-mt-24 rounded-2xl bg-slate-100 border border-slate-300 p-8 space-y-10 shadow-md"
                >
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-300">
                    <div>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-700 font-medium mb-1">
                        <span className="font-semibold text-slate-900">{activeStoryboardProject.domain}</span>
                        <span>·</span>
                        <span className="text-blue-900 font-bold">{activeStoryboardProject.metrics}</span>
                      </div>
                      <h2 className="text-2xl font-bold text-slate-900">{activeStoryboardProject.title}</h2>
                    </div>
                    <div className="flex flex-wrap items-center gap-3">
                      <button
                        type="button"
                        onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                        className="px-3 py-2 bg-white hover:bg-slate-200 text-slate-700 border border-slate-300 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
                      >
                        <ArrowUp className="w-3.5 h-3.5" />
                        <span>Back to Projects</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => toggleStarRepoProject(activeStoryboardProject.id)}
                        className="px-4 py-2 bg-amber-50 hover:bg-amber-100 text-amber-900 border border-amber-300 rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs"
                      >
                        <Star className="w-4 h-4 fill-amber-400 text-amber-500" />
                        <span>Unpin / Remove Star</span>
                      </button>
                      <button 
                        onClick={() => {
                          setCodeRepoTargetProjectId(activeStoryboardProject.id);
                          setActiveTab('coderepo');
                        }}
                        className="px-4 py-2 bg-slate-200 hover:bg-slate-300 text-slate-900 border border-slate-300 rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs"
                      >
                        <FolderGit2 className="w-4 h-4 text-blue-900" />
                        <span>Code Repo</span>
                      </button>
                      <button
                        type="button"
                        onClick={() => downloadProjectNotebook(activeStoryboardProject)}
                        className="px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs cursor-pointer"
                      >
                        <BookOpen className="w-4 h-4" />
                        <span>Download .ipynb (Colab / Jupyter)</span>
                      </button>
                      <button 
                        onClick={async () => {
                          await downloadProjectGitZip(activeStoryboardProject);
                        }}
                        className="px-4 py-2 bg-emerald-700 hover:bg-emerald-600 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-xs cursor-pointer"
                      >
                        <Download className="w-4 h-4" />
                        <span>Download Git Repo (.zip)</span>
                      </button>
                      <button 
                        onClick={() => setWhitePaperModalProject(activeStoryboardProject)}
                        className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center gap-2 transition-colors shadow-md"
                      >
                        <BookOpen className="w-4 h-4" />
                        <span>Read Full Technical White Paper</span>
                      </button>
                    </div>
                  </div>

                  {/* PROBLEM STATEMENT & BREAKING GROUND INNOVATION */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="bg-slate-100 hover:bg-slate-200 p-6 rounded-xl border border-slate-300 space-y-3 transition-all duration-200 shadow-xs">
                      <div className="flex items-center gap-2 text-rose-900 font-bold text-sm">
                        <AlertCircle className="w-4 h-4" />
                        <span>Problem Statement (Google Scholar Spec)</span>
                      </div>
                      <p className="text-slate-800 text-sm leading-relaxed">{activeStoryboardProject.problem}</p>
                    </div>

                    <div className="bg-slate-100 hover:bg-slate-200 p-6 rounded-xl border border-slate-300 space-y-3 transition-all duration-200 shadow-xs">
                      <div className="flex items-center gap-2 text-blue-900 font-bold text-sm">
                        <Lightbulb className="w-4 h-4" />
                        <span>Breaking Ground & Innovation</span>
                      </div>
                      <p className="text-slate-800 text-sm leading-relaxed">{activeStoryboardProject.breakingGround}</p>
                    </div>
                  </div>

                  {/* TRAINING LOSS, ACCURACY, DEVIATION, EFFICIENCY & MODEL COMPARISON GRAPHS */}
                  <ProjectTelemetryCharts project={activeStoryboardProject} />

                  {/* TOTAL CAPEX, 20-YEAR YOY OPEX & OPERATIONAL COST ARCHITECTURE */}
                  <ProjectEconomicsPanel project={activeStoryboardProject} />

                  {/* SYNTHETIC DATASET & SELECTED DATA TYPE TRAINING LAUNCHER */}
                  <ProjectSyntheticDatasetPanel
                    project={activeStoryboardProject}
                    onRunTrainingWithDataType={handleLaunchTrainingFromProject}
                  />

                  {/* Architecture Flow Diagram Section */}
                  <div className="bg-slate-100 hover:bg-slate-200 p-6 rounded-xl border border-slate-300 transition-all duration-200 shadow-xs">
                    <h3 className="text-sm font-bold text-slate-900 mb-4 flex items-center gap-2">
                      <Network className="w-4 h-4 text-blue-900" />
                      <span>System Architecture & Data Flow Topology (Google Scholar Spec)</span>
                    </h3>
                    <div className="flex flex-wrap items-center gap-3">
                      {activeStoryboardProject.architectureFlow.map((node, i) => (
                        <React.Fragment key={i}>
                          <div className="px-3.5 py-2 rounded-lg bg-white border border-slate-300 text-blue-950 text-xs font-mono font-semibold shadow-xs">
                            <span className="text-blue-900 font-bold mr-1.5">0{i+1}.</span>
                            {node}
                          </div>
                          {i < activeStoryboardProject.architectureFlow.length - 1 && (
                            <ArrowRight className="w-4 h-4 text-slate-400 shrink-0" />
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                  </div>

                  {/* Annotated Pipeline Steps & Code */}
                  <AnnotatedCodePipeline
                    project={activeStoryboardProject}
                    bannerTitle="Starred Research & Production Pipeline (Annotated)"
                    bannerSubtitle="Annotated production scripts with header documentation, line-block descriptions, distributed training convergence loops, and Rust-accelerated kernels."
                    copiedCodeIndex={copiedCodeIndex}
                    copiedAllId={copiedAllId}
                    indexOffset={0}
                    onCopyCode={handleCopyCode}
                    onCopyAll={handleCopyAllPipeline}
                  />
                </div>
              )}
            </div>
          )}

          {/* TAB 2: PROJECT BUILDER WIZARD */}
          {activeTab === 'builder' && (
            <div className="space-y-8">
              <div>
                <div className="flex items-center gap-2 text-xs font-mono text-rose-900 mb-1 font-semibold">
                  <span className="bg-rose-100 px-2 py-0.5 rounded text-rose-900 font-bold border border-rose-200">[Gemini Architecture Synthesizer]</span>
                  <span>Google Scholar Specification Generator</span>
                </div>
                <h2 className="text-2xl font-bold text-slate-900 tracking-tight">AI Project Builder Wizard & White Paper Generator</h2>
                <p className="text-slate-600 text-sm mt-1">Define your custom project requirements and let Gemini generate a complete production architecture, python code, and a formal Technical White Paper.</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                {/* Form Input Column */}
                <div className="lg:col-span-5 bg-slate-100 hover:bg-slate-200/90 border border-slate-300 rounded-2xl p-6 transition-all duration-200 shadow-xs">
                  <form onSubmit={handleGenerateBlueprint} className="space-y-4">
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">Project Name</label>
                      <input 
                        type="text" 
                        value={builderForm.projectName}
                        onChange={e => setBuilderForm({ ...builderForm, projectName: e.target.value })}
                        className="w-full bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">ML Domain</label>
                      <input 
                        type="text" 
                        value={builderForm.domain}
                        onChange={e => setBuilderForm({ ...builderForm, domain: e.target.value })}
                        className="w-full bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">Core Research Objective & Problem</label>
                      <textarea 
                        rows={3}
                        value={builderForm.objective}
                        onChange={e => setBuilderForm({ ...builderForm, objective: e.target.value })}
                        className="w-full bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:border-blue-900 resize-none shadow-xs"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">Data Source Type</label>
                      <input 
                        type="text" 
                        value={builderForm.dataType}
                        onChange={e => setBuilderForm({ ...builderForm, dataType: e.target.value })}
                        className="w-full bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">Target Evaluation Metric</label>
                      <input 
                        type="text" 
                        value={builderForm.targetMetric}
                        onChange={e => setBuilderForm({ ...builderForm, targetMetric: e.target.value })}
                        className="w-full bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
                        required
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1.5">Deployment Target</label>
                      <input 
                        type="text" 
                        value={builderForm.deploymentTarget}
                        onChange={e => setBuilderForm({ ...builderForm, deploymentTarget: e.target.value })}
                        className="w-full bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:border-blue-900 shadow-xs"
                        required
                      />
                    </div>

                    <button 
                      type="submit"
                      disabled={isGenerating}
                      className="w-full py-3 bg-blue-900 hover:bg-blue-800 disabled:opacity-50 text-white font-semibold rounded-xl transition-all shadow-md flex items-center justify-center gap-2 text-sm mt-6"
                    >
                      {isGenerating ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          <span>Architecting Research Project with Gemini...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-4 h-4" />
                          <span>Generate Research Architecture & White Paper</span>
                        </>
                      )}
                    </button>
                  </form>
                </div>

                {/* Generated Blueprint Output Column */}
                <div className="lg:col-span-7">
                  {builderError && (
                    <div className="p-4 bg-rose-50 border border-rose-200 text-rose-900 rounded-xl text-sm mb-6 font-medium">
                      {builderError}
                    </div>
                  )}

                  {!generatedBlueprint && !isGenerating && (
                    <div className="h-96 rounded-2xl bg-slate-100 hover:bg-slate-200 border border-slate-300 border-dashed flex flex-col items-center justify-center p-8 text-center transition-colors">
                      <Workflow className="w-12 h-12 text-blue-900 mb-4" />
                      <h3 className="text-lg font-semibold text-slate-900 mb-2">No Research Blueprint Generated Yet</h3>
                      <p className="text-slate-600 text-sm max-w-md">Configure your research project parameters on the left and click generate to build a complete ML architecture blueprint and technical white paper.</p>
                    </div>
                  )}

                  {isGenerating && (
                    <div className="h-96 rounded-2xl bg-slate-100 border border-slate-300 flex flex-col items-center justify-center p-8 text-center space-y-4">
                      <div className="w-12 h-12 rounded-full border-2 border-blue-900 border-t-transparent animate-spin" />
                      <p className="text-slate-900 font-medium text-sm">Synthesizing research-grade neural pipelines, Python codebase, & Technical White Paper...</p>
                    </div>
                  )}

                  {generatedBlueprint && !isGenerating && (
                    <div className="space-y-6">
                      <div className="p-6 rounded-2xl bg-slate-100 hover:bg-slate-200 border border-slate-300 flex items-center justify-between transition-colors shadow-xs">
                        <div>
                          <div className="flex items-center gap-3 mb-2">
                            <span className="text-xs font-semibold text-rose-900 uppercase tracking-wider font-mono">[Research Project Blueprint]</span>
                            <span className="text-xs px-2.5 py-1 rounded-md bg-blue-100 text-blue-900 font-semibold border border-blue-200">Est. {generatedBlueprint.estimatedTimeWeeks} Weeks</span>
                          </div>
                          <p className="text-slate-800 text-sm leading-relaxed mb-4">{generatedBlueprint.overview}</p>
                        </div>
                        <button 
                          onClick={() => setShowBuilderWhitePaper(true)}
                          className="px-4 py-2.5 bg-blue-900 hover:bg-blue-800 text-white rounded-xl text-xs font-semibold flex items-center gap-2 shrink-0 transition-colors shadow-sm"
                        >
                          <BookOpen className="w-4 h-4" />
                          <span>View White Paper</span>
                        </button>
                      </div>

                      <div className="space-y-4">
                        {generatedBlueprint.phases?.map((phase, idx) => (
                          <div key={idx} className="p-6 rounded-2xl bg-slate-100 hover:bg-slate-200 border border-slate-300 transition-colors shadow-xs">
                            <div className="flex items-center gap-3 mb-2">
                              <span className="w-6 h-6 rounded-full bg-blue-100 text-blue-900 border border-blue-200 flex items-center justify-center text-xs font-bold">
                                {phase.stepNumber}
                              </span>
                              <h3 className="text-base font-bold text-slate-900">{phase.title}</h3>
                            </div>
                            <p className="text-slate-700 text-sm mb-4 leading-relaxed">{phase.description}</p>
                            
                            <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden shadow-sm">
                              <div className="bg-slate-950 px-4 py-2 border-b border-slate-800 flex items-center justify-between">
                                <span className="text-xs font-mono text-slate-400">phase_{phase.stepNumber}_code.py</span>
                                <button 
                                  onClick={() => handleCopyCode(phase.pythonSnippet, 100 + idx)}
                                  className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800"
                                >
                                  {copiedCodeIndex === 100 + idx ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                                  <span>{copiedCodeIndex === 100 + idx ? 'Copied' : 'Copy'}</span>
                                </button>
                              </div>
                              <pre className="p-4 text-xs font-mono text-indigo-200 overflow-x-auto leading-relaxed">
                                <code>{phase.pythonSnippet}</code>
                              </pre>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: ML TRAINING SIMULATOR */}
          {activeTab === 'training' && (
            <MLTrainingSimulatorTab
              allProjects={allProjectsCatalog}
              researchProjects={researchProjects}
              repoProjects={nonStarredRepoProjects}
              starredRepoIds={starredRepoIds}
              simConfig={simConfig}
              setSimConfig={setSimConfig}
              isTraining={isTraining}
              trainingEpoch={trainingEpoch}
              trainingLogs={trainingLogs}
              onStartTraining={runTrainingSimulation}
            />
          )}

          {/* TAB 4: AI ML COPILOT CHAT */}
          {activeTab === 'copilot' && (
            <div className="space-y-6">
              <div>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2 text-xs font-mono text-rose-900 mb-1 font-semibold">
                      <span className="bg-rose-100 px-2 py-0.5 rounded text-rose-900 font-bold border border-rose-200">[RAG AI Copilot]</span>
                      <span>LangChain + FAISS IndexFlatIP Grounded Assistant</span>
                    </div>
                    <h2 className="text-2xl font-bold text-slate-900 tracking-tight">AI/ML Expert RAG Copilot</h2>
                    <p className="text-slate-600 text-sm mt-1">Ask questions grounded in PDFs &amp; raw text indexed from the Repository tab, or discuss architectures, loss curves, and Rust bindings.</p>
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setActiveTab('repository')}
                      className="px-3 py-1.5 rounded-xl bg-slate-200 hover:bg-slate-300 text-slate-900 text-xs font-semibold border border-slate-300 transition-colors flex items-center gap-1.5"
                    >
                      <Database className="w-3.5 h-3.5 text-rose-900" />
                      <span>Ingest More Docs ({ragStatus?.totalDocuments || 0} indexed)</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Real-Time Copilot Monitor Dashboard Overlay */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 bg-white border border-slate-300 rounded-xl p-4 shadow-xs">
                <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1 font-mono">
                  <span className="text-[10px] text-slate-500 uppercase font-bold block">API Latency</span>
                  <div className="flex items-baseline justify-between">
                    <span className="text-sm font-bold text-slate-900">{copilotMonitor.lastLatencyMs} ms</span>
                    <span className="text-xs text-blue-900 font-semibold">Avg: {copilotMonitor.avgLatencyMs} ms</span>
                  </div>
                </div>

                <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1 font-mono">
                  <span className="text-[10px] text-slate-500 uppercase font-bold block">Token Consumption (Prompt)</span>
                  <div className="flex items-baseline justify-between">
                    <span className="text-sm font-bold text-emerald-900">{copilotMonitor.lastPromptTokens} in / {copilotMonitor.lastCompletionTokens} out</span>
                    <span className="text-xs text-slate-600">Total: {copilotMonitor.totalInputTokens + copilotMonitor.totalOutputTokens}</span>
                  </div>
                </div>

                <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1 font-mono">
                  <span className="text-[10px] text-slate-500 uppercase font-bold block">Request Success Rate</span>
                  <div className="flex items-baseline justify-between">
                    <span className="text-sm font-bold text-emerald-800">
                      {copilotMonitor.totalRequests > 0
                        ? `${Math.round((copilotMonitor.successCount / copilotMonitor.totalRequests) * 100)}%`
                        : '100%'}
                    </span>
                    <span className="text-xs text-slate-600">{copilotMonitor.successCount}/{copilotMonitor.totalRequests} OK</span>
                  </div>
                </div>

                <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1 font-mono">
                  <span className="text-[10px] text-slate-500 uppercase font-bold block">Active Endpoint / Model</span>
                  <div className="flex items-baseline justify-between">
                    <span className="text-xs font-bold text-indigo-900 truncate">Gemini 4 Hydrogen</span>
                    <span className="text-[10px] text-emerald-700 font-bold">Online</span>
                  </div>
                </div>
              </div>

              <div className="rounded-2xl bg-slate-100 border border-slate-300 flex flex-col h-[620px] overflow-hidden shadow-xs">
                <div className="bg-slate-200/90 px-6 py-3.5 border-b border-slate-300 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <div className={`w-3 h-3 rounded-full ${(ragStatus?.totalVectors || 0) > 0 && useRagGrounding ? 'bg-emerald-600 animate-pulse' : 'bg-blue-700'}`} />
                    <span className="text-xs font-bold text-slate-900">
                      Gemini RAG Copilot · FAISS IndexFlatIP ({ragStatus?.totalVectors || 0} vectors across {ragStatus?.totalDocuments || 0} docs)
                    </span>
                  </div>
                  <div className="flex flex-wrap items-center gap-3">
                    <select
                      value={copilotSearchMode}
                      onChange={(e) => setCopilotSearchMode(e.target.value as 'hybrid' | 'dense' | 'bm25')}
                      className="bg-white border border-slate-300 rounded-lg px-2.5 py-1 text-xs font-mono font-semibold text-slate-800 focus:outline-none focus:border-blue-900"
                    >
                      <option value="hybrid">Hybrid (Dense + BM25 RRF)</option>
                      <option value="dense">Dense Vector Only</option>
                      <option value="bm25">BM25 Sparse Only</option>
                    </select>
                    <label className="flex items-center gap-2 text-xs font-semibold text-slate-800 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={useRagGrounding}
                        onChange={(e) => setUseRagGrounding(e.target.checked)}
                        className="rounded border-slate-400 text-blue-900 focus:ring-blue-900"
                      />
                      <span>Use Hybrid RAG Retrieval</span>
                    </label>
                  </div>
                </div>

                {/* Chat Messages */}
                <div className="flex-1 p-6 overflow-y-auto space-y-4 bg-slate-50">
                  {chatMessages.map((msg, idx) => (
                    <div key={idx} className={`flex gap-3 max-w-3xl ${msg.sender === 'user' ? 'ml-auto flex-row-reverse' : ''}`}>
                      <div className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 ${msg.sender === 'user' ? 'bg-blue-900 text-white' : 'bg-slate-200 text-blue-900 border border-slate-300'}`}>
                        {msg.sender === 'user' ? 'U' : <Bot className="w-4 h-4" />}
                      </div>
                      <div className="space-y-2 max-w-full">
                        <div className={`p-4 rounded-2xl text-sm leading-relaxed ${msg.sender === 'user' ? 'bg-blue-900 text-white rounded-tr-none shadow-xs' : 'bg-white border border-slate-300 text-slate-900 rounded-tl-none font-mono text-xs whitespace-pre-wrap shadow-xs'}`}>
                          {msg.text}
                        </div>
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="bg-slate-100 border border-slate-300 rounded-xl p-3 space-y-2">
                            <div className="text-[11px] font-mono font-bold text-rose-900 flex items-center gap-1.5">
                              <Database className="w-3.5 h-3.5" />
                              <span>Retrieved Hybrid Context Citations ({msg.citations.length})</span>
                            </div>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                              {msg.citations.map((cit, cIdx) => (
                                <div key={`${cit.docId}-${cit.chunkIndex}`} className="bg-white p-2.5 rounded-lg border border-slate-300 text-[11px] space-y-1">
                                  <div className="flex items-center justify-between gap-2 font-mono">
                                    <span className="font-bold text-blue-900 truncate">[Citation {cIdx + 1}] {cit.docTitle}</span>
                                    <span className="text-emerald-900 font-semibold shrink-0">score={cit.score}</span>
                                  </div>
                                  <div className="text-[10px] font-mono text-slate-500 flex flex-wrap items-center gap-1.5">
                                    <span>Chunk #{cit.chunkIndex}</span>
                                    <span>·</span>
                                    <span>Dense={cit.denseScore ?? cit.score}</span>
                                    <span>·</span>
                                    <span>BM25={cit.bm25Score ?? 0}</span>
                                  </div>
                                  <p className="text-slate-700 line-clamp-3 leading-snug">{cit.snippet}</p>
                                  {cit.matchedTerms && cit.matchedTerms.length > 0 && (
                                    <div className="flex flex-wrap gap-1 pt-0.5">
                                      {cit.matchedTerms.slice(0, 4).map((t) => (
                                        <span key={t} className="text-[9px] font-mono px-1 rounded bg-amber-100 text-amber-900">
                                          {t}
                                        </span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {isChatLoading && (
                    <div className="flex gap-3 max-w-md">
                      <div className="w-8 h-8 rounded-xl bg-slate-200 text-blue-900 border border-slate-300 flex items-center justify-center shrink-0">
                        <Bot className="w-4 h-4" />
                      </div>
                      <div className="p-4 rounded-2xl bg-white border border-slate-300 text-slate-600 text-xs flex items-center gap-2 shadow-xs">
                        <RefreshCw className="w-3.5 h-3.5 animate-spin text-blue-900" />
                        <span>Searching FAISS IndexFlatIP &amp; synthesizing grounded answer...</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Chat Input */}
                <form onSubmit={handleSendMessage} className="p-4 bg-slate-200/90 border-t border-slate-300 flex gap-3">
                  <input 
                    type="text" 
                    value={chatInput}
                    onChange={e => setChatInput(e.target.value)}
                    placeholder="Ask about model loss functions, Rust bindings, or evaluation metrics..."
                    className="flex-1 bg-white border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 placeholder:text-slate-500 focus:outline-none focus:border-blue-900 shadow-xs"
                  />
                  <button 
                    type="submit"
                    disabled={isChatLoading || !chatInput.trim()}
                    className="px-6 py-2.5 bg-blue-900 hover:bg-blue-800 disabled:opacity-50 text-white font-semibold rounded-xl text-sm transition-colors shadow-xs"
                  >
                    Send
                  </button>
                </form>
              </div>
            </div>
          )}

          {/* TAB 5: MLOPS ROADMAP */}
          {activeTab === 'roadmap' && (
            <div className="space-y-8">
              <div>
                <div className="flex items-center gap-2 text-xs font-mono text-rose-900 mb-1 font-semibold">
                  <span className="bg-rose-100 px-2 py-0.5 rounded text-rose-900 font-bold border border-rose-200">[Production Pipeline]</span>
                  <span>Google Scholar & Enterprise Deployment Checklist</span>
                </div>
                <h2 className="text-2xl font-bold text-slate-900 tracking-tight">End-to-End MLOps Execution Roadmap</h2>
                <p className="text-slate-600 text-sm mt-1">Track your progress from initial problem scoping through rigorous model evaluation to production deployment and monitoring.</p>
              </div>

              <div className="space-y-4">
                {checklist.map(item => (
                  <div 
                    key={item.id}
                    onClick={() => {
                      setChecklist(checklist.map(c => c.id === item.id ? { ...c, done: !c.done } : c));
                    }}
                    className={`cursor-pointer rounded-2xl bg-slate-100 border p-6 transition-all duration-200 flex items-start gap-4 hover:bg-slate-200 hover:shadow-md ${item.done ? 'border-emerald-300 bg-emerald-50/50' : 'border-slate-300'}`}
                  >
                    <div className={`w-6 h-6 rounded-lg mt-0.5 flex items-center justify-center shrink-0 border ${item.done ? 'bg-emerald-600 border-emerald-600 text-white' : 'border-slate-400 bg-white text-transparent'}`}>
                      <CheckCircle2 className="w-4 h-4" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h3 className={`text-base font-bold ${item.done ? 'text-slate-900 line-through opacity-80' : 'text-slate-900'}`}>{item.phase}</h3>
                        <span className={`text-xs px-2.5 py-0.5 rounded-full font-semibold border ${item.done ? 'bg-emerald-100 text-emerald-900 border-emerald-300' : 'bg-amber-100 text-amber-900 border-amber-300'}`}>
                          {item.done ? 'Completed' : 'Pending'}
                        </span>
                      </div>
                      <p className="text-slate-700 text-sm leading-relaxed">{item.desc}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
          {/* TAB: TECHNICAL WHITE PAPERS & ENGINEERING BLOG */}
          {activeTab === 'whitepapers' && (
            <WhitePaperBlogTab
              flagshipProjects={researchProjects}
              datacampProjects={nonStarredRepoProjects}
              onOpenProjectInRepo={(proj) => {
                setCodeRepoTargetProjectId(proj.id);
                setActiveTab('coderepo');
              }}
            />
          )}

          {/* TAB: ARCHIVES HUB */}
          {activeTab === 'archives' && (
            <div className="space-y-8">
              <div>
                <div className="flex items-center gap-2 text-xs font-mono text-slate-700 mb-1 font-semibold">
                  <span className="bg-slate-200 px-2 py-0.5 rounded text-slate-900 font-bold border border-slate-300">[Archives]</span>
                  <span>Archived Studio Utilities & Assistant Modules</span>
                </div>
                <h2 className="text-2xl font-bold text-slate-900 tracking-tight">Workspace Archives</h2>
                <p className="text-slate-600 text-sm mt-1">
                  Access archived modules including the Project Builder Wizard, RAG-grounded AI ML Copilot, and MLOps Execution Roadmap.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <button
                  onClick={() => setActiveTab('builder')}
                  className="text-left rounded-2xl bg-slate-100 hover:bg-slate-200 border border-slate-300 p-6 transition-all duration-200 hover:shadow-md flex flex-col justify-between gap-4"
                >
                  <div>
                    <div className="w-10 h-10 rounded-xl bg-white border border-slate-300 flex items-center justify-center mb-4">
                      <Wand2 className="w-5 h-5 text-blue-900" />
                    </div>
                    <div className="text-xs font-mono font-bold text-blue-900 mb-1">[01 · Architecture Generator]</div>
                    <h3 className="text-lg font-bold text-slate-900 mb-2">Project Builder Wizard</h3>
                    <p className="text-sm text-slate-700 leading-relaxed">
                      Synthesize custom AI/ML system architectures, target metrics, and end-to-end training & deployment blueprints.
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs font-bold text-blue-900">
                    <span>Open Project Builder</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </div>
                </button>

                <button
                  onClick={() => setActiveTab('copilot')}
                  className="text-left rounded-2xl bg-slate-100 hover:bg-slate-200 border border-slate-300 p-6 transition-all duration-200 hover:shadow-md flex flex-col justify-between gap-4"
                >
                  <div>
                    <div className="w-10 h-10 rounded-xl bg-white border border-slate-300 flex items-center justify-center mb-4">
                      <MessageSquareCode className="w-5 h-5 text-emerald-900" />
                    </div>
                    <div className="text-xs font-mono font-bold text-emerald-900 mb-1">[02 · FAISS RAG Assistant]</div>
                    <h3 className="text-lg font-bold text-slate-900 mb-2">AI ML Copilot</h3>
                    <p className="text-sm text-slate-700 leading-relaxed">
                      Query ingested PDFs and technical documents indexed in FAISS with similarity-scored citations.
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-900">
                    <span>Open AI ML Copilot</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </div>
                </button>

                <button
                  onClick={() => setActiveTab('roadmap')}
                  className="text-left rounded-2xl bg-slate-100 hover:bg-slate-200 border border-slate-300 p-6 transition-all duration-200 hover:shadow-md flex flex-col justify-between gap-4"
                >
                  <div>
                    <div className="w-10 h-10 rounded-xl bg-white border border-slate-300 flex items-center justify-center mb-4">
                      <Workflow className="w-5 h-5 text-rose-900" />
                    </div>
                    <div className="text-xs font-mono font-bold text-rose-900 mb-1">[03 · Production Checklist]</div>
                    <h3 className="text-lg font-bold text-slate-900 mb-2">MLOps Roadmap</h3>
                    <p className="text-sm text-slate-700 leading-relaxed">
                      Track 8 lifecycle stages from problem scoping and EDA to ONNX serialization and production drift monitoring.
                    </p>
                  </div>
                  <div className="flex items-center gap-1.5 text-xs font-bold text-rose-900">
                    <span>Open MLOps Roadmap</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </div>
                </button>
              </div>

              {/* UIDAI Biometric Security & DPI Architecture Specification */}
              <div className="mt-10 bg-white border border-slate-300 rounded-2xl p-6 md:p-8 shadow-sm space-y-6">
                <div className="flex items-center justify-between border-b border-slate-200 pb-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-slate-100 border border-slate-300 flex items-center justify-center text-slate-900">
                      <ShieldCheck className="w-6 h-6 text-blue-900" />
                    </div>
                    <div>
                      <div className="text-xs font-mono font-bold text-blue-900 uppercase">UIDAI National DPI Specification</div>
                      <h3 className="text-xl font-bold text-slate-900">Scholarly Architecture & Biometric Verification Standard</h3>
                    </div>
                  </div>
                  <span className="px-3 py-1 bg-slate-100 text-slate-800 border border-slate-300 rounded-lg text-xs font-mono font-semibold">
                    CIDR Specification · 5-Pillar Standard
                  </span>
                </div>

                <p className="text-sm text-slate-700 leading-relaxed">
                  Formal engineering specification governing how external authentication and payment systems interact with UIDAI infrastructure under strict authoritative, cryptographic, biometric, privacy, and SLA constraints:
                </p>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                  {/* Pillar 1 */}
                  <div className="bg-slate-50 border border-slate-300 rounded-xl p-5 space-y-3 flex flex-col justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs font-mono text-slate-600">
                        <span className="font-bold text-blue-900">PILLAR 01</span>
                        <span className="bg-blue-50 text-blue-900 border border-blue-200 px-2 py-0.5 rounded font-semibold">Authoritative</span>
                      </div>
                      <h4 className="font-bold text-slate-900 text-sm">Central Identities Data Repository (CIDR)</h4>
                      <p className="text-xs text-slate-700 leading-relaxed">
                        CIDR serves as the single immutable root of truth for demographic and biometric records. All edge captures originate from cryptographically certified L0/L1 Registered Devices.
                      </p>
                    </div>
                    <div className="pt-2 border-t border-slate-200 flex flex-wrap gap-1">
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#CIDR</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#L1-RegisteredDevice</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#RootOfTrust</span>
                    </div>
                  </div>

                  {/* Pillar 2 */}
                  <div className="bg-slate-50 border border-slate-300 rounded-xl p-5 space-y-3 flex flex-col justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs font-mono text-slate-600">
                        <span className="font-bold text-rose-900">PILLAR 02</span>
                        <span className="bg-rose-50 text-rose-900 border border-rose-200 px-2 py-0.5 rounded font-semibold">Data in Transit</span>
                      </div>
                      <h4 className="font-bold text-slate-900 text-sm">Hardware Enclaves &amp; Post-Quantum mTLS 1.3</h4>
                      <p className="text-xs text-slate-700 leading-relaxed">
                        PID payloads are envelope-encrypted at edge capture devices via Hybrid ML-KEM-1024 + AES-256-GCM over mTLS 1.3 / WireGuard 0-RTT tunnels and unwrapped exclusively inside FIPS 140-3 Level 3 HSMs &amp; AMD SEV-SNP enclaves.
                      </p>
                    </div>
                    <div className="pt-2 border-t border-slate-200 flex flex-wrap gap-1">
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#ML-KEM-1024</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#mTLS-1.3</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#FIPS-140-3</span>
                    </div>
                  </div>

                  {/* Pillar 3 */}
                  <div className="bg-slate-50 border border-slate-300 rounded-xl p-5 space-y-3 flex flex-col justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs font-mono text-slate-600">
                        <span className="font-bold text-emerald-900">PILLAR 03</span>
                        <span className="bg-emerald-50 text-emerald-900 border border-emerald-200 px-2 py-0.5 rounded font-semibold">Accurate</span>
                      </div>
                      <h4 className="font-bold text-slate-900 text-sm">Minutiae &amp; Normalized IrisCode Matching</h4>
                      <p className="text-xs text-slate-700 leading-relaxed">
                        Fuzzy biometric matching evaluates fingerprint minutiae constellation vectors and Normalized IrisCode Hamming distance (&lt; 0.32 threshold) across 1.50B–2.41B galleries to suppress false error rates.
                      </p>
                    </div>
                    <div className="pt-2 border-t border-slate-200 flex flex-wrap gap-1">
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#HammingDistance</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#MinutiaeMatching</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#ISO-30107-PAD</span>
                    </div>
                  </div>

                  {/* Pillar 4 */}
                  <div className="bg-slate-50 border border-slate-300 rounded-xl p-5 space-y-3 flex flex-col justify-between">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs font-mono text-slate-600">
                        <span className="font-bold text-indigo-900">PILLAR 04</span>
                        <span className="bg-indigo-50 text-indigo-900 border border-indigo-200 px-2 py-0.5 rounded font-semibold">Data at Rest</span>
                      </div>
                      <h4 className="font-bold text-slate-900 text-sm">VID Tokenization &amp; AES-256-XTS Data Vaults</h4>
                      <p className="text-xs text-slate-700 leading-relaxed">
                        Raw IDs are abstracted via NIST FF3-1 Format-Preserving &amp; 16-digit VID tokens. Shards at rest are encrypted using AES-256-XTS / AES-256-GCM-SIV with Argon2id key derivation and 90-day KMS rotation over the 20-year lifecycle.
                      </p>
                    </div>
                    <div className="pt-2 border-t border-slate-200 flex flex-wrap gap-1">
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#AES-256-XTS</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#FF3-1-FPE</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#DataVault</span>
                    </div>
                  </div>

                  {/* Pillar 5 */}
                  <div className="bg-slate-50 border border-slate-300 rounded-xl p-5 space-y-3 flex flex-col justify-between md:col-span-2 lg:col-span-2">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-xs font-mono text-slate-600">
                        <span className="font-bold text-amber-900">PILLAR 05</span>
                        <span className="bg-amber-50 text-amber-900 border border-amber-200 px-2 py-0.5 rounded font-semibold">Timely / SLA</span>
                      </div>
                      <h4 className="font-bold text-slate-900 text-sm">Sub-200ms In-Memory Matching & Asynchronous Routing</h4>
                      <p className="text-xs text-slate-700 leading-relaxed">
                        CIDR achieves sub-200ms end-to-end latency SLAs through distributed lock-free Rust/C++ indexing, non-blocking asynchronous gateway routing, and circuit breakers for AEPS micro-ATM rails.
                      </p>
                    </div>
                    <div className="pt-2 border-t border-slate-200 flex flex-wrap gap-1">
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#Sub200msSLA</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#LockFreeRust</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#AsyncRouting</span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white border border-slate-300 text-slate-700">#AEPS-Rails</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* TAB: FULL CODE REPO & DIRECTORY STRUCTURE */}
          {activeTab === 'coderepo' && (
            <CodeRepoStructureTab
              flagshipProjects={researchProjects}
              datacampProjects={nonStarredRepoProjects}
              initialProjectId={codeRepoTargetProjectId}
            />
          )}

          {/* TAB: LEARNING (YOUTUBE DATA SCIENCE, AI & ML CURRICULUM) */}
          {activeTab === 'learning' && <LearningSectionTab />}
        </main>
        {/* Footer */}
        <footer className="border-t border-slate-200 py-6 px-8 text-center text-xs text-slate-600 bg-white shadow-inner">
          AI/ML Project Architect & Studio
        </footer>
      </div>

      {/* TECHNICAL WHITE PAPER MODAL (GOOGLE SCHOLAR SPECIFICATION STANDARD) */}
      {whitePaperModalProject && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
          <div className="max-w-5xl w-full max-h-[92vh] overflow-y-auto relative shadow-2xl">
            <GoogleScholarWhitePaper
              project={whitePaperModalProject}
              onOpenProjectInRepo={(p) => {
                setCodeRepoTargetProjectId(p.id);
                setWhitePaperModalProject(null);
                setActiveTab('coderepo');
              }}
              onClose={() => setWhitePaperModalProject(null)}
              isModal={true}
            />
          </div>
        </div>
      )}

      {/* BUILDER GENERATED WHITE PAPER MODAL (GOOGLE SCHOLAR SPECIFICATION STANDARD) */}
      {showBuilderWhitePaper && generatedBlueprint && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 overflow-y-auto">
          <div className="max-w-5xl w-full max-h-[92vh] overflow-y-auto relative shadow-2xl">
            <GoogleScholarWhitePaper
              project={{
                id: `blueprint-${builderForm.projectName.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'custom-model'}`,
                title: builderForm.projectName || 'Autonomous Deep Learning Pipeline',
                domain: builderForm.domain,
                difficulty: 'Research Grade',
                description: generatedBlueprint.overview,
                objective: builderForm.objective,
                metrics: builderForm.targetMetric,
                trainingLoss: '0.0384',
                trainingAccuracy: '98.7%',
                problem: generatedBlueprint.whitePaper?.problem || `Operational bottlenecks and classical model failure modes in ${builderForm.domain}.`,
                breakingGround: generatedBlueprint.whitePaper?.breakingGround || `Novel architecture addressing ${builderForm.objective}.`,
                epochHistory: [
                  { epoch: 1, loss: 0.612, accuracy: 76.5, valAccuracy: 74.8 },
                  { epoch: 5, loss: 0.138, accuracy: 95.2, valAccuracy: 94.3 },
                  { epoch: 10, loss: 0.0384, accuracy: 98.7, valAccuracy: 98.2 }
                ],
                sampleDistribution: [
                  { label: 'Validated Class A', percentage: 52, count: '52,000' },
                  { label: 'Calibrated Class B', percentage: 48, count: '48,000' }
                ],
                performanceIndicators: {
                  latency: 'P95 2.4ms / P99 4.1ms',
                  throughput: '142,000 req/s (16-Shard Cluster)',
                  memoryFootprint: '4.15 GB VRAM / shard (FP8 + PQ-8)',
                  hardwareTarget: builderForm.deploymentTarget
                },
                techStack: generatedBlueprint.recommendedStack && generatedBlueprint.recommendedStack.length > 0 
                  ? generatedBlueprint.recommendedStack 
                  : ['PyTorch', 'FastAPI', 'Docker', 'Triton'],
                architectureFlow: generatedBlueprint.phases && generatedBlueprint.phases.length > 0
                  ? generatedBlueprint.phases.map(p => p.title)
                  : ['Data Preprocessing', 'Tensor Ingestion', 'Forward Propagation', 'Loss & Validation'],
                pipeline: generatedBlueprint.phases && generatedBlueprint.phases.length > 0
                  ? generatedBlueprint.phases.map(p => ({
                      title: p.title,
                      desc: p.description,
                      code: p.pythonSnippet,
                      language: 'python' as const
                    }))
                  : [],
                whitePaper: {
                  problem: generatedBlueprint.whitePaper?.problem || `Current limitations in ${builderForm.domain}.`,
                  breakingGround: generatedBlueprint.whitePaper?.breakingGround || `Novel architecture addressing ${builderForm.objective}.`,
                  scope: generatedBlueprint.whitePaper?.scope || `Full technical scope for ${builderForm.projectName}.`,
                  functionality: generatedBlueprint.whitePaper?.functionality || `Functional components targeting ${builderForm.targetMetric}.`,
                  abstract: generatedBlueprint.whitePaper?.abstract || generatedBlueprint.overview,
                  methodology: generatedBlueprint.whitePaper?.methodology || 'Pipeline designed for optimal data transformation.',
                  architecture: generatedBlueprint.whitePaper?.architecture || 'Modular microservices deployed with high reliability.',
                  benchmarks: generatedBlueprint.whitePaper?.benchmarks || `Target metric: ${builderForm.targetMetric}`,
                  productionConsiderations: generatedBlueprint.whitePaper?.productionConsiderations || `Deployment target: ${builderForm.deploymentTarget}`
                }
              }}
              onClose={() => setShowBuilderWhitePaper(false)}
              isModal={true}
            />
          </div>
        </div>
      )}
    </div>
  );
}
