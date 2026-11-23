import {
  CapExBreakdownItem,
  FlagshipProject,
  OpExBreakdownItem,
  OpExCostKey,
  ProjectEconomics,
  YearlyOpExRow
} from '../types';

const USD_TO_INR = 83.5;

function hashProjectId(id: string): number {
  let hash = 2166136261;
  for (let i = 0; i < id.length; i++) {
    hash ^= id.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

export function formatUsdOnly(amountUsd: number): string {
  const sign = amountUsd < 0 ? '-' : '';
  const abs = Math.abs(amountUsd);
  if (abs >= 1_000_000_000) {
    const b = abs / 1_000_000_000;
    return `${sign}$${Number(b.toFixed(b >= 10 ? 1 : 2))} B`;
  }
  if (abs >= 1_000_000) {
    const m = abs / 1_000_000;
    return `${sign}$${Number(m.toFixed(m >= 10 ? 1 : 2))} M`;
  }
  if (abs >= 1_000) {
    const k = abs / 1_000;
    return `${sign}$${Number(k.toFixed(1))} K`;
  }
  return `${sign}$${Math.round(abs).toLocaleString('en-US')}`;
}

export function formatRsOnly(amountUsd: number): string {
  const inr = amountUsd * USD_TO_INR;
  const sign = inr < 0 ? '-' : '';
  const abs = Math.abs(inr);
  if (abs >= 10_000_000) {
    const cr = abs / 10_000_000;
    const formattedCr = cr >= 100 ? Math.round(cr).toLocaleString('en-US') : Number(cr.toFixed(1)).toString();
    return `${sign}Rs ${formattedCr} Cr`;
  }
  if (abs >= 100_000) {
    const l = abs / 100_000;
    return `${sign}Rs ${Number(l.toFixed(1))} L`;
  }
  if (abs >= 1_000) {
    const k = abs / 1_000;
    return `${sign}Rs ${Number(k.toFixed(1))} K`;
  }
  return `${sign}Rs ${Math.round(abs).toLocaleString('en-US')}`;
}

export function formatDualCurrency(amountUsd: number): string {
  return `${formatUsdOnly(amountUsd)} / ${formatRsOnly(amountUsd)}`;
}

export function formatInrCompact(amountUsd: number): string {
  return formatDualCurrency(amountUsd);
}

export function formatInrExact(amountUsd: number): string {
  return formatDualCurrency(amountUsd);
}

export const formatUsdCompact = formatUsdOnly;
export const formatUsdExact = formatDualCurrency;

export function computeProjectEconomics(project: FlagshipProject): ProjectEconomics {
  if (project.economics) {
    return project.economics;
  }

  const h1 = hashProjectId(project.id);
  const h2 = hashProjectId(`${project.id}-opex`);
  const textBlob =
    `${project.id} ${project.title} ${project.domain} ${project.description} ${project.performanceIndicators.hardwareTarget} ${project.performanceIndicators.memoryFootprint} ${project.techStack.join(' ')}`.toLowerCase();

  const isWorldScale =
    project.id.startsWith('uidai-') ||
    project.id.startsWith('world-') ||
    /1\.50b|1\.38b|cidr|cbdc|iso-20022|navic|smart grid|cold-chain/.test(textBlob);

  const isGpuCluster =
    /h100|a100|rtx 4090|fpga|a10g|jetson|gpu|llm|diffusion|transformer|gnn/.test(textBlob);
  const hasRustFoss =
    project.techStack.some((t) => /rust|arrow|tvm|openvino|lmax|io_uring/i.test(t)) ||
    project.pipeline.some((p) => p.language === 'rust');
  const isCyberSec = /cyber|ids|malware|apt|vulnerability|red-team|fraud|aml|crime/.test(textBlob);

  let deploymentTier: ProjectEconomics['deploymentTier'] = 'Lean Edge / Serverless';
  let baseCapEx = 22_000;
  let baseOpExY1 = 36_000;
  let basePowerKw = 1.8;
  let fteHeadcount = 1.5;
  let yoyCohortGrowth = 0.018;
  let unitCostSummary = '$0.0002 / 1K predictions (Rs 0.015 · CPU serverless / edge)';

  if (isWorldScale) {
    deploymentTier = 'Sovereign / World-Scale (1.50B–2.41B)';
    baseCapEx = 18_500_000 + Math.round(h1 * 14_500_000);
    baseOpExY1 = 6_400_000 + Math.round(h2 * 4_800_000);
    basePowerKw = 1850 + Math.round(h1 * 950);
    fteHeadcount = 26 + Math.round(h2 * 16);
    yoyCohortGrowth = 0.024; // 1.50B -> 2.41B 20-year delta
    unitCostSummary = '$0.0014 – $0.0042 / 1K auth tx (Rs 0.12 – Rs 0.35 · 1.50B–2.41B scale)';
  } else if (project.difficulty === 'Research Grade') {
    deploymentTier = 'Enterprise / Research Cluster';
    const heavyAccelMult = /h100|a100|cluster|fpga/.test(textBlob) ? 1.55 : 0.92;
    baseCapEx = Math.round((640_000 + h1 * 1_450_000) * heavyAccelMult);
    baseOpExY1 = Math.round((460_000 + h2 * 620_000) * (isGpuCluster ? 1.18 : 0.9));
    basePowerKw = Math.round((48 + h1 * 96) * heavyAccelMult * 10) / 10;
    fteHeadcount = 5 + Math.round(h2 * 6);
    yoyCohortGrowth = 0.026;
    unitCostSummary = isGpuCluster
      ? '$0.019 / 1K accelerated tensor inferences (Rs 1.58 · bare-metal GPU)'
      : '$0.0042 / 1K deterministic inferences (Rs 0.35)';
  } else if (project.difficulty === 'Advanced' || project.difficulty === 'Intermediate') {
    deploymentTier = 'Production Multi-Node';
    const gpuMult = /t4|l4|gpu|yolo|xray/.test(textBlob) ? 1.38 : 0.85;
    baseCapEx = Math.round((68_000 + h1 * 115_000) * gpuMult);
    baseOpExY1 = Math.round((84_000 + h2 * 96_000) * gpuMult);
    basePowerKw = Math.round((6.5 + h1 * 9.5) * gpuMult * 10) / 10;
    fteHeadcount = 2 + Math.round(h2 * 2);
    yoyCohortGrowth = 0.021;
    unitCostSummary = '$0.0029 / 1K production API inferences (Rs 0.24)';
  } else {
    deploymentTier = 'Lean Edge / Serverless';
    baseCapEx = Math.round(14_500 + h1 * 19_500);
    baseOpExY1 = Math.round(24_000 + h2 * 26_000);
    basePowerKw = Math.round((1.2 + h1 * 1.9) * 10) / 10;
    fteHeadcount = 1;
    yoyCohortGrowth = 0.016;
    unitCostSummary = '$0.00014 / 1K CPU microservice inferences (Rs 0.012)';
  }

  // Open-source Rust / Zero-copy runtime efficiency saves ~12% Infra & MLOps license bloat
  const rustEfficiencyFactor = hasRustFoss ? 0.88 : 1.0;
  const totalCapExUsd = Math.round(baseCapEx * (hasRustFoss ? 0.93 : 1.0));

  // CapEx Breakdown (sums strictly to totalCapExUsd)
  const capComputeShare = isGpuCluster || isWorldScale ? 0.48 : 0.42;
  const capStorageShare = isWorldScale ? 0.22 : 0.18;
  const capSecurityShare = isWorldScale || isCyberSec ? 0.18 : 0.16;
  const capCompute = Math.round(totalCapExUsd * capComputeShare);
  const capStorage = Math.round(totalCapExUsd * capStorageShare);
  const capSecurity = Math.round(totalCapExUsd * capSecurityShare);
  const capDeploy = Math.max(0, totalCapExUsd - capCompute - capStorage - capSecurity);

  const capExBreakdown: CapExBreakdownItem[] = [
    {
      label: isGpuCluster
        ? 'Compute & GPU/Accelerator Cluster Hardware'
        : 'Compute Nodes, Inference Servers & Edge Appliances',
      amountUsd: capCompute,
      sharePct: Number(((capCompute / totalCapExUsd) * 100).toFixed(1)),
      notes: project.performanceIndicators.hardwareTarget
    },
    {
      label: 'High-Throughput NVMe, Storage Fabric & Interconnects',
      amountUsd: capStorage,
      sharePct: Number(((capStorage / totalCapExUsd) * 100).toFixed(1)),
      notes: `${project.performanceIndicators.memoryFootprint} · Zero-copy Arrow/Parquet tiers`
    },
    {
      label: 'HSM, PQC Cryptography & Zero-Trust Network Enclaves',
      amountUsd: capSecurity,
      sharePct: Number(((capSecurity / totalCapExUsd) * 100).toFixed(1)),
      notes: 'ML-KEM-1024 mTLS 1.3 + AES-256-XTS Vault & Firewalling'
    },
    {
      label: 'Datacenter Fit-Out, Power Distribution & Initial Integration',
      amountUsd: capDeploy,
      sharePct: Number(((capDeploy / totalCapExUsd) * 100).toFixed(1)),
      notes: 'Cooling racks, PDUs, CI/CD staging & baseline benchmarking'
    }
  ];

  // Raw weights for the 6 OpEx categories tailored to domain
  const rawWeights: Record<OpExCostKey, number> = {
    infra: (isGpuCluster ? 0.28 : 0.25) * rustEfficiencyFactor,
    aiOps: isCyberSec || isWorldScale ? 0.13 : 0.10,
    mlOps: (isGpuCluster ? 0.17 : 0.15) * rustEfficiencyFactor,
    energy: isGpuCluster ? 0.14 : isWorldScale ? 0.12 : 0.08,
    manpower: isWorldScale ? 0.22 : 0.29,
    maintenance: isWorldScale ? 0.11 : 0.10
  };
  const weightSum = Object.values(rawWeights).reduce((acc, v) => acc + v, 0);
  const normWeights: Record<OpExCostKey, number> = {
    infra: rawWeights.infra / weightSum,
    aiOps: rawWeights.aiOps / weightSum,
    mlOps: rawWeights.mlOps / weightSum,
    energy: rawWeights.energy / weightSum,
    manpower: rawWeights.manpower / weightSum,
    maintenance: rawWeights.maintenance / weightSum
  };

  // Compute Year-1 category costs (adjusted so power-driven energy stays physically coherent)
  const pue = isWorldScale ? 1.24 : isGpuCluster ? 1.28 : 1.32;
  const annualEnergyMwhY1 = Number(((basePowerKw * pue * 8760) / 1000).toFixed(1));
  const y1Infra = Math.round(baseOpExY1 * normWeights.infra);
  const y1AiOps = Math.round(baseOpExY1 * normWeights.aiOps);
  const y1MlOps = Math.round(baseOpExY1 * normWeights.mlOps);
  const y1Energy = Math.round(baseOpExY1 * normWeights.energy);
  const y1Manpower = Math.round(baseOpExY1 * normWeights.manpower);
  const y1Maintenance = Math.round(baseOpExY1 * normWeights.maintenance);

  // Category-specific annual YoY escalation curves over 20 years
  const rates: Record<OpExCostKey, number> = {
    infra: yoyCohortGrowth + 0.008,
    aiOps: yoyCohortGrowth + 0.005,
    mlOps: yoyCohortGrowth + 0.007,
    energy: yoyCohortGrowth + 0.014,
    manpower: 0.036,
    maintenance: 0.032
  };

  const yearlySchedule: YearlyOpExRow[] = [];
  let cumulativeOpEx = 0;

  const totals20Yr: Record<OpExCostKey, number> = {
    infra: 0,
    aiOps: 0,
    mlOps: 0,
    energy: 0,
    manpower: 0,
    maintenance: 0
  };

  for (let yr = 1; yr <= 20; yr++) {
    const t = yr - 1;
    const scaleFactor = Number(Math.pow(1 + yoyCohortGrowth, t).toFixed(3));
    const infra = Math.round(y1Infra * Math.pow(1 + rates.infra, t));
    const aiOps = Math.round(y1AiOps * Math.pow(1 + rates.aiOps, t));
    const mlOps = Math.round(y1MlOps * Math.pow(1 + rates.mlOps, t));
    const energy = Math.round(y1Energy * Math.pow(1 + rates.energy, t));
    const manpower = Math.round(y1Manpower * Math.pow(1 + rates.manpower, t));
    const maintenance = Math.round(y1Maintenance * Math.pow(1 + rates.maintenance, t));

    const totalOpEx = infra + aiOps + mlOps + energy + manpower + maintenance;
    const prevOpEx = yearlySchedule.length > 0 ? yearlySchedule[yearlySchedule.length - 1].totalOpEx : totalOpEx;
    const yoyDeltaPct =
      yr === 1 ? 0 : Number((((totalOpEx - prevOpEx) / Math.max(1, prevOpEx)) * 100).toFixed(2));

    cumulativeOpEx += totalOpEx;

    totals20Yr.infra += infra;
    totals20Yr.aiOps += aiOps;
    totals20Yr.mlOps += mlOps;
    totals20Yr.energy += energy;
    totals20Yr.manpower += manpower;
    totals20Yr.maintenance += maintenance;

    yearlySchedule.push({
      year: yr,
      calendarYear: 2025 + yr,
      scaleFactor,
      infra,
      aiOps,
      mlOps,
      energy,
      manpower,
      maintenance,
      totalOpEx,
      yoyDeltaPct,
      cumulativeOpEx,
      cumulativeTco: totalCapExUsd + cumulativeOpEx
    });
  }

  const firstYear = yearlySchedule[0];
  const year10 = yearlySchedule[9];
  const year20 = yearlySchedule[19];
  const cagrPct = Number(
    ((Math.pow(year20.totalOpEx / Math.max(1, firstYear.totalOpEx), 1 / 19) - 1) * 100).toFixed(2)
  );

  const opExCategories: OpExBreakdownItem[] = [
    {
      key: 'infra',
      label: 'Infrastructure Cost',
      shortLabel: 'Infra Cost',
      description: 'Compute clusters, NVMe/WORM storage tiers, RDMA/network bandwidth & colocation',
      year1Usd: firstYear.infra,
      year20Usd: year20.infra,
      total20YrUsd: totals20Yr.infra,
      sharePct: Number(((totals20Yr.infra / cumulativeOpEx) * 100).toFixed(1))
    },
    {
      key: 'aiOps',
      label: 'AIOps Cost',
      shortLabel: 'AIOps Cost',
      description: 'Automated anomaly triage, OpenTelemetry/Prometheus APM & zero-touch incident response',
      year1Usd: firstYear.aiOps,
      year20Usd: year20.aiOps,
      total20YrUsd: totals20Yr.aiOps,
      sharePct: Number(((totals20Yr.aiOps / cumulativeOpEx) * 100).toFixed(1))
    },
    {
      key: 'mlOps',
      label: 'MLOps Cost',
      shortLabel: 'MLOps Cost',
      description: 'Continuous retraining, feature store sync, drift alarms, shadow CI/CD & model governance',
      year1Usd: firstYear.mlOps,
      year20Usd: year20.mlOps,
      total20YrUsd: totals20Yr.mlOps,
      sharePct: Number(((totals20Yr.mlOps / cumulativeOpEx) * 100).toFixed(1))
    },
    {
      key: 'energy',
      label: 'Energy & Cooling Cost',
      shortLabel: 'Energy Cost',
      description: `${basePowerKw} kW base draw (PUE ${pue}) · ${annualEnergyMwhY1} MWh/yr initial power & HVAC`,
      year1Usd: firstYear.energy,
      year20Usd: year20.energy,
      total20YrUsd: totals20Yr.energy,
      sharePct: Number(((totals20Yr.energy / cumulativeOpEx) * 100).toFixed(1))
    },
    {
      key: 'manpower',
      label: 'Manpower & Engineering Cost',
      shortLabel: 'Manpower Cost',
      description: `${fteHeadcount} dedicated FTEs (ML researchers, Rust/systems engineers, SREs & SecOps)`,
      year1Usd: firstYear.manpower,
      year20Usd: year20.manpower,
      total20YrUsd: totals20Yr.manpower,
      sharePct: Number(((totals20Yr.manpower / cumulativeOpEx) * 100).toFixed(1))
    },
    {
      key: 'maintenance',
      label: 'Maintenance & Refresh Cost',
      shortLabel: 'Maintenance Cost',
      description: '4-yr hardware RMA/lifecycle refresh reserve, HSM/PQC key rotation & kernel SLA patching',
      year1Usd: firstYear.maintenance,
      year20Usd: year20.maintenance,
      total20YrUsd: totals20Yr.maintenance,
      sharePct: Number(((totals20Yr.maintenance / cumulativeOpEx) * 100).toFixed(1))
    }
  ];

  return {
    deploymentTier,
    totalCapExUsd,
    capExBreakdown,
    opExYear1Usd: firstYear.totalOpEx,
    opExYear10Usd: year10.totalOpEx,
    opExYear20Usd: year20.totalOpEx,
    opExYoYGrowthPct: cagrPct,
    totalOpEx20YrUsd: cumulativeOpEx,
    totalTco20YrUsd: totalCapExUsd + cumulativeOpEx,
    estimatedPowerKw: basePowerKw,
    annualEnergyMwhY1,
    fteHeadcount,
    opExCategories,
    yearlySchedule,
    unitCostSummary
  };
}
