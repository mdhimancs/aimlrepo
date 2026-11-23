export const CHART_FONTS = {
  sans: '"Plus Jakarta Sans", ui-sans-serif, system-ui, sans-serif',
  mono: '"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace',
} as const;

export const CHART_COLORS = {
  // Structural canvas & axes
  stageFloor: '#f1f5f9',
  stageStroke: '#cbd5e1',
  backPlane: '#f8fafc',
  gridLine: '#e2e8f0',
  tickText: '#334155',
  mutedText: '#475569',
  inkText: '#0f172a',
  haloStroke: '#ffffff',

  // Coordinate axis triads (High contrast WCAG AA)
  axisX: '#be123c',
  axisXLabel: '#9f1239',
  axisY: '#047857',
  axisYLabel: '#065f46',
  axisZ: '#1d4ed8',
  axisZLabel: '#1e3a8a',

  // Series & 3D bar facets
  emeraldFront: '#059669',
  emeraldTop: '#34d399',
  emeraldSide: '#065f46',

  indigoFront: '#4f46e5',
  indigoTop: '#818cf8',
  indigoSide: '#312e81',

  sapphireFront: '#1d4ed8',
  sapphireTop: '#60a5fa',
  sapphireSide: '#1e3a8a',

  crimsonLine: '#be123c',
  amberLine: '#d97706',
  slateMutedFront: '#94a3b8',
  slateMutedTop: '#cbd5e1',
  slateMutedSide: '#64748b',
} as const;
