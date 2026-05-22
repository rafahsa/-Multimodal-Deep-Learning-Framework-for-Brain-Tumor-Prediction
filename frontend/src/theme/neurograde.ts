export const neurograde = {
  colors: {
    bgPrimary: '#060810',
    bgSecondary: '#0c0f1a',
    bgElevated: '#111627',
    bgGlass: 'rgba(12, 15, 26, 0.85)',

    textPrimary: '#e8e6e1',
    textSecondary: 'rgba(232, 230, 225, 0.65)',
    textDim: 'rgba(232, 230, 225, 0.4)',

    accentCyan: '#00d4aa',
    accentCyanDim: '#00a385',
    accentCyanGlow: 'rgba(0, 212, 170, 0.15)',

    accentWarm: '#f0724b',
    accentWarmGlow: 'rgba(240, 114, 75, 0.12)',

    accentViolet: '#8b5cf6',
    accentVioletGlow: 'rgba(139, 92, 246, 0.1)',

    border: 'rgba(232, 230, 225, 0.06)',
    borderActive: 'rgba(0, 212, 170, 0.25)',
  },

  fonts: {
    display: "'Playfair Display', serif",
    body: "'Manrope', sans-serif",
    mono: "'JetBrains Mono', monospace",
  },

  radii: {
    sm: '8px',
    md: '14px',
    lg: '20px',
    xl: '28px',
    pill: '100px',
  },

  glass: {
    background: 'rgba(12, 15, 26, 0.85)',
    backdropFilter: 'blur(16px)',
    border: 'rgba(232, 230, 225, 0.06)',
  },

  shadows: {
    glowCyan: '0 6px 24px rgba(0, 212, 170, 0.15)',
    glowCyanStrong: '0 12px 40px rgba(0, 212, 170, 0.15)',
    elevated: '0 20px 60px rgba(0, 0, 0, 0.4)',
  },

  transitions: {
    default: 'all 0.3s cubic-bezier(0.22, 1, 0.36, 1)',
    fast: 'all 0.2s ease-out',
    slow: 'all 0.5s cubic-bezier(0.22, 1, 0.36, 1)',
  },
} as const;

export type NeuroGradeTheme = typeof neurograde;
