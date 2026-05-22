import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';
import type { BackendHealthState } from '../../hooks/useBackendHealth';

interface BackendStatusBannerProps {
  state: BackendHealthState;
}

const banner: CSSProperties = {
  position: 'fixed',
  top: '4.25rem',
  left: '50%',
  transform: 'translateX(-50%)',
  zIndex: 999,
  display: 'flex',
  alignItems: 'center',
  gap: '0.65rem',
  padding: '0.65rem 1.25rem',
  borderRadius: ng.radii.pill,
  border: `1px solid ${ng.colors.borderActive}`,
  background: ng.colors.bgGlass,
  backdropFilter: ng.glass.backdropFilter,
  boxShadow: ng.shadows.glowCyan,
  fontFamily: ng.fonts.mono,
  fontSize: '0.78rem',
  color: ng.colors.textPrimary,
  maxWidth: 'min(92vw, 520px)',
};

export function BackendStatusBanner({ state }: BackendStatusBannerProps) {
  if (state === 'healthy') return null;

  const message =
    state === 'connecting'
      ? 'Backend connecting… loading models may take up to 60s'
      : 'Cannot reach prediction server — retrying every 10s';

  return (
    <div style={banner} role="status" aria-live="polite">
      <svg
        width="14"
        height="14"
        viewBox="0 0 14 14"
        fill="none"
        style={{ animation: 'neuro-spin 1s linear infinite', flexShrink: 0 }}
      >
        <circle
          cx="7"
          cy="7"
          r="5"
          stroke={ng.colors.accentCyan}
          strokeWidth="1.5"
          strokeDasharray="20"
          strokeDashoffset="6"
          strokeLinecap="round"
        />
      </svg>
      {message}
      <style>{`
        @keyframes neuro-spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
