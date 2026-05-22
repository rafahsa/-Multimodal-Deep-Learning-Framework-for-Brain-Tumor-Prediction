import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

const banner: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.6rem',
  padding: '0.7rem 1rem',
  borderRadius: ng.radii.sm,
  background: `${ng.colors.accentVioletGlow}`,
  border: `1px solid rgba(139, 92, 246, 0.12)`,
  fontSize: '0.74rem',
  color: ng.colors.textDim,
  lineHeight: 1.5,
};

const icon: CSSProperties = {
  flexShrink: 0,
  color: ng.colors.accentViolet,
  opacity: 0.7,
  fontSize: '0.85rem',
};

export function PrivacyDisclaimer() {
  return (
    <div style={banner}>
      <span style={icon}>
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
          <path
            d="M7 1L2 3.5V6.5C2 9.55 4.15 12.37 7 13C9.85 12.37 12 9.55 12 6.5V3.5L7 1Z"
            stroke="currentColor"
            strokeWidth="1.2"
            strokeLinejoin="round"
          />
          <circle cx="7" cy="6" r="1" fill="currentColor" />
          <path d="M7 8v2" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
      </span>
      <span>
        <strong style={{ color: ng.colors.textSecondary, fontWeight: 600 }}>Research use only</strong>
        {' '}— No authentication enforced. You are responsible for the privacy of uploaded medical data. Files are processed transiently and never stored.
      </span>
    </div>
  );
}
