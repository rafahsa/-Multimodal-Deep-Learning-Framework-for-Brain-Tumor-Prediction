import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

interface ErrorBannerProps {
  message: string;
  suggestion?: string;
  onDismiss?: () => void;
}

const banner: CSSProperties = {
  background: ng.colors.accentWarmGlow,
  border: `1px solid rgba(240, 114, 75, 0.3)`,
  borderRadius: ng.radii.md,
  padding: '1.2rem 1.6rem',
  display: 'flex',
  gap: '1rem',
  alignItems: 'flex-start',
};

const iconStyle: CSSProperties = {
  color: ng.colors.accentWarm,
  fontSize: '1.2rem',
  lineHeight: 1,
  flexShrink: 0,
  marginTop: '2px',
};

const messageStyle: CSSProperties = {
  flex: 1,
};

const titleStyle: CSSProperties = {
  color: ng.colors.accentWarm,
  fontWeight: 600,
  fontSize: '0.9rem',
  marginBottom: '0.3rem',
};

const bodyStyle: CSSProperties = {
  color: ng.colors.textSecondary,
  fontSize: '0.84rem',
  lineHeight: 1.5,
};

const dismissBtn: CSSProperties = {
  background: 'none',
  border: 'none',
  color: ng.colors.textDim,
  cursor: 'pointer',
  fontSize: '1.1rem',
  padding: '0 0.2rem',
  lineHeight: 1,
  transition: ng.transitions.fast,
};

export function ErrorBanner({ message, suggestion, onDismiss }: ErrorBannerProps) {
  return (
    <div style={banner} role="alert">
      <span style={iconStyle}>&#9888;</span>
      <div style={messageStyle}>
        <div style={titleStyle}>{message}</div>
        {suggestion && <div style={bodyStyle}>{suggestion}</div>}
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          style={dismissBtn}
          aria-label="Dismiss error"
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.color = ng.colors.textPrimary;
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.color = ng.colors.textDim;
          }}
        >
          &times;
        </button>
      )}
    </div>
  );
}
