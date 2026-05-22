import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

interface SpinnerProps {
  size?: number;
  style?: CSSProperties;
}

export function Spinner({ size = 28, style }: SpinnerProps) {
  const keyframesId = 'neuro-spin';

  return (
    <>
      <style>{`
        @keyframes ${keyframesId} {
          to { transform: rotate(360deg); }
        }
      `}</style>
      <div
        role="status"
        aria-label="Loading"
        style={{
          width: size,
          height: size,
          borderRadius: '50%',
          border: `3px solid ${ng.colors.border}`,
          borderTopColor: ng.colors.accentCyan,
          animation: `${keyframesId} 0.7s linear infinite`,
          ...style,
        }}
      />
    </>
  );
}
