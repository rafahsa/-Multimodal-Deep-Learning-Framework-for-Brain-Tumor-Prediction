import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

const shimmer: CSSProperties = {
  background: `linear-gradient(
    90deg,
    ${ng.colors.bgSecondary} 0%,
    rgba(0, 212, 170, 0.08) 50%,
    ${ng.colors.bgSecondary} 100%
  )`,
  backgroundSize: '200% 100%',
  animation: 'neuro-shimmer 1.4s ease-in-out infinite',
  borderRadius: ng.radii.sm,
};

function Bone({ width, height, style }: { width: string; height: string; style?: CSSProperties }) {
  return <div style={{ ...shimmer, width, height, ...style }} aria-hidden />;
}

const card: CSSProperties = {
  background: ng.colors.bgElevated,
  border: `1px solid ${ng.colors.border}`,
  borderRadius: ng.radii.lg,
  padding: '2rem',
};

export function PredictionSkeleton() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }} aria-busy="true" aria-label="Loading prediction">
      <div style={card}>
        <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
          <div style={{ flex: 1, minWidth: 200, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <Bone width="40%" height="12px" />
            <Bone width="70%" height="20px" />
            <Bone width="50%" height="48px" />
            <Bone width="35%" height="24px" />
          </div>
          <Bone width="140px" height="140px" style={{ borderRadius: '50%', flexShrink: 0 }} />
        </div>
        <Bone width="100%" height="1px" style={{ margin: '1.5rem 0' }} />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '1rem' }}>
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
              <Bone width="60%" height="10px" />
              <Bone width="80%" height="14px" />
            </div>
          ))}
        </div>
      </div>

      <div style={{ ...card, padding: '1.25rem' }}>
        <Bone width="30%" height="14px" style={{ marginBottom: '1rem' }} />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.85rem' }}>
          {Array.from({ length: 3 }).map((_, i) => (
            <Bone key={i} width="100%" height="100px" />
          ))}
        </div>
      </div>

      <style>{`
        @keyframes neuro-shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </div>
  );
}
