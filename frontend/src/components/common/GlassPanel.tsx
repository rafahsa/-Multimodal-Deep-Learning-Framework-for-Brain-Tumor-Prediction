import { type CSSProperties, type ReactNode } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

interface GlassPanelProps {
  children: ReactNode;
  style?: CSSProperties;
}

const panel: CSSProperties = {
  background: ng.glass.background,
  backdropFilter: ng.glass.backdropFilter,
  WebkitBackdropFilter: ng.glass.backdropFilter,
  border: `1px solid ${ng.glass.border}`,
  borderRadius: ng.radii.lg,
  padding: '2rem',
  transition: ng.transitions.default,
};

export function GlassPanel({ children, style }: GlassPanelProps) {
  return <div style={{ ...panel, ...style }}>{children}</div>;
}
