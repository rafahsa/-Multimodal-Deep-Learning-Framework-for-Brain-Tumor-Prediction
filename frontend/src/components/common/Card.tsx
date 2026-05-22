import { type CSSProperties, type ReactNode } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

interface CardProps {
  children: ReactNode;
  style?: CSSProperties;
  className?: string;
}

const card: CSSProperties = {
  background: ng.colors.bgElevated,
  border: `1px solid ${ng.colors.border}`,
  borderRadius: ng.radii.lg,
  padding: '1.8rem',
  transition: ng.transitions.default,
};

export function Card({ children, style, className }: CardProps) {
  return (
    <div className={className} style={{ ...card, ...style }}>
      {children}
    </div>
  );
}
