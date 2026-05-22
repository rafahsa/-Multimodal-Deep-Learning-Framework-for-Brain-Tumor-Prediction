import { type CSSProperties, type ReactNode } from 'react';
import { useReveal } from '../../hooks/useReveal';

interface RevealProps {
  children: ReactNode;
  className?: string;
  delay?: 0 | 1 | 2 | 3;
  style?: CSSProperties;
}

export function Reveal({ children, className = '', delay = 0, style }: RevealProps) {
  const [ref, visible] = useReveal<HTMLDivElement>();

  const delayClass =
    delay > 0 ? ` reveal-delay-${delay}` : '';

  return (
    <div
      ref={ref}
      className={`reveal${visible ? ' visible' : ''}${delayClass}${className ? ` ${className}` : ''}`}
      style={style}
    >
      {children}
    </div>
  );
}
