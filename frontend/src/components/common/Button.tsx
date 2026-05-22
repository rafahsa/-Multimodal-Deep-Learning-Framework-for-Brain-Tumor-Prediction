import { type ButtonHTMLAttributes, type CSSProperties, useState } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

type Variant = 'primary' | 'outline';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

const base: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '0.6rem',
  padding: '0.9rem 2rem',
  fontSize: '0.88rem',
  fontWeight: 600,
  fontFamily: ng.fonts.body,
  borderRadius: ng.radii.sm,
  cursor: 'pointer',
  transition: ng.transitions.default,
  border: 'none',
  textDecoration: 'none',
  lineHeight: 1,
};

const variants: Record<Variant, { idle: CSSProperties; hover: CSSProperties }> = {
  primary: {
    idle: {
      background: ng.colors.accentCyan,
      color: ng.colors.bgPrimary,
    },
    hover: {
      background: '#00eabc',
      transform: 'translateY(-2px)',
      boxShadow: `0 12px 40px ${ng.colors.accentCyanGlow}`,
    },
  },
  outline: {
    idle: {
      background: 'transparent',
      color: ng.colors.textPrimary,
      border: '1px solid rgba(232, 230, 225, 0.15)',
    },
    hover: {
      borderColor: ng.colors.accentCyan,
      background: ng.colors.accentCyanGlow,
    },
  },
};

export function Button({ variant = 'primary', style, disabled, ...rest }: ButtonProps) {
  const [hovered, setHovered] = useState(false);
  const v = variants[variant];

  const disabledStyle: CSSProperties = disabled
    ? { opacity: 0.45, cursor: 'not-allowed', pointerEvents: 'none' as const }
    : {};

  return (
    <button
      {...rest}
      disabled={disabled}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        ...base,
        ...v.idle,
        ...(hovered && !disabled ? v.hover : {}),
        ...disabledStyle,
        ...style,
      }}
    />
  );
}
