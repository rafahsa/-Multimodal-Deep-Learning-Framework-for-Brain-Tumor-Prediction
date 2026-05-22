import { type CSSProperties, useState, useEffect } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

const nav: CSSProperties = {
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  zIndex: 1000,
  padding: '1rem 2.5rem',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  backdropFilter: 'blur(24px) saturate(1.3)',
  WebkitBackdropFilter: 'blur(24px) saturate(1.3)',
  background: 'rgba(6, 8, 16, 0.8)',
  borderBottom: `1px solid ${ng.colors.border}`,
  transition: 'transform 0.5s cubic-bezier(0.22, 1, 0.36, 1)',
};

const brandStyle: CSSProperties = {
  fontFamily: ng.fonts.display,
  fontSize: '1.35rem',
  fontWeight: 700,
  letterSpacing: '-0.02em',
  color: ng.colors.textPrimary,
  textDecoration: 'none',
};

const linkStyle: CSSProperties = {
  fontSize: '0.82rem',
  fontWeight: 500,
  color: ng.colors.textSecondary,
  textDecoration: 'none',
  letterSpacing: '0.04em',
  textTransform: 'uppercase' as const,
  transition: ng.transitions.fast,
};

interface NavbarProps {
  links?: { label: string; href: string }[];
}

export function Navbar({ links }: NavbarProps) {
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    let lastScroll = 0;
    const onScroll = () => {
      const curr = window.scrollY;
      setHidden(curr > lastScroll && curr > 120);
      lastScroll = curr;
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <header
      className="nav-bar"
      style={{
        ...nav,
        transform: hidden ? 'translateY(-100%)' : 'translateY(0)',
      }}
    >
      <a href="/" className="nav-brand" style={brandStyle}>
        Neuro<em style={{ color: ng.colors.accentCyan, fontStyle: 'italic' }}>Grade</em>
      </a>

      {links && links.length > 0 && (
        <nav className="nav-links" style={{ listStyle: 'none' }}>
          {links.map((l) => (
            <a
              key={l.href}
              href={l.href}
              style={linkStyle}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.color = ng.colors.accentCyan;
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.color = ng.colors.textSecondary;
              }}
            >
              {l.label}
            </a>
          ))}
        </nav>
      )}
    </header>
  );
}
