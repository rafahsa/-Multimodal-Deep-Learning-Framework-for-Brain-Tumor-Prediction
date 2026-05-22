import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

const footer: CSSProperties = {
  padding: '2.5rem',
  borderTop: `1px solid ${ng.colors.border}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  flexWrap: 'wrap',
  gap: '1.5rem',
};

const leftText: CSSProperties = {
  fontSize: '0.78rem',
  color: ng.colors.textDim,
  lineHeight: 1.6,
};

const linkList: CSSProperties = {
  display: 'flex',
  gap: '2rem',
  listStyle: 'none',
  margin: 0,
  padding: 0,
};

const linkItem: CSSProperties = {
  fontSize: '0.78rem',
  color: ng.colors.textDim,
  textDecoration: 'none',
  transition: 'color 0.3s',
};

export function Footer() {
  const links = [
    { label: 'Predict', href: '#' },
    { label: 'Architecture', href: '/' },
    { label: 'GitHub', href: 'https://github.com' },
  ];

  return (
    <footer style={footer}>
      <span style={leftText}>
        <strong style={{ color: ng.colors.textSecondary }}>NeuroGrade</strong> &mdash;
        Multimodal Deep Learning Framework for Brain Tumor Grade Classification &copy; 2026
      </span>
      <ul style={linkList}>
        {links.map((l) => (
          <li key={l.label}>
            <a
              href={l.href}
              style={linkItem}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.color = ng.colors.accentCyan;
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.color = ng.colors.textDim;
              }}
            >
              {l.label}
            </a>
          </li>
        ))}
      </ul>
    </footer>
  );
}
