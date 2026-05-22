import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';
import type { OperatingMode } from '../../types/prediction';
import { OPERATING_MODES } from '../../types/prediction';

interface ModeSelectorProps {
  selected: OperatingMode;
  onChange: (mode: OperatingMode) => void;
  disabled?: boolean;
  compact?: boolean;
}

const MODES: OperatingMode[] = ['balanced', 'high_sensitivity'];

const grid: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(2, 1fr)',
  gap: '0.85rem',
};

function ModeCard({
  modeId,
  isActive,
  onSelect,
  disabled,
  compact,
}: {
  modeId: OperatingMode;
  isActive: boolean;
  onSelect: () => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  const mode = OPERATING_MODES[modeId];
  const isBalanced = modeId === 'balanced';

  const cardStyle: CSSProperties = {
    position: 'relative',
    textAlign: 'left',
    padding: compact ? '1rem 1.1rem' : '1.25rem 1.35rem',
    borderRadius: ng.radii.md,
    border: `1px solid ${isActive ? ng.colors.borderActive : ng.colors.border}`,
    background: isActive
      ? `linear-gradient(145deg, ${ng.colors.accentCyanGlow} 0%, ${ng.colors.bgElevated} 55%)`
      : ng.colors.bgSecondary,
    boxShadow: isActive ? ng.shadows.glowCyan : 'none',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.55 : 1,
    transition: ng.transitions.default,
    outline: 'none',
    width: '100%',
    fontFamily: ng.fonts.body,
  };

  const statGrid: CSSProperties = {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '0.45rem 0.75rem',
    marginTop: compact ? '0.65rem' : '0.85rem',
    paddingTop: compact ? '0.65rem' : '0.85rem',
    borderTop: `1px solid ${ng.colors.border}`,
  };

  return (
    <button
      type="button"
      role="radio"
      aria-checked={isActive}
      aria-label={`${mode.name}, threshold ${mode.threshold}`}
      disabled={disabled}
      onClick={onSelect}
      style={cardStyle}
      onMouseEnter={(e) => {
        if (disabled || isActive) return;
        e.currentTarget.style.borderColor = 'rgba(0, 212, 170, 0.12)';
        e.currentTarget.style.background = ng.colors.bgElevated;
      }}
      onMouseLeave={(e) => {
        if (disabled || isActive) return;
        e.currentTarget.style.borderColor = ng.colors.border;
        e.currentTarget.style.background = ng.colors.bgSecondary;
      }}
    >
      {isActive && (
        <span
          style={{
            position: 'absolute',
            top: '0.75rem',
            right: '0.75rem',
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: ng.colors.accentCyan,
            boxShadow: `0 0 12px ${ng.colors.accentCyan}`,
          }}
          aria-hidden
        />
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
        <span
          style={{
            fontFamily: ng.fonts.display,
            fontSize: compact ? '1.05rem' : '1.15rem',
            fontWeight: 700,
            color: isActive ? ng.colors.textPrimary : ng.colors.textSecondary,
            letterSpacing: '-0.02em',
            lineHeight: 1.25,
          }}
        >
          {mode.name}
        </span>

        <span
          style={{
            display: 'inline-flex',
            alignSelf: 'flex-start',
            alignItems: 'center',
            gap: '0.35rem',
            padding: '0.2rem 0.55rem',
            borderRadius: ng.radii.pill,
            background: isActive ? `${ng.colors.accentCyan}18` : 'rgba(232, 230, 225, 0.04)',
            border: `1px solid ${isActive ? `${ng.colors.accentCyan}35` : ng.colors.border}`,
            color: isActive ? ng.colors.accentCyan : ng.colors.textDim,
            fontSize: '0.72rem',
            fontFamily: ng.fonts.mono,
            fontWeight: 600,
          }}
        >
          τ = {mode.threshold}
        </span>

        <p
          style={{
            margin: compact ? '0.35rem 0 0' : '0.45rem 0 0',
            color: ng.colors.textDim,
            fontSize: compact ? '0.74rem' : '0.78rem',
            lineHeight: 1.5,
          }}
        >
          {isBalanced
            ? 'Equal precision and recall — suited for routine screening when false positives and false negatives carry similar cost.'
            : 'Higher recall, fewer missed HGG — suited for triage when under-calling high-grade disease is the greater risk.'}
        </p>
      </div>

      <div style={statGrid}>
        <StatChip label="Precision" value={mode.precision.toFixed(4)} highlight={isActive} />
        <StatChip label="Recall" value={mode.recall.toFixed(4)} highlight={isActive} />
        <StatChip
          label="False Neg."
          value={`${mode.expectedFN} cases`}
          mono
          highlight={isActive}
        />
        <StatChip
          label="False Pos."
          value={`${mode.expectedFP} cases`}
          mono
          highlight={isActive}
        />
      </div>
    </button>
  );
}

function StatChip({
  label,
  value,
  mono,
  highlight,
}: {
  label: string;
  value: string;
  mono?: boolean;
  highlight?: boolean;
}) {
  return (
    <div>
      <div
        style={{
          color: ng.colors.textDim,
          fontSize: '0.62rem',
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          marginBottom: '0.12rem',
        }}
      >
        {label}
      </div>
      <div
        style={{
          color: highlight ? ng.colors.textPrimary : ng.colors.textSecondary,
          fontSize: '0.78rem',
          fontWeight: 600,
          fontFamily: mono ? ng.fonts.mono : ng.fonts.body,
        }}
      >
        {value}
      </div>
    </div>
  );
}

export function ModeSelector({ selected, onChange, disabled, compact }: ModeSelectorProps) {
  return (
    <div
      role="radiogroup"
      aria-label="Clinical operating mode"
      style={grid}
      className="mode-selector-grid"
    >
      {MODES.map((modeId) => (
        <ModeCard
          key={modeId}
          modeId={modeId}
          isActive={selected === modeId}
          onSelect={() => onChange(modeId)}
          disabled={disabled}
          compact={compact}
        />
      ))}
      <style>{`
        @media (max-width: 640px) {
          .mode-selector-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}
