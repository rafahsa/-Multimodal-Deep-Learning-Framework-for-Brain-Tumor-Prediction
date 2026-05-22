import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';
import type { PredictionResult } from '../../types/prediction';
import { formatRelativeTime } from '../../utils/formatRelativeTime';

interface HistoryItemProps {
  prediction: PredictionResult;
  isActive: boolean;
  onSelect: () => void;
}

export function HistoryItem({ prediction, isActive, onSelect }: HistoryItemProps) {
  const classification = prediction.thresholds.balanced.classification;
  const isHGG = classification === 'HGG';
  const gradeColor = isHGG ? ng.colors.accentWarm : ng.colors.accentCyan;
  const probPct = (prediction.calibrated_probability * 100).toFixed(1);

  const card: CSSProperties = {
    width: '100%',
    textAlign: 'left',
    padding: '0.85rem 0.95rem',
    borderRadius: ng.radii.sm,
    border: `1px solid ${isActive ? ng.colors.borderActive : ng.colors.border}`,
    background: isActive
      ? `linear-gradient(145deg, ${ng.colors.accentCyanGlow} 0%, ${ng.colors.bgElevated} 70%)`
      : ng.colors.bgSecondary,
    boxShadow: isActive ? ng.shadows.glowCyan : 'none',
    cursor: 'pointer',
    transition: ng.transitions.default,
    outline: 'none',
    fontFamily: ng.fonts.body,
  };

  return (
    <button
      type="button"
      onClick={onSelect}
      style={card}
      aria-current={isActive ? 'true' : undefined}
      onMouseEnter={(e) => {
        if (isActive) return;
        e.currentTarget.style.borderColor = 'rgba(0, 212, 170, 0.12)';
        e.currentTarget.style.background = ng.colors.bgElevated;
      }}
      onMouseLeave={(e) => {
        if (isActive) return;
        e.currentTarget.style.borderColor = ng.colors.border;
        e.currentTarget.style.background = ng.colors.bgSecondary;
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: '0.5rem',
          marginBottom: '0.45rem',
        }}
      >
        <span
          style={{
            color: ng.colors.textPrimary,
            fontSize: '0.82rem',
            fontWeight: 600,
            lineHeight: 1.3,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
            flex: 1,
          }}
          title={prediction.patient_label}
        >
          {prediction.patient_label}
        </span>
        <span
          style={{
            fontFamily: ng.fonts.mono,
            fontSize: '0.72rem',
            fontWeight: 700,
            color: gradeColor,
            padding: '0.15rem 0.4rem',
            borderRadius: ng.radii.pill,
            border: `1px solid ${gradeColor}35`,
            background: `${gradeColor}12`,
            flexShrink: 0,
          }}
        >
          {classification}
        </span>
      </div>

      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.5rem',
        }}
      >
        <span
          style={{
            fontFamily: ng.fonts.mono,
            fontSize: '0.88rem',
            fontWeight: 600,
            color: gradeColor,
          }}
        >
          {probPct}%
        </span>
        <span
          style={{
            color: ng.colors.textDim,
            fontSize: '0.68rem',
            fontFamily: ng.fonts.mono,
          }}
        >
          {formatRelativeTime(prediction.timestamp)}
        </span>
      </div>
    </button>
  );
}
