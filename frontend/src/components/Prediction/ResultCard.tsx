import { type CSSProperties, useEffect, useRef, useState } from 'react';
import { neurograde as ng } from '../../theme/neurograde';
import type { PredictionResult, OperatingMode } from '../../types/prediction';
import {
  OPERATING_MODES,
  deriveClassification,
  getConfidenceLevel,
} from '../../types/prediction';
import { ProbabilityGauge } from './ProbabilityGauge';

interface ResultCardProps {
  result: PredictionResult;
  operatingMode: OperatingMode;
}

const card: CSSProperties = {
  background: ng.colors.bgElevated,
  border: `1px solid ${ng.colors.border}`,
  borderRadius: ng.radii.lg,
  padding: '2rem',
  position: 'relative',
  overflow: 'hidden',
};

const headerRow: CSSProperties = {
  display: 'flex',
  alignItems: 'flex-start',
  justifyContent: 'space-between',
  gap: '1.5rem',
  flexWrap: 'wrap',
};

const leftCol: CSSProperties = {
  flex: 1,
  minWidth: '200px',
  display: 'flex',
  flexDirection: 'column',
  gap: '1rem',
};

const thresholdRow: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(2, 1fr)',
  gap: '0.65rem',
  marginTop: '0.25rem',
};

function ConfidenceBadge({ level }: { level: string }) {
  const colors: Record<string, { bg: string; text: string; border: string }> = {
    High: {
      bg: `${ng.colors.accentCyan}15`,
      text: ng.colors.accentCyan,
      border: `${ng.colors.accentCyan}30`,
    },
    Medium: {
      bg: `${ng.colors.accentViolet}15`,
      text: ng.colors.accentViolet,
      border: `${ng.colors.accentViolet}30`,
    },
    Low: {
      bg: `${ng.colors.accentWarm}15`,
      text: ng.colors.accentWarm,
      border: `${ng.colors.accentWarm}30`,
    },
  };

  const c = colors[level] ?? colors.Low;

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.35rem',
        padding: '0.3rem 0.7rem',
        borderRadius: ng.radii.pill,
        background: c.bg,
        border: `1px solid ${c.border}`,
        color: c.text,
        fontSize: '0.72rem',
        fontWeight: 600,
        fontFamily: ng.fonts.mono,
        textTransform: 'uppercase' as const,
        letterSpacing: '0.04em',
      }}
    >
      <span
        style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          background: c.text,
        }}
      />
      {level} Confidence
    </span>
  );
}

function MetaStat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
      <span
        style={{
          color: ng.colors.textDim,
          fontSize: '0.7rem',
          textTransform: 'uppercase' as const,
          letterSpacing: '0.06em',
        }}
      >
        {label}
      </span>
      <span
        style={{
          color: ng.colors.textSecondary,
          fontSize: '0.82rem',
          fontWeight: 500,
          fontFamily: mono ? ng.fonts.mono : ng.fonts.body,
        }}
      >
        {value}
      </span>
    </div>
  );
}

function ThresholdOutcome({
  modeId,
  isActive,
  apiClassification,
}: {
  modeId: OperatingMode;
  isActive: boolean;
  apiClassification: 'HGG' | 'LGG';
}) {
  const mode = OPERATING_MODES[modeId];
  const isHGG = apiClassification === 'HGG';
  const gradeColor = isHGG ? ng.colors.accentWarm : ng.colors.accentCyan;

  return (
    <div
      style={{
        padding: '0.75rem 0.9rem',
        borderRadius: ng.radii.sm,
        border: `1px solid ${isActive ? ng.colors.borderActive : ng.colors.border}`,
        background: isActive
          ? `linear-gradient(160deg, ${ng.colors.accentCyanGlow} 0%, transparent 70%)`
          : ng.colors.bgSecondary,
        boxShadow: isActive ? ng.shadows.glowCyan : 'none',
        transition: ng.transitions.default,
        opacity: isActive ? 1 : 0.72,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.5rem',
          marginBottom: '0.35rem',
        }}
      >
        <span
          style={{
            color: isActive ? ng.colors.textPrimary : ng.colors.textDim,
            fontSize: '0.68rem',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
          }}
        >
          {mode.name}
        </span>
        {isActive && (
          <span
            style={{
              fontSize: '0.6rem',
              fontFamily: ng.fonts.mono,
              color: ng.colors.accentCyan,
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}
          >
            Active
          </span>
        )}
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: '0.5rem',
        }}
      >
        <span
          style={{
            fontFamily: ng.fonts.mono,
            fontSize: '1.35rem',
            fontWeight: 700,
            color: gradeColor,
            letterSpacing: '-0.02em',
          }}
        >
          {apiClassification}
        </span>
        <span style={{ color: ng.colors.textDim, fontSize: '0.72rem', fontFamily: ng.fonts.mono }}>
          τ = {mode.threshold}
        </span>
      </div>
    </div>
  );
}

export function ResultCard({ result, operatingMode }: ResultCardProps) {
  const mode = OPERATING_MODES[operatingMode];
  const classification = deriveClassification(result.calibrated_probability, operatingMode);
  const isHGG = classification === 'HGG';
  const confidence = getConfidenceLevel(result.calibrated_probability);
  const gradeColor = isHGG ? ng.colors.accentWarm : ng.colors.accentCyan;

  const prevClassification = useRef(classification);
  const [gradeAnimating, setGradeAnimating] = useState(false);

  useEffect(() => {
    if (prevClassification.current !== classification) {
      setGradeAnimating(true);
      prevClassification.current = classification;
      const t = window.setTimeout(() => setGradeAnimating(false), 480);
      return () => window.clearTimeout(t);
    }
    prevClassification.current = classification;
  }, [classification]);

  return (
    <div style={card}>
      <div
        style={{
          position: 'absolute',
          top: '-60px',
          right: '-60px',
          width: '200px',
          height: '200px',
          borderRadius: '50%',
          background: `radial-gradient(circle, ${gradeColor}10 0%, transparent 70%)`,
          pointerEvents: 'none',
          transition: ng.transitions.slow,
        }}
      />

      <div style={headerRow}>
        <div style={leftCol}>
          <div>
            <span
              style={{
                color: ng.colors.textDim,
                fontSize: '0.72rem',
                textTransform: 'uppercase' as const,
                letterSpacing: '0.08em',
              }}
            >
              Patient Scan
            </span>
            <div
              style={{
                color: ng.colors.textPrimary,
                fontSize: '1rem',
                fontWeight: 600,
                marginTop: '0.15rem',
              }}
            >
              {result.patient_label}
            </div>
          </div>

          <div>
            <span
              style={{
                color: ng.colors.textDim,
                fontSize: '0.72rem',
                textTransform: 'uppercase' as const,
                letterSpacing: '0.08em',
              }}
            >
              Predicted Grade
            </span>
            <div
              className={gradeAnimating ? 'neuro-grade-flip' : undefined}
              style={{
                fontFamily: ng.fonts.display,
                fontSize: '2.8rem',
                fontWeight: 900,
                color: gradeColor,
                letterSpacing: '-0.03em',
                lineHeight: 1,
                marginTop: '0.2rem',
                textShadow: `0 0 40px ${gradeColor}30`,
                transition: 'color 0.35s cubic-bezier(0.22, 1, 0.36, 1), text-shadow 0.35s ease',
              }}
            >
              {classification}
            </div>
            <span
              style={{
                display: 'inline-block',
                marginTop: '0.3rem',
                color: ng.colors.textSecondary,
                fontSize: '0.78rem',
              }}
            >
              {isHGG ? 'High-Grade Glioma' : 'Low-Grade Glioma'}
            </span>
          </div>

          <ConfidenceBadge level={confidence} />
        </div>

        <ProbabilityGauge probability={result.calibrated_probability} />
      </div>

      <div
        style={{
          height: '1px',
          background: ng.colors.border,
          margin: '1.5rem 0',
        }}
      />

      <div style={{ marginBottom: '1.25rem' }}>
        <span
          style={{
            color: ng.colors.textDim,
            fontSize: '0.7rem',
            textTransform: 'uppercase' as const,
            letterSpacing: '0.08em',
            display: 'block',
            marginBottom: '0.55rem',
          }}
        >
          Classification by Operating Mode
        </span>
        <div style={thresholdRow} className="result-threshold-row">
          <ThresholdOutcome
            modeId="balanced"
            isActive={operatingMode === 'balanced'}
            apiClassification={result.thresholds.balanced.classification}
          />
          <ThresholdOutcome
            modeId="high_sensitivity"
            isActive={operatingMode === 'high_sensitivity'}
            apiClassification={result.thresholds.high_sensitivity.classification}
          />
        </div>
        <p
          style={{
            margin: '0.65rem 0 0',
            color: ng.colors.textDim,
            fontSize: '0.72rem',
            lineHeight: 1.45,
          }}
        >
          Calibrated P(HGG) stays fixed at{' '}
          <span style={{ fontFamily: ng.fonts.mono, color: ng.colors.textSecondary }}>
            {(result.calibrated_probability * 100).toFixed(1)}%
          </span>
          . Switching mode only changes the decision threshold.
        </p>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '1rem',
        }}
      >
        <MetaStat label="Operating Mode" value={mode.name} />
        <MetaStat label="Threshold" value={`τ = ${mode.threshold}`} mono />
        <MetaStat
          label="Processing Time"
          value={
            result.processing_duration_ms >= 1000
              ? `${(result.processing_duration_ms / 1000).toFixed(1)}s`
              : `${result.processing_duration_ms}ms`
          }
          mono
        />
        <MetaStat label="Device" value={result.device_used.toUpperCase()} mono />
        <MetaStat
          label="Uncalibrated P"
          value={`${(result.uncalibrated_probability * 100).toFixed(1)}%`}
          mono
        />
        <MetaStat label="Ensemble Logit" value={result.ensemble_logit.toFixed(3)} mono />
      </div>

      <style>{`
        @keyframes neuro-grade-flip {
          0% { opacity: 1; transform: scale(1) translateY(0); }
          35% { opacity: 0.35; transform: scale(0.94) translateY(6px); }
          100% { opacity: 1; transform: scale(1) translateY(0); }
        }
        .neuro-grade-flip {
          animation: neuro-grade-flip 0.48s cubic-bezier(0.22, 1, 0.36, 1);
        }
        @media (max-width: 520px) {
          .result-threshold-row {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}
