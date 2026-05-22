import { type CSSProperties, useId, useState } from 'react';
import { neurograde as ng } from '../../theme/neurograde';
import type { PredictionResult } from '../../types/prediction';
import { MODEL_CONFIGS, weightedContribution } from './modelConfig';
import { ModelContributionBar } from './ModelContributionBar';
import { EnsembleFormula } from './EnsembleFormula';

interface ModelDetailPanelProps {
  result: PredictionResult;
}

const toggleBtn: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '0.5rem',
  width: '100%',
  marginTop: '1.25rem',
  padding: '0.85rem 1.25rem',
  borderRadius: ng.radii.md,
  border: `1px solid ${ng.colors.border}`,
  background: ng.colors.bgSecondary,
  color: ng.colors.textSecondary,
  fontFamily: ng.fonts.body,
  fontSize: '0.85rem',
  fontWeight: 600,
  cursor: 'pointer',
  transition: ng.transitions.default,
  outline: 'none',
};

const panelInner: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '1.25rem',
  paddingTop: '1.25rem',
};

const modelGrid: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
  gap: '0.85rem',
};

function ModelCard({
  name,
  architecture,
  typeBadge,
  color,
  glow,
  probability,
  coefficient,
  weighted,
}: {
  name: string;
  architecture: string;
  typeBadge: string;
  color: string;
  glow: string;
  probability: number;
  coefficient: number;
  weighted: number;
}) {
  return (
    <div
      style={{
        padding: '1.1rem 1.15rem',
        borderRadius: ng.radii.md,
        border: `1px solid ${ng.colors.border}`,
        background: `linear-gradient(160deg, ${glow} 0%, ${ng.colors.bgElevated} 65%)`,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          position: 'absolute',
          top: '-20px',
          right: '-20px',
          width: '80px',
          height: '80px',
          borderRadius: '50%',
          background: `radial-gradient(circle, ${color}18 0%, transparent 70%)`,
          pointerEvents: 'none',
        }}
      />

      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: '0.5rem',
          marginBottom: '0.65rem',
        }}
      >
        <span style={{ color: ng.colors.textPrimary, fontSize: '0.88rem', fontWeight: 700 }}>
          {name}
        </span>
        <span
          style={{
            fontSize: '0.62rem',
            fontFamily: ng.fonts.mono,
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            padding: '0.2rem 0.45rem',
            borderRadius: ng.radii.pill,
            border: `1px solid ${color}40`,
            color,
            background: `${color}12`,
            flexShrink: 0,
          }}
        >
          {typeBadge}
        </span>
      </div>

      <p
        style={{
          margin: '0 0 0.85rem',
          color: ng.colors.textDim,
          fontSize: '0.72rem',
          lineHeight: 1.4,
        }}
      >
        {architecture}
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
          <span
            style={{
              color: ng.colors.textDim,
              fontSize: '0.68rem',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            P(HGG)
          </span>
          <span
            style={{
              fontFamily: ng.fonts.mono,
              fontSize: '1.25rem',
              fontWeight: 700,
              color,
            }}
          >
            {(probability * 100).toFixed(1)}%
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: ng.colors.textDim, fontSize: '0.72rem' }}>Coefficient β</span>
          <span style={{ fontFamily: ng.fonts.mono, fontSize: '0.78rem', color: ng.colors.textSecondary }}>
            {coefficient >= 0 ? '+' : ''}
            {coefficient.toFixed(3)}
          </span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ color: ng.colors.textDim, fontSize: '0.72rem' }}>|β × p|</span>
          <span style={{ fontFamily: ng.fonts.mono, fontSize: '0.78rem', color: ng.colors.textPrimary }}>
            {weighted.toFixed(4)}
          </span>
        </div>
      </div>
    </div>
  );
}

export function ModelDetailPanel({ result }: ModelDetailPanelProps) {
  const [expanded, setExpanded] = useState(false);
  const panelId = useId();

  return (
    <div>
      <button
        type="button"
        style={toggleBtn}
        aria-expanded={expanded}
        aria-controls={panelId}
        onClick={() => setExpanded((v) => !v)}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = ng.colors.borderActive;
          e.currentTarget.style.color = ng.colors.textPrimary;
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = ng.colors.border;
          e.currentTarget.style.color = ng.colors.textSecondary;
        }}
      >
        <svg
          width="16"
          height="16"
          viewBox="0 0 16 16"
          fill="none"
          style={{
            transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
            transition: ng.transitions.fast,
          }}
        >
          <path
            d="M4 6l4 4 4-4"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        {expanded ? 'Hide Model Details' : 'View Model Details'}
        <span
          style={{
            fontFamily: ng.fonts.mono,
            fontSize: '0.7rem',
            color: ng.colors.textDim,
            fontWeight: 500,
          }}
        >
          3-model ensemble
        </span>
      </button>

      <div
        id={panelId}
        style={{
          display: 'grid',
          gridTemplateRows: expanded ? '1fr' : '0fr',
          transition: 'grid-template-rows 0.45s cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      >
        <div style={{ overflow: 'hidden' }}>
          <div
            style={{
              ...panelInner,
              opacity: expanded ? 1 : 0,
              transform: expanded ? 'translateY(0)' : 'translateY(-8px)',
              transition: 'opacity 0.35s ease, transform 0.45s cubic-bezier(0.22, 1, 0.36, 1)',
            }}
          >
            <div style={modelGrid}>
              {MODEL_CONFIGS.map((cfg) => {
                const prob = result.model_probabilities[cfg.probabilityKey];
                const coef = result.meta_learner_coefficients[cfg.coefficientKey];
                const weight = weightedContribution(coef, prob);
                return (
                  <ModelCard
                    key={cfg.key}
                    name={cfg.name}
                    architecture={cfg.architecture}
                    typeBadge={cfg.typeBadge}
                    color={cfg.color}
                    glow={cfg.glow}
                    probability={prob}
                    coefficient={coef}
                    weighted={weight}
                  />
                );
              })}
            </div>

            <ModelContributionBar
              modelProbabilities={result.model_probabilities}
              metaLearnerCoefficients={result.meta_learner_coefficients}
            />

            <EnsembleFormula result={result} />
          </div>
        </div>
      </div>

      <style>{`
        @media (max-width: 520px) {
          .model-detail-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </div>
  );
}
