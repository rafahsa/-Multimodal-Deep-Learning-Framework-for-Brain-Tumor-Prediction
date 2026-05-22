import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';
import type { ModelProbabilities, MetaLearnerCoefficients } from '../../types/prediction';
import { MODEL_CONFIGS, weightedContribution } from './modelConfig';

interface ModelContributionBarProps {
  modelProbabilities: ModelProbabilities;
  metaLearnerCoefficients: MetaLearnerCoefficients;
}

const wrap: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.85rem',
};

const barTrack: CSSProperties = {
  display: 'flex',
  height: '28px',
  borderRadius: ng.radii.pill,
  overflow: 'hidden',
  border: `1px solid ${ng.colors.border}`,
  background: ng.colors.bgSecondary,
};

const legend: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.55rem',
};

export function ModelContributionBar({
  modelProbabilities,
  metaLearnerCoefficients,
}: ModelContributionBarProps) {
  const segments = MODEL_CONFIGS.map((cfg) => {
    const prob = modelProbabilities[cfg.probabilityKey];
    const coef = metaLearnerCoefficients[cfg.coefficientKey];
    const weight = weightedContribution(coef, prob);
    return { cfg, prob, coef, weight };
  });

  const totalWeight = segments.reduce((sum, s) => sum + s.weight, 0) || 1;

  return (
    <div style={wrap}>
      <div>
        <span
          style={{
            color: ng.colors.textDim,
            fontSize: '0.7rem',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
            display: 'block',
            marginBottom: '0.5rem',
          }}
        >
          Weighted Contribution to Logit
        </span>
        <div style={barTrack} role="img" aria-label="Stacked bar of model contributions to ensemble logit">
          {segments.map(({ cfg, weight }) => {
            const pct = (weight / totalWeight) * 100;
            if (pct < 0.5) return null;
            return (
              <div
                key={cfg.key}
                title={`${cfg.name}: |β|×p = ${weight.toFixed(3)} (${pct.toFixed(0)}%)`}
                style={{
                  width: `${pct}%`,
                  minWidth: pct > 0 ? '4px' : 0,
                  background: `linear-gradient(180deg, ${cfg.color} 0%, ${cfg.color}99 100%)`,
                  transition: ng.transitions.slow,
                  boxShadow: `inset 0 -2px 0 ${cfg.color}40`,
                }}
              />
            );
          })}
        </div>
      </div>

      <div style={legend}>
        {segments.map(({ cfg, prob, coef, weight }) => {
          const pctShare = (weight / totalWeight) * 100;
          return (
            <div
              key={cfg.key}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: '0.75rem',
                flexWrap: 'wrap',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', minWidth: '140px' }}>
                <span
                  style={{
                    width: '10px',
                    height: '10px',
                    borderRadius: '2px',
                    background: cfg.color,
                    flexShrink: 0,
                  }}
                />
                <span style={{ color: ng.colors.textPrimary, fontSize: '0.82rem', fontWeight: 600 }}>
                  {cfg.shortName}
                </span>
                <span
                  style={{
                    fontSize: '0.65rem',
                    fontFamily: ng.fonts.mono,
                    color: ng.colors.textDim,
                    textTransform: 'uppercase',
                    letterSpacing: '0.04em',
                  }}
                >
                  {cfg.typeBadge}
                </span>
              </div>
              <div
                style={{
                  display: 'flex',
                  gap: '1.25rem',
                  flexWrap: 'wrap',
                  fontFamily: ng.fonts.mono,
                  fontSize: '0.75rem',
                }}
              >
                <span style={{ color: cfg.color }}>
                  β={coef >= 0 ? '+' : ''}
                  {coef.toFixed(2)}
                </span>
                <span style={{ color: ng.colors.textSecondary }}>p={prob.toFixed(3)}</span>
                <span style={{ color: ng.colors.textDim }}>
                  |β·p|={weight.toFixed(3)}{' '}
                  <span style={{ opacity: 0.7 }}>({pctShare.toFixed(0)}%)</span>
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
