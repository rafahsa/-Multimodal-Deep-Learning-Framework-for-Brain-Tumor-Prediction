import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';
import type { PredictionResult } from '../../types/prediction';
import { MODEL_CONFIGS, sigmoid } from './modelConfig';

interface EnsembleFormulaProps {
  result: PredictionResult;
}

const panel: CSSProperties = {
  padding: '1.15rem 1.25rem',
  borderRadius: ng.radii.md,
  background: `linear-gradient(135deg, ${ng.colors.bgSecondary} 0%, rgba(17, 22, 39, 0.95) 100%)`,
  border: `1px solid ${ng.colors.border}`,
  position: 'relative',
  overflow: 'hidden',
};

const mono: CSSProperties = {
  fontFamily: ng.fonts.mono,
  fontSize: '0.82rem',
  lineHeight: 1.75,
  wordBreak: 'break-word',
};

function Term({
  coef,
  prob,
  color,
  label,
}: {
  coef: number;
  prob: number;
  color: string;
  label: string;
}) {
  const sign = coef >= 0 ? '+' : '−';
  const absCoef = Math.abs(coef).toFixed(2);
  return (
    <span>
      <span style={{ color: ng.colors.textDim }}> {sign} </span>
      <span style={{ color, fontWeight: 600 }}>{absCoef}</span>
      <span style={{ color: ng.colors.textDim }}>·</span>
      <span style={{ color }}>p<sub style={{ fontSize: '0.65em' }}>{label}</sub></span>
      <span style={{ color: ng.colors.textSecondary }}> ({prob.toFixed(3)})</span>
    </span>
  );
}

export function EnsembleFormula({ result }: EnsembleFormulaProps) {
  const { model_probabilities: probs, meta_learner_coefficients: coefs, ensemble_logit: logit } =
    result;
  const computedSigmoid = sigmoid(logit);
  const intercept = coefs.intercept;

  const swin = MODEL_CONFIGS.find((c) => c.key === 'swinunetr')!;
  const mil = MODEL_CONFIGS.find((c) => c.key === 'mil')!;
  const resnet = MODEL_CONFIGS.find((c) => c.key === 'resnet')!;

  return (
    <div style={panel}>
      <div
        style={{
          position: 'absolute',
          top: 0,
          right: 0,
          width: '120px',
          height: '120px',
          background: `radial-gradient(circle at 100% 0%, ${ng.colors.accentVioletGlow} 0%, transparent 70%)`,
          pointerEvents: 'none',
        }}
      />

      <span
        style={{
          color: ng.colors.textDim,
          fontSize: '0.7rem',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          display: 'block',
          marginBottom: '0.75rem',
        }}
      >
        Meta-Learner Ensemble Formula
      </span>

      <div style={{ ...mono, color: ng.colors.textPrimary, marginBottom: '1rem' }}>
        <span style={{ color: ng.colors.textSecondary }}>P(HGG)</span>
        <span style={{ color: ng.colors.textDim }}> = σ(</span>
        <span>
          <span style={{ color: swin.color, fontWeight: 600 }}>
            {Math.abs(coefs.swinunetr).toFixed(2)}
          </span>
          <span style={{ color: ng.colors.textDim }}>·</span>
          <span style={{ color: swin.color }}>
            p<sub style={{ fontSize: '0.65em' }}>swin</sub>
          </span>
          <span style={{ color: ng.colors.textSecondary }}> ({probs.swinunetr.toFixed(3)})</span>
        </span>
        <Term coef={coefs.mil} prob={probs.mil} color={mil.color} label="mil" />
        <Term coef={coefs.resnet} prob={probs.resnet} color={resnet.color} label="res" />
        <span style={{ color: ng.colors.textDim }}>
          {' '}
          {intercept >= 0 ? '+' : '−'} {Math.abs(intercept).toFixed(2)}
        </span>
        <span style={{ color: ng.colors.textDim }}> )</span>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
          gap: '0.75rem',
          paddingTop: '0.85rem',
          borderTop: `1px solid ${ng.colors.border}`,
        }}
      >
        <div>
          <span
            style={{
              color: ng.colors.textDim,
              fontSize: '0.68rem',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            Ensemble Logit (z)
          </span>
          <div
            style={{
              fontFamily: ng.fonts.mono,
              fontSize: '1.1rem',
              fontWeight: 700,
              color: ng.colors.accentCyan,
              marginTop: '0.2rem',
            }}
          >
            {logit >= 0 ? '+' : ''}
            {logit.toFixed(4)}
          </div>
        </div>
        <div>
          <span
            style={{
              color: ng.colors.textDim,
              fontSize: '0.68rem',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            σ(z) — Uncalibrated
          </span>
          <div
            style={{
              fontFamily: ng.fonts.mono,
              fontSize: '1.1rem',
              fontWeight: 700,
              color: ng.colors.textPrimary,
              marginTop: '0.2rem',
            }}
          >
            {(computedSigmoid * 100).toFixed(2)}%
          </div>
        </div>
        <div>
          <span
            style={{
              color: ng.colors.textDim,
              fontSize: '0.68rem',
              textTransform: 'uppercase',
              letterSpacing: '0.06em',
            }}
          >
            Platt-Calibrated P(HGG)
          </span>
          <div
            style={{
              fontFamily: ng.fonts.mono,
              fontSize: '1.1rem',
              fontWeight: 700,
              color: ng.colors.accentWarm,
              marginTop: '0.2rem',
            }}
          >
            {(result.calibrated_probability * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      <p
        style={{
          margin: '0.85rem 0 0',
          color: ng.colors.textDim,
          fontSize: '0.72rem',
          lineHeight: 1.5,
        }}
      >
        Coefficients β are fixed meta-learner weights from training. Values in parentheses are
        per-model P(HGG) for this scan. Platt scaling maps σ(z) to the calibrated probability shown
        in the result card.
      </p>
    </div>
  );
}
