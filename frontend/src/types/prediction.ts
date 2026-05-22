export interface ModelProbabilities {
  resnet: number;
  swinunetr: number;
  mil: number;
}

export interface MetaLearnerCoefficients {
  resnet: number;
  swinunetr: number;
  mil: number;
  intercept: number;
}

export interface ThresholdResult {
  value: number;
  classification: 'HGG' | 'LGG';
  mode_name: string;
}

export interface Thresholds {
  balanced: ThresholdResult;
  high_sensitivity: ThresholdResult;
}

export interface PredictionResult {
  prediction_id: string;
  patient_label: string;
  calibrated_probability: number;
  uncalibrated_probability: number;
  model_probabilities: ModelProbabilities;
  ensemble_logit: number;
  meta_learner_coefficients: MetaLearnerCoefficients;
  thresholds: Thresholds;
  processing_duration_ms: number;
  timestamp: string;
  device_used: string;
}

export type OperatingMode = 'balanced' | 'high_sensitivity';

export interface OperatingModeConfig {
  id: OperatingMode;
  name: string;
  threshold: number;
  description: string;
  precision: number;
  recall: number;
  f1: number;
  expectedFN: number;
  expectedFP: number;
}

export const OPERATING_MODES: Record<OperatingMode, OperatingModeConfig> = {
  balanced: {
    id: 'balanced',
    name: 'Balanced Screening',
    threshold: 0.41,
    description: 'Maximizes F1 score with equal precision and recall (0.9365)',
    precision: 0.9365,
    recall: 0.9365,
    f1: 0.9365,
    expectedFN: 4,
    expectedFP: 4,
  },
  high_sensitivity: {
    id: 'high_sensitivity',
    name: 'High-Sensitivity Triage',
    threshold: 0.38,
    description: 'Minimizes missed HGG cases with recall 0.9524',
    precision: 0.9091,
    recall: 0.9524,
    f1: 0.9302,
    expectedFN: 3,
    expectedFP: 6,
  },
};

export interface HealthStatus {
  status: string;
  models_loaded: boolean;
  device: string;
  gpu_name?: string;
  version: string;
  message?: string;
}

export type ConfidenceLevel = 'High' | 'Medium' | 'Low';

export function getConfidenceLevel(probability: number): ConfidenceLevel {
  if (probability > 0.85 || probability < 0.15) return 'High';
  if ((probability >= 0.65 && probability <= 0.85) || (probability >= 0.15 && probability <= 0.35))
    return 'Medium';
  return 'Low';
}

export function deriveClassification(
  calibratedProbability: number,
  mode: OperatingMode,
): 'HGG' | 'LGG' {
  return calibratedProbability >= OPERATING_MODES[mode].threshold ? 'HGG' : 'LGG';
}
