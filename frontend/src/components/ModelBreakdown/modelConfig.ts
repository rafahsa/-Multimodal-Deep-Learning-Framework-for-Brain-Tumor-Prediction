import { neurograde as ng } from '../../theme/neurograde';
import type { ModelProbabilities, MetaLearnerCoefficients } from '../../types/prediction';

export type ModelKey = 'resnet' | 'swinunetr' | 'mil';

export interface ModelDisplayConfig {
  key: ModelKey;
  name: string;
  shortName: string;
  architecture: string;
  typeBadge: string;
  color: string;
  glow: string;
  probabilityKey: keyof ModelProbabilities;
  coefficientKey: keyof Omit<MetaLearnerCoefficients, 'intercept'>;
}

export const MODEL_CONFIGS: ModelDisplayConfig[] = [
  {
    key: 'swinunetr',
    name: 'SwinUNETR-3D',
    shortName: 'Swin',
    architecture: 'Transformer encoder',
    typeBadge: 'Transformer',
    color: ng.colors.accentCyan,
    glow: ng.colors.accentCyanGlow,
    probabilityKey: 'swinunetr',
    coefficientKey: 'swinunetr',
  },
  {
    key: 'mil',
    name: 'DualStreamMIL-3D',
    shortName: 'MIL',
    architecture: 'Multiple-instance learning',
    typeBadge: 'MIL',
    color: ng.colors.accentViolet,
    glow: ng.colors.accentVioletGlow,
    probabilityKey: 'mil',
    coefficientKey: 'mil',
  },
  {
    key: 'resnet',
    name: 'ResNet50-3D',
    shortName: 'ResNet',
    architecture: '3D convolutional network',
    typeBadge: 'CNN',
    color: ng.colors.accentWarm,
    glow: ng.colors.accentWarmGlow,
    probabilityKey: 'resnet',
    coefficientKey: 'resnet',
  },
];

export function weightedContribution(coefficient: number, probability: number): number {
  return Math.abs(coefficient * probability);
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
