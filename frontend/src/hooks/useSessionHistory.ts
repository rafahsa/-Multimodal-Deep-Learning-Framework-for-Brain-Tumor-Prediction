import { useState, useCallback } from 'react';
import type { PredictionResult } from '../types/prediction';

const MAX_HISTORY = 50;

export function useSessionHistory() {
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);

  const addPrediction = useCallback((prediction: PredictionResult) => {
    setPredictions((prev) => {
      const withoutDuplicate = prev.filter((p) => p.prediction_id !== prediction.prediction_id);
      const next = [...withoutDuplicate, prediction];
      if (next.length <= MAX_HISTORY) return next;
      return next.slice(next.length - MAX_HISTORY);
    });
  }, []);

  const getPredictions = useCallback((): PredictionResult[] => {
    return [...predictions].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );
  }, [predictions]);

  const clearHistory = useCallback(() => {
    setPredictions([]);
  }, []);

  return {
    addPrediction,
    getPredictions,
    clearHistory,
    count: predictions.length,
  };
}
