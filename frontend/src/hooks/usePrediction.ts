import { useState, useCallback, useRef } from 'react';
import type { PredictionResult } from '../types/prediction';
import type { UploadStage } from '../components/Upload/UploadProgress';
import { predictTumor } from '../services/api';

export type PredictionStatus = 'idle' | 'uploading' | 'processing' | 'success' | 'error';

interface UsePredictionReturn {
  predict: (files: { t1: File; t1ce: File; t2: File; flair: File }, patientLabel?: string) => Promise<void>;
  result: PredictionResult | null;
  status: PredictionStatus;
  stage: UploadStage;
  uploadPercent: number;
  error: string | null;
  errorSuggestion: string | null;
  reset: () => void;
}

export function usePrediction(): UsePredictionReturn {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [status, setStatus] = useState<PredictionStatus>('idle');
  const [stage, setStage] = useState<UploadStage>('uploading');
  const [uploadPercent, setUploadPercent] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [errorSuggestion, setErrorSuggestion] = useState<string | null>(null);
  const abortRef = useRef(false);

  const reset = useCallback(() => {
    setResult(null);
    setStatus('idle');
    setStage('uploading');
    setUploadPercent(0);
    setError(null);
    setErrorSuggestion(null);
    abortRef.current = false;
  }, []);

  const predict = useCallback(
    async (files: { t1: File; t1ce: File; t2: File; flair: File }, patientLabel?: string) => {
      reset();
      abortRef.current = false;
      setStatus('uploading');
      setStage('uploading');

      try {
        const onProgress = (pct: number) => {
          setUploadPercent(pct);
          if (pct >= 100) {
            setStage('preprocessing');
            setStatus('processing');

            setTimeout(() => {
              if (!abortRef.current) setStage('running_models');
            }, 800);
            setTimeout(() => {
              if (!abortRef.current) setStage('calibrating');
            }, 1600);
          }
        };

        const prediction = await predictTumor(files, patientLabel, onProgress);

        if (abortRef.current) return;

        setStage('done');
        setResult(prediction);
        setStatus('success');
      } catch (err: unknown) {
        if (abortRef.current) return;

        setStatus('error');
        setStage('uploading');

        if (err && typeof err === 'object' && 'response' in err) {
          const axiosErr = err as { response?: { data?: { message?: string; suggestion?: string; error?: string } } };
          const data = axiosErr.response?.data;
          setError(data?.message ?? 'Prediction failed. Please try again.');
          setErrorSuggestion(data?.suggestion ?? null);
        } else if (err instanceof Error) {
          if (err.message.includes('Network Error') || err.message.includes('timeout')) {
            setError('Cannot reach the prediction server. Is the backend running?');
            setErrorSuggestion('Start the backend with: uvicorn app.main:app --host 0.0.0.0 --port 8000');
          } else {
            setError(err.message);
          }
        } else {
          setError('An unexpected error occurred.');
        }
      }
    },
    [reset],
  );

  return { predict, result, status, stage, uploadPercent, error, errorSuggestion, reset };
}
