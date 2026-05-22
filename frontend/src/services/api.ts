import axios, { isAxiosError } from 'axios';
import type { PredictionResult, HealthStatus } from '../types/prediction';

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:8000',
  timeout: 300_000,
});

export async function predictTumor(
  files: { t1: File; t1ce: File; t2: File; flair: File },
  patientLabel?: string,
  onUploadProgress?: (pct: number) => void,
): Promise<PredictionResult> {
  const form = new FormData();
  form.append('t1', files.t1);
  form.append('t1ce', files.t1ce);
  form.append('t2', files.t2);
  form.append('flair', files.flair);
  if (patientLabel) form.append('patient_label', patientLabel);

  const { data } = await client.post<PredictionResult>('/api/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress(e) {
      if (e.total && onUploadProgress) {
        onUploadProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });
  return data;
}

export async function checkHealth(): Promise<HealthStatus> {
  try {
    const { data } = await client.get<HealthStatus>('/api/health', { timeout: 8_000 });
    return data;
  } catch (err: unknown) {
    if (isAxiosError(err) && err.response?.status === 503 && err.response.data) {
      return err.response.data as HealthStatus;
    }
    throw err;
  }
}

export default client;
