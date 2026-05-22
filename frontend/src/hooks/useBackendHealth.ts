import { useState, useEffect, useCallback } from 'react';
import { checkHealth } from '../services/api';

export type BackendHealthState = 'connecting' | 'healthy' | 'unreachable';

const POLL_MS = 10_000;

export function useBackendHealth() {
  const [state, setState] = useState<BackendHealthState>('connecting');

  const probe = useCallback(async () => {
    try {
      const health = await checkHealth();
      if (health.status === 'healthy' && health.models_loaded) {
        setState('healthy');
        return true;
      }
      setState('connecting');
      return false;
    } catch {
      setState('unreachable');
      return false;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    let intervalId: ReturnType<typeof setInterval> | undefined;

    const run = async () => {
      const ok = await probe();
      if (cancelled || ok) return;
      intervalId = setInterval(() => {
        void probe();
      }, POLL_MS);
    };

    void run();

    return () => {
      cancelled = true;
      if (intervalId) clearInterval(intervalId);
    };
  }, [probe]);

  return { state, retry: probe };
}
