import {
  type CSSProperties,
  useState,
  useCallback,
  useEffect,
  useRef,
  useMemo,
} from 'react';
import { neurograde as ng } from '../theme/neurograde';
import {
  Button,
  GlassPanel,
  ErrorBanner,
  PrivacyDisclaimer,
  Reveal,
  PredictionSkeleton,
} from '../components/common';
import { ModalityUploadZone, UploadProgress, validateModalityFiles } from '../components/Upload';
import type { ModalityFiles, Modality } from '../components/Upload';
import { ResultCard } from '../components/Prediction';
import { ModelDetailPanel } from '../components/ModelBreakdown';
import { ModeSelector } from '../components/OperatingMode';
import { SessionHistorySidebar } from '../components/History';
import { usePrediction } from '../hooks/usePrediction';
import { useSessionHistory } from '../hooks/useSessionHistory';
import type { OperatingMode } from '../types/prediction';

const mainColumn: CSSProperties = {
  flex: 1,
  minWidth: 0,
};

const hero: CSSProperties = {
  textAlign: 'center',
  paddingTop: '3rem',
  paddingBottom: '2.5rem',
};

const heading: CSSProperties = {
  fontFamily: ng.fonts.display,
  fontWeight: 700,
  letterSpacing: '-0.02em',
  lineHeight: 1.2,
  marginBottom: '0.8rem',
};

const sub: CSSProperties = {
  color: ng.colors.textSecondary,
  fontSize: '1.05rem',
  maxWidth: '580px',
  margin: '0 auto 0',
  lineHeight: 1.6,
};

interface PredictPageProps {
  backendReady?: boolean;
}

const uploadSection: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '1rem',
};

const patientRow: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.8rem',
};

const patientInput: CSSProperties = {
  flex: 1,
  padding: '0.7rem 1rem',
  borderRadius: ng.radii.sm,
  border: `1px solid ${ng.colors.border}`,
  background: ng.colors.bgSecondary,
  color: ng.colors.textPrimary,
  fontSize: '0.85rem',
  fontFamily: ng.fonts.body,
  outline: 'none',
  transition: ng.transitions.fast,
};

const predictBtnWrap: CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  marginTop: '0.5rem',
};

const resultSection = (highlighted: boolean): CSSProperties => ({
  marginTop: '2rem',
  animation: 'neuro-fade-up 0.5s cubic-bezier(0.22, 1, 0.36, 1)',
  borderRadius: ng.radii.lg,
  outline: highlighted ? `2px solid ${ng.colors.borderActive}` : '2px solid transparent',
  outlineOffset: '4px',
  transition: ng.transitions.default,
});

const modeSection: CSSProperties = {
  marginBottom: '1rem',
};

const modeLabel: CSSProperties = {
  color: ng.colors.textDim,
  fontSize: '0.72rem',
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: '0.08em',
  marginBottom: '0.65rem',
  display: 'block',
};

const newScanBtn: CSSProperties = {
  display: 'flex',
  justifyContent: 'center',
  marginTop: '1.2rem',
};

export function PredictPage({ backendReady = true }: PredictPageProps) {
  const [files, setFiles] = useState<ModalityFiles>({
    t1: null,
    t1ce: null,
    t2: null,
    flair: null,
  });
  const [patientLabel, setPatientLabel] = useState('');
  const [operatingMode, setOperatingMode] = useState<OperatingMode>('balanced');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [historyMobileOpen, setHistoryMobileOpen] = useState(false);
  const [highlightResult, setHighlightResult] = useState(false);

  const resultAnchorRef = useRef<HTMLDivElement>(null);
  const lastAddedIdRef = useRef<string | null>(null);

  const { addPrediction, getPredictions, clearHistory } = useSessionHistory();
  const historyList = getPredictions();

  const { predict, result, status, stage, uploadPercent, error, errorSuggestion, reset } =
    usePrediction();

  const uploadValidation = useMemo(() => validateModalityFiles(files), [files]);
  const isProcessing = status === 'uploading' || status === 'processing';
  const canPredict =
    uploadValidation.canSubmit && !uploadValidation.totalError && !isProcessing && backendReady;

  const displayedResult = useMemo(() => {
    if (selectedId) {
      return historyList.find((p) => p.prediction_id === selectedId) ?? null;
    }
    if (status === 'success' && result) return result;
    return null;
  }, [selectedId, historyList, status, result]);

  const showSkeleton = isProcessing && displayedResult === null;
  const showUpload = !displayedResult && status !== 'success' && !isProcessing;
  const showResults = displayedResult !== null && !showSkeleton;

  useEffect(() => {
    if (status === 'success' && result && result.prediction_id !== lastAddedIdRef.current) {
      addPrediction(result);
      lastAddedIdRef.current = result.prediction_id;
      setSelectedId(result.prediction_id);
      setHistoryMobileOpen(false);
    }
  }, [status, result, addPrediction]);

  const scrollToResult = useCallback(() => {
    resultAnchorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setHighlightResult(true);
    const t = window.setTimeout(() => setHighlightResult(false), 1200);
    return () => window.clearTimeout(t);
  }, []);

  const handleFileChange = useCallback((modality: Modality, file: File | null) => {
    setFiles((prev) => ({ ...prev, [modality]: file }));
  }, []);

  const handlePredict = useCallback(() => {
    if (!files.t1 || !files.t1ce || !files.t2 || !files.flair) return;
    predict(
      { t1: files.t1, t1ce: files.t1ce, t2: files.t2, flair: files.flair },
      patientLabel || undefined,
    );
  }, [files, patientLabel, predict]);

  const handleNewScan = useCallback(() => {
    reset();
    setSelectedId(null);
    setFiles({ t1: null, t1ce: null, t2: null, flair: null });
    setPatientLabel('');
  }, [reset]);

  const handleHistorySelect = useCallback(
    (predictionId: string) => {
      setSelectedId(predictionId);
      setHistoryMobileOpen(false);
      requestAnimationFrame(() => scrollToResult());
    },
    [scrollToResult],
  );

  const handleClearHistory = useCallback(() => {
    clearHistory();
    setSelectedId(null);
    if (status === 'success') {
      reset();
    }
  }, [clearHistory, status, reset]);

  return (
    <div>
      <Reveal className="predict-hero" style={hero}>
        <h1 className="predict-hero-title" style={heading}>
          Brain Tumor{' '}
          <em style={{ color: ng.colors.accentCyan, fontStyle: 'italic' }}>Classification</em>
        </h1>
        <p className="predict-hero-sub" style={sub}>
          Upload 4 NIfTI modality scans to receive a calibrated HGG / LGG prediction powered by a
          three-model ensemble with Platt scaling.
        </p>
      </Reveal>

      <div className="predict-page-layout">
        <div className="predict-main-column" style={mainColumn}>
          {showUpload && (
            <Reveal delay={1} style={uploadSection}>
              <PrivacyDisclaimer />

              <div style={modeSection}>
                <span style={modeLabel}>Clinical Operating Mode</span>
                <ModeSelector
                  selected={operatingMode}
                  onChange={setOperatingMode}
                  disabled={isProcessing}
                />
              </div>

              <GlassPanel>
                <ModalityUploadZone
                  files={files}
                  onFileChange={handleFileChange}
                  disabled={isProcessing}
                />

                <div className="patient-label-row" style={{ ...patientRow, marginTop: '1rem' }}>
                  <label
                    style={{
                      color: ng.colors.textDim,
                      fontSize: '0.78rem',
                      fontWeight: 500,
                      whiteSpace: 'nowrap',
                      textTransform: 'uppercase' as const,
                      letterSpacing: '0.06em',
                    }}
                  >
                    Patient Label
                  </label>
                  <input
                    type="text"
                    value={patientLabel}
                    onChange={(e) => setPatientLabel(e.target.value)}
                    placeholder="Optional — auto-derived from filename"
                    style={patientInput}
                    disabled={isProcessing}
                    onFocus={(e) => {
                      e.currentTarget.style.borderColor = ng.colors.borderActive;
                    }}
                    onBlur={(e) => {
                      e.currentTarget.style.borderColor = ng.colors.border;
                    }}
                  />
                </div>

                <div style={predictBtnWrap}>
                  <Button
                    variant="primary"
                    disabled={!canPredict}
                    onClick={handlePredict}
                    title={!backendReady ? 'Waiting for backend to finish loading models' : undefined}
                    style={{
                      padding: '0.95rem 3rem',
                      fontSize: '0.92rem',
                      fontWeight: 700,
                      letterSpacing: '0.02em',
                    }}
                  >
                    {isProcessing ? (
                      <>
                        <svg
                          width="16"
                          height="16"
                          viewBox="0 0 16 16"
                          fill="none"
                          style={{ animation: 'neuro-spin 0.7s linear infinite' }}
                        >
                          <circle
                            cx="8"
                            cy="8"
                            r="6"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeDasharray="28"
                            strokeDashoffset="8"
                            strokeLinecap="round"
                          />
                        </svg>
                        Processing...
                      </>
                    ) : (
                      <>
                        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                          <path
                            d="M8 2C4.69 2 2 4.69 2 8s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm0 10.8A4.8 4.8 0 013.2 8 4.8 4.8 0 018 3.2 4.8 4.8 0 0112.8 8 4.8 4.8 0 018 12.8z"
                            fill="currentColor"
                          />
                          <path d="M7 5.5l4 2.5-4 2.5V5.5z" fill="currentColor" />
                        </svg>
                        Analyze Scan
                      </>
                    )}
                  </Button>
                </div>
              </GlassPanel>

              {status === 'error' && error && (
                <ErrorBanner
                  message={error}
                  suggestion={errorSuggestion ?? undefined}
                  onDismiss={reset}
                />
              )}
            </Reveal>
          )}

          {isProcessing && (
            <Reveal delay={1}>
              <UploadProgress stage={stage} uploadPercent={uploadPercent} />
            </Reveal>
          )}

          {showSkeleton && (
            <Reveal delay={2}>
              <PredictionSkeleton />
            </Reveal>
          )}

          {showResults && displayedResult && (
            <Reveal
              delay={1}
              style={resultSection(highlightResult)}
            >
            <div ref={resultAnchorRef} id="prediction-result">
              <div style={modeSection}>
                <span style={modeLabel}>Clinical Operating Mode</span>
                <ModeSelector
                  selected={operatingMode}
                  onChange={setOperatingMode}
                  compact
                />
              </div>

              <ResultCard result={displayedResult} operatingMode={operatingMode} />
              <ModelDetailPanel result={displayedResult} />

              <div style={newScanBtn}>
                <Button variant="outline" onClick={handleNewScan}>
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path
                      d="M2 7a5 5 0 019.33-2.5M12 7a5 5 0 01-9.33 2.5"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                    />
                    <path
                      d="M11.5 1.5v3h-3"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M2.5 12.5v-3h3"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  Upload New Scan
                </Button>
              </div>
            </div>
            </Reveal>
          )}
        </div>

        {historyList.length > 0 && (
          <SessionHistorySidebar
            predictions={historyList}
            selectedId={selectedId}
            onSelect={handleHistorySelect}
            onClear={handleClearHistory}
            mobileOpen={historyMobileOpen}
            onMobileToggle={() => setHistoryMobileOpen((v) => !v)}
          />
        )}
      </div>

      <style>{`
        @keyframes neuro-fade-up {
          from { opacity: 0; transform: translateY(16px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes neuro-spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
