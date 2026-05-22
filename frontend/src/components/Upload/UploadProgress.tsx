import { type CSSProperties, useEffect, useState } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

export type UploadStage = 'uploading' | 'preprocessing' | 'running_models' | 'calibrating' | 'done';

interface UploadProgressProps {
  stage: UploadStage;
  uploadPercent?: number;
}

const STAGES: { key: UploadStage; label: string; icon: string }[] = [
  { key: 'uploading', label: 'Uploading scans', icon: '↑' },
  { key: 'preprocessing', label: 'Preprocessing volumes', icon: '◎' },
  { key: 'running_models', label: 'Running 3 model inference', icon: '⧫' },
  { key: 'calibrating', label: 'Calibrating ensemble', icon: '◈' },
  { key: 'done', label: 'Complete', icon: '✓' },
];

function stageIndex(s: UploadStage): number {
  return STAGES.findIndex((st) => st.key === s);
}

const wrap: CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.6rem',
  padding: '1.5rem',
  borderRadius: ng.radii.md,
  background: ng.colors.bgSecondary,
  border: `1px solid ${ng.colors.border}`,
};

function ElapsedTimer() {
  const [secs, setSecs] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setSecs((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, []);

  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return (
    <span
      style={{
        fontFamily: ng.fonts.mono,
        fontSize: '0.76rem',
        color: ng.colors.textDim,
        tabularNums: 'tabular-nums',
        fontVariantNumeric: 'tabular-nums',
      } as CSSProperties}
    >
      {m > 0 ? `${m}m ` : ''}{s}s elapsed
    </span>
  );
}

export function UploadProgress({ stage, uploadPercent }: UploadProgressProps) {
  const activeIdx = stageIndex(stage);
  const isDone = stage === 'done';

  const progress = isDone
    ? 100
    : stage === 'uploading'
      ? (uploadPercent ?? 0) * 0.2
      : 20 + activeIdx * 25;

  return (
    <div style={wrap}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span
          style={{
            fontSize: '0.85rem',
            fontWeight: 600,
            color: isDone ? ng.colors.accentCyan : ng.colors.textPrimary,
          }}
        >
          {isDone ? 'Analysis Complete' : 'Processing...'}
        </span>
        <ElapsedTimer />
      </div>

      {/* Progress bar */}
      <div
        style={{
          height: '4px',
          borderRadius: '2px',
          background: ng.colors.bgElevated,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${progress}%`,
            borderRadius: '2px',
            background: isDone
              ? ng.colors.accentCyan
              : `linear-gradient(90deg, ${ng.colors.accentCyan}, ${ng.colors.accentViolet})`,
            transition: 'width 0.6s cubic-bezier(0.22, 1, 0.36, 1)',
          }}
        />
      </div>

      {/* Stage steps */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', marginTop: '0.4rem' }}>
        {STAGES.map((st, i) => {
          const isActive = i === activeIdx;
          const isPast = i < activeIdx;

          return (
            <div
              key={st.key}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.6rem',
                padding: '0.3rem 0.4rem',
                borderRadius: ng.radii.sm,
                background: isActive ? `${ng.colors.accentCyan}08` : 'transparent',
                transition: ng.transitions.fast,
              }}
            >
              <span
                style={{
                  width: '18px',
                  height: '18px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '0.7rem',
                  fontWeight: 600,
                  borderRadius: '50%',
                  flexShrink: 0,
                  ...(isPast
                    ? {
                        background: `${ng.colors.accentCyan}20`,
                        color: ng.colors.accentCyan,
                      }
                    : isActive
                      ? {
                          background: ng.colors.accentCyan,
                          color: ng.colors.bgPrimary,
                          boxShadow: `0 0 10px ${ng.colors.accentCyanGlow}`,
                        }
                      : {
                          background: ng.colors.bgElevated,
                          color: ng.colors.textDim,
                        }),
                }}
              >
                {isPast ? '✓' : st.icon}
              </span>
              <span
                style={{
                  fontSize: '0.78rem',
                  fontWeight: isActive ? 600 : 400,
                  color: isPast
                    ? ng.colors.textDim
                    : isActive
                      ? ng.colors.textPrimary
                      : ng.colors.textDim,
                  textDecoration: isPast ? 'line-through' : 'none',
                }}
              >
                {st.label}
              </span>
              {isActive && !isDone && (
                <span
                  style={{
                    marginLeft: 'auto',
                    width: '6px',
                    height: '6px',
                    borderRadius: '50%',
                    background: ng.colors.accentCyan,
                    animation: 'neuro-pulse 1.5s ease-in-out infinite',
                  }}
                />
              )}
            </div>
          );
        })}
      </div>

      <style>{`
        @keyframes neuro-pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.4; transform: scale(0.7); }
        }
      `}</style>
    </div>
  );
}
