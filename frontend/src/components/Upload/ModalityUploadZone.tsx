import { type CSSProperties, useState, useRef, useCallback, type DragEvent } from 'react';
import { neurograde as ng } from '../../theme/neurograde';
import {
  validateModalityFiles,
  validateSlotFile,
  type SlotIssue,
} from '../../utils/uploadValidation';

export type Modality = 't1' | 't1ce' | 't2' | 'flair';

export interface ModalityFiles {
  t1: File | null;
  t1ce: File | null;
  t2: File | null;
  flair: File | null;
}

interface ModalityUploadZoneProps {
  files: ModalityFiles;
  onFileChange: (modality: Modality, file: File | null) => void;
  disabled?: boolean;
}

const MODALITY_META: Record<Modality, { label: string; full: string; color: string }> = {
  t1: { label: 'T1', full: 'T1-weighted', color: ng.colors.accentCyan },
  t1ce: { label: 'T1ce', full: 'T1 Contrast-Enhanced', color: ng.colors.accentViolet },
  t2: { label: 'T2', full: 'T2-weighted', color: ng.colors.accentWarm },
  flair: { label: 'FLAIR', full: 'Fluid-Attenuated IR', color: '#60a5fa' },
};

const MODALITIES: Modality[] = ['t1', 't1ce', 't2', 'flair'];

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const statusBar: CSSProperties = {
  marginTop: '1rem',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '0.5rem',
  fontFamily: ng.fonts.mono,
  fontSize: '0.78rem',
  padding: '0.7rem 1rem',
  borderRadius: ng.radii.sm,
  border: `1px solid ${ng.colors.border}`,
  background: ng.colors.bgSecondary,
};

function SlotIcon({ filled, color }: { filled: boolean; color: string }) {
  if (filled) {
    return (
      <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
        <circle cx="16" cy="16" r="14" stroke={color} strokeWidth="1.5" fill={`${color}15`} />
        <path d="M10 16.5l4 4 8.5-8.5" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none" style={{ opacity: 0.4 }}>
      <rect x="4" y="4" width="24" height="24" rx="6" stroke={ng.colors.textDim} strokeWidth="1.5" strokeDasharray="4 3" />
      <path d="M16 11v10M11 16h10" stroke={ng.colors.textDim} strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}

function UploadSlot({
  modality,
  file,
  onFile,
  disabled,
}: {
  modality: Modality;
  file: File | null;
  onFile: (f: File | null) => void;
  disabled?: boolean;
}) {
  const [dragOver, setDragOver] = useState(false);
  const [issue, setIssue] = useState<SlotIssue>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const meta = MODALITY_META[modality];

  const handleFile = useCallback(
    (f: File) => {
      const slotIssue = validateSlotFile(f);
      if (slotIssue?.type === 'error') {
        setIssue(slotIssue);
        return;
      }
      setIssue(slotIssue);
      onFile(f);
    },
    [onFile],
  );

  const onDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [disabled, handleFile],
  );

  const hasError = issue?.type === 'error';
  const hasWarn = issue?.type === 'warn';

  const slot: CSSProperties = {
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.6rem',
    padding: '1.4rem 1rem',
    minHeight: '140px',
    borderRadius: ng.radii.md,
    border: `1.5px ${file ? 'solid' : 'dashed'} ${
      dragOver
        ? meta.color
        : file
          ? `${meta.color}80`
          : hasError
            ? `${ng.colors.accentWarm}60`
            : hasWarn
              ? `${ng.colors.accentViolet}50`
              : ng.colors.border
    }`,
    background: dragOver
      ? `${meta.color}08`
      : file
        ? `${meta.color}06`
        : ng.colors.bgSecondary,
    cursor: disabled ? 'not-allowed' : 'pointer',
    transition: ng.transitions.default,
    opacity: disabled ? 0.5 : 1,
    overflow: 'hidden',
  };

  const badge: CSSProperties = {
    position: 'absolute',
    top: '0.6rem',
    left: '0.6rem',
    fontFamily: ng.fonts.mono,
    fontSize: '0.68rem',
    fontWeight: 600,
    letterSpacing: '0.06em',
    color: meta.color,
    background: `${meta.color}18`,
    padding: '0.2rem 0.5rem',
    borderRadius: ng.radii.sm,
    textTransform: 'uppercase' as const,
  };

  return (
    <div
      style={slot}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => {
        if (!disabled) inputRef.current?.click();
      }}
    >
      <span style={badge}>{meta.label}</span>
      <input
        ref={inputRef}
        type="file"
        accept=".nii,.nii.gz,application/gzip"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) handleFile(f);
          e.target.value = '';
        }}
      />

      <SlotIcon filled={!!file} color={meta.color} />

      {file ? (
        <>
          <span
            style={{
              color: ng.colors.textPrimary,
              fontSize: '0.82rem',
              fontWeight: 500,
              textAlign: 'center',
              wordBreak: 'break-all',
              maxWidth: '90%',
            }}
          >
            {file.name}
          </span>
          <span
            style={{
              color: ng.colors.textDim,
              fontSize: '0.72rem',
              fontFamily: ng.fonts.mono,
            }}
          >
            {formatSize(file.size)}
          </span>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onFile(null);
              setIssue(null);
            }}
            style={{
              position: 'absolute',
              top: '0.5rem',
              right: '0.5rem',
              background: 'none',
              border: 'none',
              color: ng.colors.textDim,
              cursor: 'pointer',
              fontSize: '1rem',
              lineHeight: 1,
              padding: '0.2rem',
            }}
            aria-label={`Remove ${meta.label} file`}
          >
            &times;
          </button>
        </>
      ) : (
        <>
          <span style={{ color: ng.colors.textSecondary, fontSize: '0.82rem' }}>
            {meta.full}
          </span>
          <span style={{ color: ng.colors.textDim, fontSize: '0.72rem' }}>
            Drag & drop or click to browse
          </span>
        </>
      )}

      {issue && (
        <span
          style={{
            color: issue.type === 'error' ? ng.colors.accentWarm : ng.colors.accentViolet,
            fontSize: '0.68rem',
            fontWeight: 500,
            position: 'absolute',
            bottom: '0.4rem',
            textAlign: 'center',
            padding: '0 0.5rem',
            maxWidth: '100%',
          }}
        >
          {issue.message}
        </span>
      )}
    </div>
  );
}

export function ModalityUploadZone({ files, onFileChange, disabled }: ModalityUploadZoneProps) {
  const readyCount = MODALITIES.filter((m) => files[m] !== null).length;
  const validation = validateModalityFiles(files);
  const allReady = validation.canSubmit;

  const totalMb = (validation.totalBytes / (1024 * 1024)).toFixed(1);

  return (
    <div>
      <div className="upload-modality-grid">
        {MODALITIES.map((m) => (
          <UploadSlot
            key={m}
            modality={m}
            file={files[m]}
            onFile={(f) => onFileChange(m, f)}
            disabled={disabled}
          />
        ))}
      </div>

      {validation.totalError && (
        <div
          style={{
            ...statusBar,
            marginTop: '0.75rem',
            color: ng.colors.accentWarm,
            borderColor: `${ng.colors.accentWarm}40`,
          }}
        >
          {validation.totalError}
        </div>
      )}

      <div
        style={{
          ...statusBar,
          color: allReady ? ng.colors.accentCyan : ng.colors.textDim,
          borderColor: allReady ? `${ng.colors.accentCyan}30` : ng.colors.border,
        }}
      >
        {allReady ? (
          <>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <circle cx="7" cy="7" r="6" stroke={ng.colors.accentCyan} strokeWidth="1.5" />
              <path d="M4 7.2l2 2 4-4" stroke={ng.colors.accentCyan} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            All 4 modalities ready · {totalMb} MB total
          </>
        ) : (
          <>
            {readyCount} / 4 modalities uploaded — all 4 required
            {validation.totalBytes > 0 && ` · ${totalMb} MB`}
          </>
        )}
      </div>
    </div>
  );
}

export { validateModalityFiles } from '../../utils/uploadValidation';
