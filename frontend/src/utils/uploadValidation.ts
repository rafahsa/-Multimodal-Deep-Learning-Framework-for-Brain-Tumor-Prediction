import type { ModalityFiles } from '../components/Upload/ModalityUploadZone';

export const MAX_FILE_MB = 200;
export const MAX_TOTAL_MB = 500;
export const WARN_FILE_MB = 180;

const MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024;
const MAX_TOTAL_BYTES = MAX_TOTAL_MB * 1024 * 1024;
const WARN_FILE_BYTES = WARN_FILE_MB * 1024 * 1024;

export function isValidNiftiFilename(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.endsWith('.nii.gz') || lower.endsWith('.nii');
}

export interface UploadValidationResult {
  valid: boolean;
  totalBytes: number;
  totalError: string | null;
  canSubmit: boolean;
}

export function validateModalityFiles(files: ModalityFiles): UploadValidationResult {
  const entries = Object.values(files).filter((f): f is File => f !== null);
  const totalBytes = entries.reduce((sum, f) => sum + f.size, 0);

  if (totalBytes > MAX_TOTAL_BYTES) {
    return {
      valid: false,
      totalBytes,
      totalError: `Total upload size ${(totalBytes / (1024 * 1024)).toFixed(1)} MB exceeds ${MAX_TOTAL_MB} MB limit`,
      canSubmit: false,
    };
  }

  const allPresent = entries.length === 4;
  return {
    valid: allPresent,
    totalBytes,
    totalError: null,
    canSubmit: allPresent,
  };
}

export type SlotIssue = { type: 'error' | 'warn'; message: string } | null;

export function validateSlotFile(file: File): SlotIssue {
  if (!isValidNiftiFilename(file.name)) {
    return { type: 'error', message: 'Only .nii or .nii.gz files' };
  }
  if (file.size > MAX_FILE_BYTES) {
    return {
      type: 'error',
      message: `File exceeds ${MAX_FILE_MB} MB limit`,
    };
  }
  if (file.size > WARN_FILE_BYTES) {
    return {
      type: 'warn',
      message: `Large file (${(file.size / (1024 * 1024)).toFixed(0)} MB) — may slow upload`,
    };
  }
  return null;
}

export { MAX_FILE_BYTES, WARN_FILE_BYTES, MAX_TOTAL_BYTES };
