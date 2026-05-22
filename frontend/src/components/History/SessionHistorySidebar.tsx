import { type CSSProperties } from 'react';
import { neurograde as ng } from '../../theme/neurograde';
import type { PredictionResult } from '../../types/prediction';
import { HistoryItem } from './HistoryItem';

interface SessionHistorySidebarProps {
  predictions: PredictionResult[];
  selectedId: string | null;
  onSelect: (predictionId: string) => void;
  onClear: () => void;
  mobileOpen: boolean;
  onMobileToggle: () => void;
}

const sidebar: CSSProperties = {
  width: '280px',
  flexShrink: 0,
  position: 'sticky',
  top: '5.5rem',
  alignSelf: 'flex-start',
  maxHeight: 'calc(100vh - 7rem)',
  display: 'flex',
  flexDirection: 'column',
  borderRadius: ng.radii.lg,
  border: `1px solid ${ng.colors.border}`,
  background: ng.colors.bgGlass,
  backdropFilter: ng.glass.backdropFilter,
  overflow: 'hidden',
};

const header: CSSProperties = {
  padding: '1rem 1.1rem 0.75rem',
  borderBottom: `1px solid ${ng.colors.border}`,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '0.5rem',
};

const list: CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '0.75rem',
  display: 'flex',
  flexDirection: 'column',
  gap: '0.55rem',
};

export function SessionHistorySidebar({
  predictions,
  selectedId,
  onSelect,
  onClear,
  mobileOpen,
  onMobileToggle,
}: SessionHistorySidebarProps) {
  if (predictions.length === 0) return null;

  return (
    <div className="session-history-wrap">
      <button
        type="button"
        className="history-mobile-toggle"
        onClick={onMobileToggle}
        aria-expanded={mobileOpen}
        style={{
          width: '100%',
          marginBottom: '0.75rem',
          padding: '0.75rem 1rem',
          borderRadius: ng.radii.md,
          border: `1px solid ${ng.colors.borderActive}`,
          background: ng.colors.bgElevated,
          color: ng.colors.textPrimary,
          fontFamily: ng.fonts.body,
          fontSize: '0.85rem',
          fontWeight: 600,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
        }}
      >
        Session History ({predictions.length})
        <span style={{ marginLeft: 'auto', color: ng.colors.accentCyan }}>
          {mobileOpen ? '▾' : '▸'}
        </span>
      </button>

      <aside
        className={`session-history-sidebar${mobileOpen ? ' session-history-sidebar--open' : ''}`}
        style={sidebar}
        aria-label="Session prediction history"
      >
        <div style={header}>
          <div>
            <span
              style={{
                color: ng.colors.textDim,
                fontSize: '0.68rem',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                display: 'block',
              }}
            >
              Session History
            </span>
            <span
              style={{
                color: ng.colors.textSecondary,
                fontSize: '0.75rem',
                marginTop: '0.2rem',
                display: 'block',
              }}
            >
              {predictions.length} scan{predictions.length === 1 ? '' : 's'} · lost on refresh
            </span>
          </div>
          <button
            type="button"
            onClick={onClear}
            title="Clear session history"
            style={{
              background: 'transparent',
              border: `1px solid ${ng.colors.border}`,
              borderRadius: ng.radii.pill,
              color: ng.colors.textDim,
              fontSize: '0.65rem',
              fontFamily: ng.fonts.mono,
              padding: '0.25rem 0.5rem',
              cursor: 'pointer',
              textTransform: 'uppercase',
              letterSpacing: '0.04em',
            }}
          >
            Clear
          </button>
        </div>

        <div style={list}>
          {predictions.map((p) => (
            <HistoryItem
              key={p.prediction_id}
              prediction={p}
              isActive={selectedId === p.prediction_id}
              onSelect={() => onSelect(p.prediction_id)}
            />
          ))}
        </div>
      </aside>

      <style>{`
        @media (min-width: 1101px) {
          .session-history-wrap .history-mobile-toggle {
            display: none;
          }
          .session-history-wrap .session-history-sidebar {
            display: flex;
          }
        }
        @media (max-width: 1100px) {
          .session-history-wrap {
            width: 100%;
          }
          .session-history-wrap .session-history-sidebar {
            display: none;
            width: 100% !important;
            max-height: 320px;
            position: static !important;
          }
          .session-history-wrap .session-history-sidebar--open {
            display: flex !important;
          }
        }
      `}</style>
    </div>
  );
}
