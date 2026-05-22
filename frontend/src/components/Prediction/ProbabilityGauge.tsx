import { type CSSProperties, useEffect, useState } from 'react';
import { neurograde as ng } from '../../theme/neurograde';

interface ProbabilityGaugeProps {
  probability: number;
  size?: number;
}

function probColor(p: number): string {
  if (p >= 0.65) return ng.colors.accentWarm;
  if (p <= 0.35) return ng.colors.accentCyan;
  return ng.colors.accentViolet;
}

export function ProbabilityGauge({ probability, size = 160 }: ProbabilityGaugeProps) {
  const [animatedP, setAnimatedP] = useState(0);

  useEffect(() => {
    const raf = requestAnimationFrame(() => setAnimatedP(probability));
    return () => cancelAnimationFrame(raf);
  }, [probability]);

  const cx = size / 2;
  const cy = size / 2;
  const r = (size - 20) / 2;
  const strokeWidth = 8;

  const startAngle = 135;
  const endAngle = 405;
  const sweep = endAngle - startAngle;

  function polarToXY(angleDeg: number, radius: number) {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  }

  function arcPath(from: number, to: number, radius: number) {
    const s = polarToXY(from, radius);
    const e = polarToXY(to, radius);
    const largeArc = to - from > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${radius} ${radius} 0 ${largeArc} 1 ${e.x} ${e.y}`;
  }

  const fillAngle = startAngle + sweep * animatedP;
  const color = probColor(probability);

  const pctText = `${(probability * 100).toFixed(1)}%`;
  const label = probability >= 0.5 ? 'P(HGG)' : 'P(LGG)';

  const container: CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '0.4rem',
  };

  return (
    <div style={container}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <filter id="gauge-glow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feFlood floodColor={color} floodOpacity="0.5" />
            <feComposite in2="blur" operator="in" />
            <feMerge>
              <feMergeNode />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background arc */}
        <path
          d={arcPath(startAngle, endAngle, r)}
          fill="none"
          stroke={ng.colors.bgElevated}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Filled arc */}
        {animatedP > 0.001 && (
          <path
            d={arcPath(startAngle, fillAngle, r)}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            filter="url(#gauge-glow)"
            style={{
              transition: 'all 0.8s cubic-bezier(0.22, 1, 0.36, 1)',
            }}
          />
        )}

        {/* Center text */}
        <text
          x={cx}
          y={cy - 6}
          textAnchor="middle"
          fill={color}
          fontFamily={ng.fonts.mono}
          fontSize={size * 0.17}
          fontWeight="600"
        >
          {pctText}
        </text>
        <text
          x={cx}
          y={cy + 14}
          textAnchor="middle"
          fill={ng.colors.textDim}
          fontFamily={ng.fonts.mono}
          fontSize={size * 0.085}
          fontWeight="400"
        >
          {label}
        </text>

        {/* LGG / HGG labels */}
        <text
          x={polarToXY(startAngle, r - 18).x}
          y={polarToXY(startAngle, r - 18).y}
          textAnchor="middle"
          fill={ng.colors.accentCyan}
          fontFamily={ng.fonts.mono}
          fontSize="9"
          fontWeight="500"
        >
          LGG
        </text>
        <text
          x={polarToXY(endAngle, r - 18).x}
          y={polarToXY(endAngle, r - 18).y}
          textAnchor="middle"
          fill={ng.colors.accentWarm}
          fontFamily={ng.fonts.mono}
          fontSize="9"
          fontWeight="500"
        >
          HGG
        </text>
      </svg>
    </div>
  );
}
