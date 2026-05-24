import { AbsoluteFill, Img, staticFile } from 'remotion';
import { useEntrance } from '../hooks/useEntrance';
import { colors, fonts } from '../tokens';

export const Slide4Stat: React.FC = () => {
  const { enter, enterFromRight, float, expandWidth, progress } = useEntrance();
  const doodleY = float(6, 105);
  const pct = Math.round(progress(8, 35) * 70);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(145deg,#101a1c 0%,#141e20 30%,#181e1e 58%,#1c1e1e 100%)',
        fontFamily: fonts.primary,
        overflow: 'hidden',
      }}
    >
      {/* Doodle — large center-right watermark */}
      <div
        style={{
          position: 'absolute',
          right: -80,
          top: 580 + doodleY,
          width: 700,
          height: 700,
          opacity: 0.14,
          zIndex: 0,
          ...enterFromRight(5, 100),
        }}
      >
        <Img src={staticFile('doodles/1-d-04.png')} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
      </div>

      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '100px 72px 100px 72px',
          zIndex: 1,
        }}
      >
        {/* ZONE TOP: Counter + Label + Big stat + body */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div style={{ ...enter(0, 20), fontSize: 28, color: colors.textSecondary, letterSpacing: 2 }}>04 / 10</div>
          <div style={{ ...enter(5, 35), fontFamily: fonts.marker, fontSize: 64, color: colors.green, lineHeight: 1.0 }}>
            TYPICAL INDIAN PLATE
          </div>
          <div
            style={{
              fontFamily: fonts.primary,
              fontWeight: 800,
              fontSize: 220,
              lineHeight: 0.9,
              color: colors.green,
              letterSpacing: '-8px',
            }}
          >
            {pct}%
          </div>
          <div style={{ ...enter(14, 30), fontFamily: fonts.primary, fontWeight: 400, fontSize: 44, color: colors.white, lineHeight: 1.6 }}>
            Roti, rice, dal —{' '}
            <span style={{ color: colors.textSecondary }}>barely 30g of protein per meal.</span>
          </div>
        </div>

        {/* ZONE BOT: Divider + quote + arrow */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div style={{ height: 4, background: colors.green, borderRadius: 2, ...expandWidth(22) }} />
          <div style={{ ...enter(24, 30), fontFamily: fonts.handwritten, fontSize: 54, color: colors.textSecondary, lineHeight: 1.4 }}>
            "Your plate is feeding fat, not muscle."
          </div>
          <div style={{ ...enter(32, 20), color: colors.green, fontSize: 44 }}>→</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
