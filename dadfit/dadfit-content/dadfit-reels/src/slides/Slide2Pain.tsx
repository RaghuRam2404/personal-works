import { AbsoluteFill, Img, staticFile } from 'remotion';
import { useEntrance } from '../hooks/useEntrance';
import { colors, fonts } from '../tokens';

export const Slide2Pain: React.FC = () => {
  const { enter, enterFromRight, float, expandWidth } = useEntrance();
  const doodleY = float(4, 90);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(145deg,#1c1212 0%,#201616 35%,#1f1b1b 65%,#1E1E1E 100%)',
        fontFamily: fonts.primary,
        overflow: 'hidden',
      }}
    >
      <AbsoluteFill
        style={{
          background: 'radial-gradient(ellipse at 20% 30%, rgba(255,107,107,0.04) 0%, transparent 55%)',
          pointerEvents: 'none',
        }}
      />

      {/* Doodle — large center-right watermark */}
      <div
        style={{
          position: 'absolute',
          right: -80,
          top: 580 + doodleY,
          width: 700,
          height: 700,
          opacity: 0.15,
          zIndex: 0,
          ...enterFromRight(6, 100),
        }}
      >
        <Img src={staticFile('doodles/1-d-02.png')} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
      </div>

      {/* Content — 3-zone space-between */}
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
        {/* ZONE TOP: Counter + Hook heading */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
          <div style={{ ...enter(0, 20), fontWeight: 400, fontSize: 28, color: colors.textSecondary, letterSpacing: 2 }}>
            02 / 10
          </div>
          <div
            style={{
              ...enter(6, 40),
              fontFamily: fonts.marker,
              fontSize: 108,
              color: colors.red,
              lineHeight: 1.0,
            }}
          >
            SOUND FAMILIAR?
          </div>
        </div>

        {/* ZONE MID: Big body text */}
        <div
          style={{
            ...enter(14, 40),
            fontFamily: fonts.primary,
            fontWeight: 700,
            fontSize: 82,
            lineHeight: 1.3,
            color: colors.white,
          }}
        >
          The scale says 75kg —{' '}
          <span style={{ color: colors.textSecondary }}>hasn't moved in years. But your gut keeps quietly growing.</span>
        </div>

        {/* ZONE BOT: Sub insight + arrow */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
          <div style={{ height: 3, background: colors.textSecondary, borderRadius: 2, opacity: 0.25, ...expandWidth(22) }} />
          <div
            style={{
              ...enter(22, 30),
              fontFamily: fonts.primary,
              fontWeight: 400,
              fontSize: 50,
              color: colors.textSecondary,
              lineHeight: 1.55,
            }}
          >
            Your weight isn't the problem. Your{' '}
            <strong style={{ color: colors.white }}>body composition</strong> is.
          </div>
          <div style={{ ...enter(30, 20), color: colors.green, fontSize: 44, marginTop: 8 }}>→</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
