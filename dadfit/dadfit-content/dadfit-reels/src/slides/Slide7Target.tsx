import { AbsoluteFill, Img, staticFile } from 'remotion';
import { useEntrance } from '../hooks/useEntrance';
import { colors, fonts } from '../tokens';

export const Slide7Target: React.FC = () => {
  const { enter, enterFromRight, float } = useEntrance();
  const doodleY = float(6, 100);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(145deg,#111a13 0%,#151f16 35%,#192118 58%,#1c1e1c 100%)',
        fontFamily: fonts.primary,
        overflow: 'hidden',
      }}
    >
      <AbsoluteFill
        style={{
          background: 'radial-gradient(ellipse at 60% 40%, rgba(52,195,99,0.05) 0%, transparent 55%)',
          pointerEvents: 'none',
        }}
      />

      {/* Doodle — large center-right watermark */}
      <div
        style={{
          position: 'absolute',
          right: -80,
          top: 700 + doodleY,
          width: 700,
          height: 700,
          opacity: 0.15,
          zIndex: 0,
          ...enterFromRight(5, 100),
        }}
      >
        <Img src={staticFile('doodles/1-d-07.png')} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
      </div>

      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '100px 72px 200px 72px',
          zIndex: 1,
        }}
      >
        {/* ZONE TOP: Counter + Headline + Formula */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
          <div style={{ ...enter(0, 20), fontSize: 28, color: colors.textSecondary, letterSpacing: 2 }}>07 / 10</div>
          <div
            style={{
              ...enter(5, 40),
              fontFamily: fonts.primary,
              fontWeight: 800,
              fontSize: 100,
              lineHeight: 1.05,
              color: colors.white,
              letterSpacing: '-2px',
            }}
          >
            YOUR DAILY{' '}
            <span style={{ color: colors.green }}>PROTEIN TARGET</span>
          </div>
          <div
            style={{
              ...enter(14, 35),
              fontFamily: fonts.primary,
              fontWeight: 400,
              fontSize: 50,
              color: colors.white,
              lineHeight: 1.6,
            }}
          >
            DadFit formula: <strong style={{ color: colors.green }}>1.75g of protein</strong> per kg of bodyweight.
          </div>
        </div>

        {/* ZONE BOT: Callout + arrow */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div
            style={{
              ...enter(20, 40),
              background: colors.secondaryBg,
              borderLeft: `6px solid ${colors.green}`,
              borderRadius: '0 16px 16px 0',
              padding: '44px 44px',
            }}
          >
            <div style={{ fontFamily: fonts.primary, fontWeight: 600, fontSize: 52, color: colors.white, lineHeight: 1.45 }}>
              <span style={{ color: colors.green, fontSize: 54, fontWeight: 800 }}>✦</span>{' '}
              At 75kg — hit{' '}
              <span style={{ color: colors.green, fontWeight: 800, fontSize: 60 }}>130g</span>{' '}
              of protein every single day.
            </div>
          </div>
          <div style={{ ...enter(30, 20), color: colors.green, fontSize: 44 }}>→</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
