import { AbsoluteFill, Img, staticFile } from 'remotion';
import { useEntrance } from '../hooks/useEntrance';
import { colors, fonts } from '../tokens';

export const Slide5Protein: React.FC = () => {
  const { enter, enterFromRight, float, expandWidth } = useEntrance();
  const doodleY = float(5, 95);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(150deg,#131520 0%,#171b26 35%,#1a1e24 60%,#1c1e22 100%)',
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
          opacity: 0.15,
          zIndex: 0,
          ...enterFromRight(5, 100),
        }}
      >
        <Img src={staticFile('doodles/1-d-05.png')} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
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
        {/* ZONE TOP: Counter + Headline */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
          <div style={{ ...enter(0, 20), fontSize: 28, color: colors.textSecondary, letterSpacing: 2 }}>05 / 10</div>
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
            PROTEIN DEFICIENCY{' '}
            <span style={{ color: colors.green }}>GROWS FAT</span>
          </div>
        </div>

        {/* ZONE MID: Body */}
        <div
          style={{
            ...enter(14, 35),
            fontFamily: fonts.primary,
            fontWeight: 400,
            fontSize: 52,
            color: colors.white,
            lineHeight: 1.65,
          }}
        >
          Without protein, your body can't build or protect muscle.{' '}
          <span style={{ color: colors.textSecondary }}>Fat fills the gap — and belly fat grows first.</span>
        </div>

        {/* ZONE BOT: Divider + quote + arrow */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div style={{ height: 4, background: colors.green, borderRadius: 2, ...expandWidth(22) }} />
          <div style={{ ...enter(22, 30), fontFamily: fonts.handwritten, fontSize: 54, color: colors.green }}>
            Muscle loss accelerates after 35.
          </div>
          <div style={{ ...enter(30, 20), color: colors.green, fontSize: 44 }}>→</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
