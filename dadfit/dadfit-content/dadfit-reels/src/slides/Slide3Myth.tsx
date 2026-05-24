import { AbsoluteFill, Img, staticFile } from 'remotion';
import { useEntrance } from '../hooks/useEntrance';
import { colors, fonts } from '../tokens';

export const Slide3Myth: React.FC = () => {
  const { enter, enterFromRight, float } = useEntrance();
  const doodleY = float(4, 100);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(160deg,#1c1212 0%,#1f1616 30%,#1e1c1c 60%,#1E1E1E 100%)',
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
          width: 680,
          height: 680,
          opacity: 0.14,
          zIndex: 0,
          ...enterFromRight(4, 90),
        }}
      >
        <Img src={staticFile('doodles/1-d-03.png')} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
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
        {/* ZONE TOP: Counter + Myth card */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ ...enter(0, 20), fontSize: 28, color: colors.textSecondary, letterSpacing: 2 }}>03 / 10</div>
          <div style={{ ...enter(5, 35), fontFamily: fonts.marker, fontSize: 80, color: colors.red, lineHeight: 1.0 }}>MYTH —</div>
          <div
            style={{
              ...enter(10, 40),
              background: colors.secondaryBg,
              borderLeft: `6px solid ${colors.red}`,
              borderRadius: '0 16px 16px 0',
              padding: '48px 44px',
              fontFamily: fonts.primary,
              fontWeight: 800,
              fontSize: 62,
              lineHeight: 1.15,
              color: colors.white,
              letterSpacing: '-1px',
            }}
          >
            "75kg means my weight is normal — I'm probably fine."
          </div>
        </div>

        {/* ZONE BOT: Truth card + arrow */}
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
            <div style={{ fontFamily: fonts.marker, fontSize: 44, color: colors.green, marginBottom: 20 }}>TRUTH —</div>
            <div style={{ fontFamily: fonts.primary, fontWeight: 700, fontSize: 52, lineHeight: 1.25, color: colors.white }}>
              Low muscle + high visceral fat = <span style={{ color: colors.red }}>skinny-fat</span>. The scale hides the real danger.
            </div>
          </div>
          <div style={{ ...enter(32, 20), color: colors.green, fontSize: 44 }}>→</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
