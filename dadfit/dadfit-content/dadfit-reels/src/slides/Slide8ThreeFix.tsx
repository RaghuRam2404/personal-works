import { AbsoluteFill, Img, staticFile } from 'remotion';
import { useEntrance } from '../hooks/useEntrance';
import { colors, fonts } from '../tokens';

const steps = [
  'Hit 130g protein first — before carbs, calories, or the scale.',
  'Add resistance training 3× a week — forces carbs into muscle.',
  'Track protein today — one honest day shows you the gap.',
];

export const Slide8ThreeFix: React.FC = () => {
  const { enter, enterFromRight, float, expandWidth } = useEntrance();
  const doodleY = float(5, 98);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(155deg,#131516 0%,#171a1c 35%,#1a1d1f 60%,#1E1E1E 100%)',
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
        <Img src={staticFile('doodles/1-d-08.png')} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
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
        {/* ZONE TOP: Counter + Headline + Steps */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 36 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div style={{ ...enter(0, 20), fontSize: 28, color: colors.textSecondary, letterSpacing: 2 }}>08 / 10</div>
            <div
              style={{
                ...enter(5, 40),
                fontFamily: fonts.primary,
                fontWeight: 800,
                fontSize: 108,
                lineHeight: 1.0,
                color: colors.white,
                letterSpacing: '-2px',
              }}
            >
              YOUR <span style={{ color: colors.green }}>3-STEP FIX</span>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 40 }}>
            {steps.map((text, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 32,
                  ...enter(10 + i * 10, 35),
                }}
              >
                <div
                  style={{
                    width: 80,
                    height: 80,
                    borderRadius: '50%',
                    background: colors.green,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    fontFamily: fonts.primary,
                    fontWeight: 800,
                    fontSize: 38,
                    color: colors.primaryBg,
                    marginTop: 4,
                  }}
                >
                  {i + 1}
                </div>
                <div style={{ fontFamily: fonts.primary, fontWeight: 600, fontSize: 44, color: colors.white, lineHeight: 1.35 }}>
                  {text}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ZONE BOT: Quote + arrow */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div style={{ height: 4, background: colors.green, borderRadius: 2, ...expandWidth(28) }} />
          <div style={{ ...enter(28, 30), fontFamily: fonts.handwritten, fontSize: 54, color: colors.green }}>
            Simple. No fancy diet. Start today.
          </div>
          <div style={{ ...enter(35, 20), color: colors.green, fontSize: 44 }}>→</div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
