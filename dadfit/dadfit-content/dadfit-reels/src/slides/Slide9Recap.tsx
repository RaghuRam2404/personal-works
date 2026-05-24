import { AbsoluteFill, Img, staticFile } from 'remotion';
import { useEntrance } from '../hooks/useEntrance';
import { colors, fonts } from '../tokens';

const recapItems = [
  '75kg can still mean skinny-fat — check body composition.',
  'Indian plates are 70% carbs — protein is dangerously low.',
  'No protein = no muscle = visceral fat grows.',
  'Target 130g of protein daily at 75kg.',
  'Resistance training turns excess carbs into muscle, not fat.',
];

export const Slide9Recap: React.FC = () => {
  const { enter, expandWidth, float, enterFromRight } = useEntrance();
  const doodleY = float(4, 88);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(145deg,#0f1710 0%,#141e14 35%,#182018 60%,#1c1e1c 100%)',
        fontFamily: fonts.primary,
        overflow: 'hidden',
      }}
    >
      <AbsoluteFill
        style={{
          background: 'radial-gradient(ellipse at 50% 20%, rgba(52,195,99,0.05) 0%, transparent 50%)',
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
          opacity: 0.12,
          zIndex: 0,
          ...enterFromRight(5, 100),
        }}
      >
        <Img src={staticFile('doodles/1-d-01.png')} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
      </div>

      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '100px 72px 160px 72px',
          zIndex: 1,
        }}
      >
        {/* ZONE TOP: Counter + Title + List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 36 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div style={{ ...enter(0, 20), fontSize: 28, color: colors.textSecondary, letterSpacing: 2 }}>09 / 10</div>
            <div style={{ ...enter(4, 35), fontFamily: fonts.marker, fontSize: 84, color: colors.green, lineHeight: 1.0 }}>
              QUICK RECAP
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 36 }}>
            {recapItems.map((text, i) => (
              <div
                key={i}
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: 28,
                  ...enter(8 + i * 8, 30),
                }}
              >
                <span style={{ color: colors.green, fontWeight: 800, fontSize: 50, flexShrink: 0, lineHeight: 1.1 }}>
                  {i + 1}
                </span>
                <span style={{ fontFamily: fonts.primary, fontWeight: 700, fontSize: 44, color: colors.white, lineHeight: 1.25 }}>
                  {text}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* ZONE BOT: Divider + Save this */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          <div style={{ height: 3, background: colors.textSecondary, borderRadius: 2, opacity: 0.25, ...expandWidth(40) }} />
          <div style={{ ...enter(42, 25), fontFamily: fonts.handwritten, fontSize: 58, color: colors.textSecondary }}>
            Save this.
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
