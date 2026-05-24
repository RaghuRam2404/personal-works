import { AbsoluteFill, Img, staticFile } from 'remotion';
import { useEntrance } from '../hooks/useEntrance';
import { colors, fonts } from '../tokens';

export const Slide1Cover: React.FC = () => {
  const { enter, enterFromRight, float, progress } = useEntrance();

  const glowOpacity = progress(0, 50) * 0.12;
  const doodleY = float(6, 100);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(145deg,#0c1610 0%,#121e15 30%,#17211a 58%,#1c1e1c 100%)',
        fontFamily: fonts.primary,
        overflow: 'hidden',
      }}
    >
      {/* Radial glow */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at 75% 88%, rgba(52,195,99,${glowOpacity}) 0%, transparent 60%)`,
          pointerEvents: 'none',
        }}
      />

      {/* Doodle — slides in from right, floats */}
      <div
        style={{
          position: 'absolute',
          right: 120,
          bottom: 60 + doodleY,
          width: 520,
          height: 520,
          ...enterFromRight(8, 120),
        }}
      >
        <Img
          src={staticFile('doodles/1-d-01.png')}
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
        />
      </div>

      {/* Hero text — centred, scales in */}
      <AbsoluteFill
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '80px 100px',
          textAlign: 'center',
        }}
      >
        <div
          style={{
            ...enter(0, 50),
            fontFamily: fonts.primary,
            fontWeight: 800,
            fontSize: 140,
            lineHeight: 1.0,
            color: colors.white,
            textTransform: 'uppercase',
            letterSpacing: '-3px',
          }}
        >
          75KG AND STILL{' '}
          <span style={{ color: colors.green, display: 'block', ...enter(10, 40) }}>
            GETTING FATTER
          </span>
        </div>
      </AbsoluteFill>

      {/* DadFit brand tag bottom-left */}
      <div
        style={{
          position: 'absolute',
          bottom: 60,
          left: 72,
          ...enter(20),
          fontFamily: fonts.primary,
          fontWeight: 700,
          fontSize: 36,
          color: 'rgba(255,255,255,0.5)',
          letterSpacing: 1,
        }}
      >
        @dadfit.in
      </div>
    </AbsoluteFill>
  );
};
