import { AbsoluteFill, Img, staticFile } from 'remotion';
import { useEntrance } from '../hooks/useEntrance';
import { colors, fonts } from '../tokens';

export const Slide10CTA: React.FC = () => {
  const { enter, pop, enterFromRight, float, expandWidth } = useEntrance();
  const doodleY = float(5, 102);

  return (
    <AbsoluteFill
      style={{
        background: 'linear-gradient(145deg,#060c08 0%,#090f09 25%,#0c120d 50%,#0f1410 75%,#111512 100%)',
        fontFamily: fonts.primary,
        overflow: 'hidden',
      }}
    >
      <AbsoluteFill
        style={{
          background: 'radial-gradient(ellipse at 50% 100%, rgba(52,195,99,0.10) 0%, rgba(52,195,99,0.02) 45%, transparent 65%)',
          pointerEvents: 'none',
        }}
      />

      {/* Doodle — large center-right watermark */}
      <div
        style={{
          position: 'absolute',
          right: -60,
          top: 580 + doodleY,
          width: 720,
          height: 720,
          opacity: 0.22,
          ...enterFromRight(8, 100),
        }}
      >
        <Img src={staticFile('doodles/1-d-09.png')} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
      </div>

      <div
        style={{
          position: 'absolute',
          inset: 0,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '120px 72px 120px 72px',
          textAlign: 'center',
        }}
      >
        {/* Logo */}
        <div style={{ ...pop(0) }}>
          <Img
            src={staticFile('logo.png')}
            style={{ height: 100, objectFit: 'contain', mixBlendMode: 'screen' }}
          />
        </div>

        {/* Middle: bars + CTA */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 40, alignSelf: 'stretch', alignItems: 'center' }}>
          <div style={{ height: 4, background: colors.green, borderRadius: 2, alignSelf: 'stretch', ...expandWidth(10) }} />
          <div
            style={{
              ...enter(14, 40),
              fontFamily: fonts.primary,
              fontWeight: 700,
              fontSize: 70,
              color: colors.white,
              lineHeight: 1.3,
            }}
          >
            Save this — your body composition fix starts here.
          </div>
          <div style={{ height: 4, background: colors.green, borderRadius: 2, alignSelf: 'stretch', ...expandWidth(20) }} />
        </div>

        {/* Bottom: sub text + handle */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 28, alignItems: 'center' }}>
          <div style={{ ...enter(24, 30), fontFamily: fonts.primary, fontWeight: 400, fontSize: 44, color: colors.textSecondary, lineHeight: 1.5 }}>
            Join us to fix the real problem
          </div>
          <div style={{ ...enter(30, 35), fontFamily: fonts.marker, fontSize: 84, color: colors.green }}>
            @dadfit.in
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
