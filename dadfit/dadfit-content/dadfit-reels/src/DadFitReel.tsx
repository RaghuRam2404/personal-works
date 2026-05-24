import { AbsoluteFill, Sequence, useCurrentFrame, interpolate } from 'remotion';
import { loadFont } from '@remotion/google-fonts/Inter';
import { loadFont as loadCaveat } from '@remotion/google-fonts/Caveat';
import { loadFont as loadPermanentMarker } from '@remotion/google-fonts/PermanentMarker';
import { NET_SLIDE_ADVANCE, SLIDE_DURATION, TRANSITION_DURATION } from './tokens';

import { Slide1Cover } from './slides/Slide1Cover';
import { Slide2Pain } from './slides/Slide2Pain';
import { Slide3Myth } from './slides/Slide3Myth';
import { Slide4Stat } from './slides/Slide4Stat';
import { Slide5Protein } from './slides/Slide5Protein';
import { Slide6DeskJob } from './slides/Slide6DeskJob';
import { Slide7Target } from './slides/Slide7Target';
import { Slide8ThreeFix } from './slides/Slide8ThreeFix';
import { Slide9Recap } from './slides/Slide9Recap';
import { Slide10CTA } from './slides/Slide10CTA';

loadFont();
loadCaveat();
loadPermanentMarker();

const slides = [
  Slide1Cover,
  Slide2Pain,
  Slide3Myth,
  Slide4Stat,
  Slide5Protein,
  Slide6DeskJob,
  Slide7Target,
  Slide8ThreeFix,
  Slide9Recap,
  Slide10CTA,
];

/**
 * A single slide wrapped in cross-fade enter/exit opacity.
 */
const FadingSlide: React.FC<{
  slideIndex: number;
  Component: React.FC;
}> = ({ slideIndex, Component }) => {
  const from = slideIndex * NET_SLIDE_ADVANCE;
  return (
    <Sequence from={from} durationInFrames={SLIDE_DURATION} layout="none">
      <SlideWithFade>
        <Component />
      </SlideWithFade>
    </Sequence>
  );
};

const SlideWithFade: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(
    frame,
    [0, TRANSITION_DURATION, SLIDE_DURATION - TRANSITION_DURATION, SLIDE_DURATION],
    [0, 1, 1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  return (
    <AbsoluteFill style={{ opacity }}>
      {children}
    </AbsoluteFill>
  );
};

export const DadFitReel: React.FC = () => {
  return (
    <AbsoluteFill style={{ backgroundColor: '#080808' }}>
      {slides.map((Slide, i) => (
        <FadingSlide key={i} slideIndex={i} Component={Slide} />
      ))}
    </AbsoluteFill>
  );
};
