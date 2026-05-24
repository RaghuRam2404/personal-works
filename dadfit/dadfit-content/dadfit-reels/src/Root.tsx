import { Composition } from 'remotion';
import { DadFitReel } from './DadFitReel';
import { TOTAL_FRAMES, FPS } from './tokens';

export const RemotionRoot = () => {
  return (
    <Composition
      id="75KG-FATTER"
      component={DadFitReel}
      durationInFrames={TOTAL_FRAMES}
      fps={FPS}
      width={1080}
      height={1920}
    />
  );
};
