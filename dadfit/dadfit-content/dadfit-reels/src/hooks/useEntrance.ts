import { useCurrentFrame, useVideoConfig, interpolate, spring, Easing } from 'remotion';

const EASE_OUT_EXPO = Easing.bezier(0.16, 1, 0.3, 1);

/**
 * Returns animation helpers scoped to the current Sequence's local frame.
 */
export const useEntrance = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  /** Smooth spring-driven value from 0 → 1 starting at `delay` frames. */
  const spr = (delay = 0, config?: { damping?: number; stiffness?: number }) =>
    spring({
      frame: frame - delay,
      fps,
      config: { damping: 22, stiffness: 120, mass: 1, ...config },
    });

  /** Fade + slide-up entrance. Returns a React style object. */
  const enter = (delay = 0, distance = 40): React.CSSProperties => {
    const progress = interpolate(frame - delay, [0, 20], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: EASE_OUT_EXPO,
    });
    return {
      opacity: interpolate(frame - delay, [0, 12], [0, 1], {
        extrapolateLeft: 'clamp',
        extrapolateRight: 'clamp',
      }),
      transform: `translateY(${(1 - progress) * distance}px)`,
    };
  };

  /** Slide-in from the right. */
  const enterFromRight = (delay = 0, distance = 80): React.CSSProperties => {
    const x = interpolate(frame - delay, [0, 24], [distance, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: EASE_OUT_EXPO,
    });
    const opacity = interpolate(frame - delay, [0, 14], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
    });
    return { opacity, transform: `translateX(${x}px)` };
  };

  /** Scale + fade pop-in (for logos, numbers). */
  const pop = (delay = 0): React.CSSProperties => {
    const s = spr(delay, { damping: 18, stiffness: 180 });
    return {
      opacity: Math.min(1, s * 2),
      transform: `scale(${0.85 + s * 0.15})`,
    };
  };

  /** Expand width from 0 to 100% (for green bars). */
  const expandWidth = (delay = 0): React.CSSProperties => {
    const w = interpolate(frame - delay, [0, 28], [0, 100], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: EASE_OUT_EXPO,
    });
    return { width: `${w}%` };
  };

  /** Raw interpolated 0→1 ease-out progress (for number counting etc.) */
  const progress = (delay = 0, duration = 40) =>
    interpolate(frame - delay, [0, duration], [0, 1], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: EASE_OUT_EXPO,
    });

  /** Gentle float offset for doodle idle animation (sinusoidal). */
  const float = (amplitude = 8, period = 90) =>
    Math.sin((frame / period) * Math.PI * 2) * amplitude;

  return { frame, fps, spr, enter, enterFromRight, pop, expandWidth, progress, float };
};
