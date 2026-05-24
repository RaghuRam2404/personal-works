// DadFit brand tokens — mirrors the carousel CSS custom properties
export const colors = {
  primaryBg: '#1E1E1E',
  secondaryBg: '#292929',
  pageBg: '#080808',
  green: '#34C363',
  red: '#FF6B6B',
  white: '#FFFFFF',
  textSecondary: '#ADADAD',
  textTertiary: '#666666',
} as const;

export const fonts = {
  primary: "'Inter', sans-serif",
  handwritten: "'Caveat', cursive",
  marker: "'Permanent Marker', cursive",
} as const;

// Slide timing
export const FPS = 30;
export const SLIDE_DURATION = 120; // 4 seconds per slide
export const TRANSITION_DURATION = 20; // 20-frame cross-fade
export const NET_SLIDE_ADVANCE = SLIDE_DURATION - TRANSITION_DURATION; // 100 frames
export const TOTAL_SLIDES = 10;
export const TOTAL_FRAMES = NET_SLIDE_ADVANCE * TOTAL_SLIDES + TRANSITION_DURATION;
// = 100*10 + 20 = 1020 frames (~34s)
