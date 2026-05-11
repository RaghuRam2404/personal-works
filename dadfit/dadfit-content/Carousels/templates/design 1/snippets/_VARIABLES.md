# Snippet Variable Reference

Every snippet file uses `{{VAR}}` placeholders. The renderer substitutes these at assembly time.

## Universal (all slides)
| Variable | Description |
|----------|-------------|
| `{{SLIDE_TYPE}}` | e.g. `A1`, `B4`, `C1` — used for `.slide-type-A1` CSS class |
| `{{COUNTER}}` | e.g. `01 / 09` — zero-padded current / total |
| `{{DOODLE_SRC}}` | e.g. `../doodles/7-d-03.png` |
| `{{DOODLE_ALT}}` | Short alt text for doodle image |

## Type A — Cover
| Variable | Slides | Description |
|----------|--------|-------------|
| `{{LOGO_SRC}}` | A1–A5 | `../../../../Resources/Images/logo.png` |
| `{{HEADLINE}}` | A1,A2,A4,A5 | Main headline HTML (spans allowed) |
| `{{SUBTEXT}}` | A1,A2 | 1-line sub-text below headline |
| `{{CAVEAT_OPENER}}` | A4 | e.g. `Be honest —` |
| `{{CHALLENGE_WORD}}` | A5 | e.g. `YOU READY?` in Permanent Marker |
| `{{PHOTO_SRC}}` | A2,A3 | Path to photo |
| `{{PHOTO_ALT}}` | A2,A3 | Photo alt text |

## Type B — Content
| Variable | Slides | Description |
|----------|--------|-------------|
| `{{HEADLINE}}` | B1–B6 | Main headline HTML |
| `{{BODY}}` | B1,B2,B4 | Body copy (2–3 lines max) |
| `{{CAVEAT}}` | B1,B5,B6 | Caveat footer line |
| `{{CALLOUT}}` | B4 | Callout box sentence (replaces body) |
| `{{WRONG_ITEMS}}` | B5 | 3 lines in WRONG column |
| `{{RIGHT_ITEMS}}` | B5 | 3 lines in RIGHT column |
| `{{STEP1}}` | B6 | Step 1 text |
| `{{STEP2}}` | B6 | Step 2 text |
| `{{STEP3}}` | B6 | Step 3 text |
| `{{PHOTO_SRC}}` | B2,B3 | Path to photo |
| `{{PHOTO_ALT}}` | B2,B3 | Photo alt text |

## Type C — Problem / Pain
| Variable | Slides | Description |
|----------|--------|-------------|
| `{{PAIN_LABEL}}` | C1 | e.g. `SOUND FAMILIAR?` |
| `{{PAIN_LINE1}}` | C1 | First pain statement line |
| `{{PAIN_LINE2}}` | C1 | Second pain statement line |
| `{{REFRAME}}` | C1 | 1-line reframe (grey) |
| `{{MYTH_TEXT}}` | C3 | The myth (in red card) |
| `{{TRUTH_TEXT}}` | C3 | The truth (in green card) |
| `{{EXCUSE1}}` | C4 | First excuse |
| `{{EXCUSE2}}` | C4 | Second excuse |
| `{{EXCUSE3}}` | C4 | Third excuse |
| `{{C4_CAVEAT}}` | C4 | Caveat footer e.g. `All of these are solvable.` |

## Type D — Proof / Stat
| Variable | Slides | Description |
|----------|--------|-------------|
| `{{STAT_LABEL}}` | D1,D3 | e.g. `THE TRUTH:` |
| `{{BIG_NUMBER}}` | D1 | e.g. `73%` |
| `{{STAT_CONTEXT}}` | D1 | 1-line context sentence |
| `{{STAT_CAVEAT}}` | D1 | Caveat quote |
| `{{LEFT_NUMBER}}` | D3 | Left (red) stat number |
| `{{LEFT_LABEL}}` | D3 | Left stat description |
| `{{RIGHT_NUMBER}}` | D3 | Right (green) stat number |
| `{{RIGHT_LABEL}}` | D3 | Right stat description |
| `{{D3_FOOTER}}` | D3 | Footer line |
| `{{HEADLINE}}` | D4 | e.g. `What happens in 90 days.` |
| `{{D4_LABEL}}` | D4 | Permanent Marker label e.g. `REAL TIMELINE` |
| `{{ROW1_LABEL}}` | D4 | e.g. `Week 1` |
| `{{ROW1_RESULT}}` | D4 | e.g. `Energy up` |
| `{{ROW1_PCT}}` | D4 | Bar fill percent e.g. `20%` |
| `{{ROW2_LABEL}}` | D4 | e.g. `Week 4` |
| `{{ROW2_RESULT}}` | D4 | e.g. `Habit locked` |
| `{{ROW2_PCT}}` | D4 | e.g. `50%` |
| `{{ROW3_LABEL}}` | D4 | e.g. `Week 12` |
| `{{ROW3_RESULT}}` | D4 | e.g. `−8 to −12 kg` |
| `{{ROW3_PCT}}` | D4 | e.g. `100%` |

## Type E — Transition / Bridge
| Variable | Slides | Description |
|----------|--------|-------------|
| `{{E1_OPENER}}` | E1 | Caveat opener e.g. `But here's the thing —` |
| `{{E1_STATEMENT}}` | E1 | Bold pivot statement |
| `{{E1_SUBTEXT}}` | E1 | Optional grey sub-line |
| `{{E2_PART_LABEL}}` | E2 | e.g. `PART 2` |
| `{{E2_SECTION_TITLE}}` | E2 | e.g. `The Nutrition Fix` |
| `{{E2_DESCRIPTION}}` | E2 | Grey description line |
| `{{E3_QUOTE}}` | E3 | Full quote text |
| `{{E3_FOOTER}}` | E3 | Footer line e.g. `Now let's build yours →` |
| `{{E4_PILL1}}` | E4 | Part 1 pill label |
| `{{E4_PILL2}}` | E4 | Part 2 pill label (active) |
| `{{E4_PILL3}}` | E4 | Part 3 pill label |
| `{{E4_HEADLINE}}` | E4 | Big headline |
| `{{E4_DESCRIPTION}}` | E4 | Grey description |

## Type F — Pattern Interrupt
| Variable | Slides | Description |
|----------|--------|-------------|
| `{{F2_LINE1}}` | F2 | First giant line (white) |
| `{{F2_LINE2}}` | F2 | Second giant line (green) |
| `{{F3_LINE1}}` | F3 | First Caveat word (white) |
| `{{F3_LINE2}}` | F3 | Second Caveat word (green) |
| `{{F4_NUMBER}}` | F4 | Big decorative digit (e.g. `5`) |
| `{{F4_TIP_LABEL}}` | F4 | e.g. `TIP 5 OF 7` |
| `{{PHOTO_SRC}}` | F1 | Photo path |
| `{{PHOTO_ALT}}` | F1 | Photo alt text |

## Type G — Recap
| Variable | Slides | Description |
|----------|--------|-------------|
| `{{BULLET1}}` | G1 | Bullet 1 text |
| `{{BULLET2}}` | G1 | Bullet 2 text |
| `{{BULLET3}}` | G1 | Bullet 3 text |
| `{{BULLET4}}` | G1 | Bullet 4 text (optional — leave blank to omit) |
| `{{BULLET5}}` | G1 | Bullet 5 text (optional — leave blank to omit) |
| `{{G1_CAVEAT}}` | G1 | Footer e.g. `Save this.` |
| `{{G3_STATEMENT}}` | G3 | One giant truth statement |
| `{{G3_CAVEAT}}` | G3 | Footer e.g. `Save it. Share it.` |
| `{{HABIT1}}` | G4 | Habit 1 name |
| `{{HABIT1_FILLED}}` | G4 | Number of filled dots (0–7) |
| `{{HABIT2}}` | G4 | Habit 2 name |
| `{{HABIT2_FILLED}}` | G4 | Number of filled dots (0–7) |
| `{{HABIT3}}` | G4 | Habit 3 name |
| `{{HABIT3_FILLED}}` | G4 | Number of filled dots (0–7) |
| `{{G4_CAVEAT}}` | G4 | Footer line |
| `{{CARD1_EMOJI}}` | G2 | Emoji for card 1 |
| `{{CARD1_LABEL}}` | G2 | 2-word label card 1 |
| `{{CARD2_EMOJI}}` | G2 | |
| `{{CARD2_LABEL}}` | G2 | |
| `{{CARD3_EMOJI}}` | G2 | |
| `{{CARD3_LABEL}}` | G2 | |
| `{{CARD4_EMOJI}}` | G2 | |
| `{{CARD4_LABEL}}` | G2 | |
| `{{CARD5_EMOJI}}` | G2 | |
| `{{CARD5_LABEL}}` | G2 | |
| `{{CARD6_EMOJI}}` | G2 | |
| `{{CARD6_LABEL}}` | G2 | |

## Type H — CTA
| Variable | Slides | Description |
|----------|--------|-------------|
| `{{LOGO_SRC}}` | H1,H2 | `../../../../Resources/Images/logo.png` |
| `{{CTA_TEXT}}` | H1 | Main CTA sentence(s) |
| `{{H1_FOLLOW}}` | H1 | Follow line |
| `{{H3_HEADLINE}}` | H3 | Big headline |
| `{{H3_DM_WORD}}` | H3 | The DM keyword e.g. `DAD` |
| `{{TESTIMONIAL_QUOTE}}` | H4 | Full testimonial quote |
| `{{TESTIMONIAL_ATTRIBUTION}}` | H4 | e.g. `Suresh, 38 · DadFit member` |
| `{{H4_CTA_LINE}}` | H4 | e.g. `Follow for more wins` |
| `{{FOUNDER_PHOTO_SRC}}` | H2 | Path to founder/Raghu photo ONLY |
| `{{FOUNDER_NAME}}` | H2 | e.g. `Raghu Ram` |
| `{{FOUNDER_CREDENTIALS}}` | H2 | e.g. `NASM Certified Trainer` |
| `{{H2_MESSAGE}}` | H2 | Personal message line |
| `{{H2_CTA}}` | H2 | CTA line with DM keyword |
