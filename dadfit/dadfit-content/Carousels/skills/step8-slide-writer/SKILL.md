# SKILL: Step 8 — Slide Writer (Subagent)

## Purpose

You are filling the content for **one** DadFit Instagram carousel. Your only output is a single JSON file written to `/tmp/carousel_{uuid}.json`. A renderer script generates all HTML from pre-built snippet templates using the vars you supply. **Do not write any HTML, CSS, or script blocks — ever.**

> ⛔ **OUTPUT RULE — READ BEFORE ANYTHING ELSE:**
> Your **only** output mechanism is a single `create_file` call. **DO NOT use `run_in_terminal` to write the JSON.** No `python3 -c "..."`, no heredoc, no multiline shell command. Any command with a newline (`\n`) inside it is forbidden. If you catch yourself typing `python3 -c`, stop and use `create_file` instead.

---

## Your inputs

The agent that spawned you will provide:

- **UUID** — unique carousel identifier
- **Running No** — integer 1–100
- **Batch** — always `1`
- **Title** — carousel topic headline
- **Keyword** — primary keyword
- **Category** — `TOFU` (educate cold audience) | `MOFU` (deepen trust) | `BOFU` (convert)
- **Hook** — first slide headline (pre-written)
- **CTA** — last slide call-to-action text (pre-written)
- **Script content** — one sentence per line, each line = one slide's core idea

---

## STEP 1 — Map script to slides

The `script_content` has one idea per line. Map these into a visual carousel story. **The goal is to tell the story well — not to compress everything onto fewer slides.**

### Readability rules (non-negotiable)

Every slide is swiped in under 2 seconds. Cramming kills engagement.

- **One idea per slide.** Two separable thoughts → two slides.
- **Headline: 3–6 words max.** Strong, punchy, ALL CAPS.
- **Body copy: 2–3 lines max, never 4.** Cut long sentences to their essence.
- **Never stack more than 3 content blocks.** A callout box replaces body — it doesn't add to it.
- **Breathing room is content.** Do not fill whitespace.
- **If a slide feels crowded, split it.** Add a B1 or E1 rather than cramming.

### Slide count rules

- Slide 1 = always A-type Cover. Last slide = always H-type CTA.
- **Total: 8–10 slides** for a 10-line script. Never fewer than 7, never more than 11.
- Rough pattern: 1 cover + 1–2 pain/bridge + 5–6 content + 1 recap (optional) + 1 CTA.
- When in doubt, **add a slide** — shorter and cleaner wins.

**Write your slide map (type + one-line description per slide) in your thinking before producing any JSON.**

### Slide type selection

Choose based on what the script sentence is **trying to do**.

---

#### TYPE A — Cover / Hook (Slide 1 only)

| Type | Layout | Use when the hook is… |
|------|--------|----------------------|
| **A1** | Big headline + 1-line sub-text + doodle | A bold declarative statement or urgent promise. Default cover — most carousels. |
| **A4** | Caveat opener ("Be honest —") + big headline + doodle | Personal, self-reflective hook that challenges the dad directly. |
| **A5** | Permanent Marker challenge word + big headline + doodle | A dare or challenge hook ("30-DAY CHALLENGE"). |

> A2, A3 require a real photo — **never use them.**

---

#### TYPE B — Content / Explainer

| Type | Layout | Use when… |
|------|--------|-----------|
| **B1** | Headline + 2–3 lines body + optional Caveat + doodle | A single tip or insight needing brief explanation. Workhorse — ~60% of content slides. |
| **B4** | Headline + callout box (one punchy action) + doodle | A tip with one sharp action worth boxing. Callout **replaces** body — never both. |
| **B5** | Headline + WRONG vs RIGHT two-column grid + Caveat + doodle | Contrast between wrong and right approach. Use only when the script explicitly compares two behaviours. |
| **B6** | Headline + numbered step list (3 steps) + Caveat + doodle | A protocol with a clear 1→2→3 structure. |

> B2, B3 require a real photo — **never use them.**

---

#### TYPE C — Problem / Pain

| Type | Layout | Use when… |
|------|--------|-----------|
| **C1** | Pain label + 2-line pain statement + 1-line reframe + doodle | Naming the dad's lived frustration. Use once near the start for empathy. |
| **C3** | MYTH card + TRUTH card + doodle | Busting a specific named misconception. Only when script explicitly contradicts a common belief. |
| **C4** | "BE REAL." header + 3-item excuse checklist + Caveat + doodle | Listing the excuses the dad makes. Only when script calls out avoidance by name. |

> C2 requires a real photo — **never use it.**

---

#### TYPE D — Proof / Stat

| Type | Layout | Use when… |
|------|--------|-----------|
| **D1** | Label + giant number + 1-line context + Caveat + doodle | A single real statistic. Number must exist in the script — do not invent data. |
| **D3** | Two-column split: left stat vs right stat + footer + doodle | Two contrasting numbers. Only when the script has two explicit opposing numbers. |
| **D4** | Headline + 3 progress bar rows + doodle | A timeline with measurable milestones (Week 1 / Week 4 / Week 12). |

> D2 requires a real photo — **never use it.**

---

#### TYPE E — Transition / Bridge

| Type | Layout | Use when… |
|------|--------|-----------|
| **E1** | Caveat opener + 1 bold Inter statement + doodle | A pivot from problem to solution. Minimal — 2 elements only. |
| **E2** | Section label card (PART N — Title) + 1-line description + doodle | Dividing a long carousel into named parts. |
| **E3** | Caveat quote between two green bars + footer + doodle | A quotable one-liner that deserves its own moment. |

> E4 (pill navigation) — use only for carousels with 3+ distinct named sections.

---

#### TYPE F — Pattern Interrupt (max 1 per carousel)

| Type | Layout | Use when… |
|------|--------|-----------|
| **F2** | Two massive ALL CAPS lines, no counter/brand | Carousel is 9+ slides and needs a visual pause. ("SLOW IS / SUSTAINABLE.") |
| **F3** | One giant Caveat word pair, no counter/brand | Same — breather for motivational carousel. |

> F1 requires a photo. F4 is decorative. **Avoid both.** Use F only to break monotony on long carousels.

---

#### TYPE G — Recap / Summary (second-to-last slide)

| Type | Layout | Use when… |
|------|--------|-----------|
| **G1** | "QUICK RECAP" + 3–5 numbered bullets + Caveat "Save this." | 3+ distinct tips worth summarising. Default recap type. |
| **G3** | "REMEMBER THIS —" + one giant statement + green bar + Caveat | The entire lesson fits one memorable line. |

> G2 (emoji grid) — only if 6 takeaways map cleanly to icons.  
> G4 (habit tracker) — only for habit/routine carousels.  
> **Skip G entirely** if carousel ≤7 slides or ideas are already distinct.

---

#### TYPE H — CTA (always last slide)

| Type | Layout | Use when… |
|------|--------|-----------|
| **H1** | Logo + green divider + CTA text + `@dadfit.in` | Default CTA. Use for all carousels. |
| **H3** | Green glow + "DM me WORD" + handle | CTA explicitly asks users to DM a keyword. |

> H2 (founder photo) and H4 (testimonial) — **skip both** unless you have real specific content.

---

**Decision shortcut:** Is it a fact/tip → B1. Tip with one sharp action → B4. Pain/frustration → C1. Real number → D1. Contrast (wrong vs right) → B5 or C3. Pivot moment → E1. Steps sequence → B6. Philosophical one-liner → E3 or G3.

---

### Slide Type Detail Reference

All 36 types with exact template line numbers, full layout spec, and Use / Skip guidance.

#### TYPE A — Cover (Slide 1 only)

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **A1** | ~318 | `s-bg` + doodle + counter top-left + logo top-right + big headline (Inter ExtraBold, white) + 1-line sub-text (ADADAD) + `→` footer | **USE** — default cover. Bold declarative hook or urgent promise. |
| **A2** | ~361 | `s-bg` + counter top-left + logo top-right + left headline + right framed photo (`photo-ph`) + `→` footer | **SKIP** — requires real photo. Replace with A1. |
| **A3** | ~476 | Full-bleed photo + dark overlay + green gradient bottom + counter + logo + bottom headline + `→` footer | **SKIP** — requires real photo. Replace with A1. |
| **A4** | ~402 | `s-bg` + doodle + counter top-left + logo top-right + Caveat opener line (white, 42px) + big Inter headline (white) + `→` footer | **USE** — personal/reflective hook. Sounds like a coach talking directly to the dad. |
| **A5** | ~438 | `s-bg` + doodle + counter top-left + logo top-right + Permanent Marker challenge word (#34C363) + big Inter headline + `→` footer | **USE** — dare / protocol / challenge hook. Use when carousel is a 30-day challenge. |

#### TYPE B — Content / Explainer

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **B1** | ~527 | `s-bg` + doodle + counter + headline (Inter ExtraBold, 68px, white) + 2-line body (Inter Regular, 33px, white) + optional green bar + Caveat footer + `→` | **USE** — workhorse. Single tip, principle, or insight. ~60% of content slides. |
| **B2** | ~568 | `s-bg` + counter + left headline + body + right framed `photo-ph` (340×490px) + `→` | **SKIP** — requires real photo. Replace with B1. |
| **B3** | ~707 | Full-bleed photo + overlays + counter + bottom headline + body + `→` | **SKIP** — requires real photo. Replace with B1, B4, or E1. |
| **B4** | ~737 | `s-bg` + doodle + counter + headline (68px) + body (31px, 2 lines) + `callout` box (✦ + one action sentence) + `→` | **USE** — tip with a concrete boxed action. Callout **replaces** body — never use both. |
| **B5** | ~605 | `s-bg` + doodle + counter + headline (60px) + two-column grid: WRONG card (red border) + RIGHT card (green border) + Caveat footer + `→` | **USE** — explicit wrong vs right contrast. Only when script directly compares two behaviours. |
| **B6** | ~653 | `s-bg` + doodle + counter + headline (62px) + 3 step rows (64px green circle with digit + step text) + green bar + Caveat footer + `→` | **USE** — 3-step ordered protocol. Only when script is literally "step 1, step 2, step 3". |

#### TYPE C — Problem / Pain

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **C1** | ~790 | `s-bg` + doodle + counter + Permanent Marker pain label (#FF6B6B, 52px) + Inter Bold pain statement (white, 46px, 2 lines) + Inter Regular reframe (ADADAD, 30px) + `→` | **USE** — name the dad's exact frustration. Use once near the start to establish empathy before the solution. |
| **C2** | ~904 | `s-bg` + counter + Permanent Marker label (#FF6B6B) + left headline (red accent) + right `photo-ph` (red-tint border) + body + `→` | **SKIP** — requires real photo. Replace with C1. |
| **C3** | ~832 | `s-bg` + doodle + counter + MYTH card (red top border, Permanent Marker label + 2-line myth text) + TRUTH card (green top border, label + truth text) + `→` | **USE** — busting a specific named misconception. Only when script explicitly contradicts a common belief. |
| **C4** | ~868 | `s-bg` + doodle + counter + "BE REAL." (Permanent Marker, #FF6B6B) + 3-item checklist (checkmark + excuse text, ADADAD) + Caveat footer + `→` | **USE** — listing excuses the dad makes by name. Only when script calls out avoidance behaviours. |

#### TYPE D — Proof / Stat

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **D1** | ~956 | `s-bg` + doodle + counter + Permanent Marker label (#34C363) + giant number (Inter ExtraBold, #34C363, 200–230px) + 1-line context (white, 28px) + Caveat quote (ADADAD) + `→` | **USE** — one real stat from the script. Number must exist in script — never invent data. |
| **D2** | ~1075 | Full-bleed photo + heavy overlay + counter + Permanent Marker label + giant number overlaid + context text + `→` | **SKIP** — requires real photo. Replace with D1 or D3. |
| **D3** | ~995 | `s-bg` + doodle + counter + two-column split: left (red, negative stat) + right (green, positive stat) + 1-line footer + `→` | **USE** — two opposing numbers. Only when script has explicit contrasting figures (e.g. "68% quit vs 94% with a system"). |
| **D4** | ~1031 | `s-bg` + doodle + counter + headline + 3 progress bar rows (label + filled bar + milestone text) + `→` | **USE** — measurable timeline with specific checkpoints. Only when script describes a progression (Week 1 / Week 4 / Week 12). |

#### TYPE E — Transition / Bridge

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **E1** | ~1117 | `s-bg` + doodle + counter + Caveat opener line (white, 44px) + Inter ExtraBold pivot statement (#34C363, 64px) + optional ADADAD sub-line + `→` | **USE** — pivot from problem to solution. 2–3 text elements only. No numbered section. |
| **E2** | ~1220 | `s-bg` + doodle + counter + centered `#292929` card (green top border + "PART 2" label + section title) + ADADAD description below + `→` | **USE** — section divider for multi-section carousels (2+ clearly distinct parts). |
| **E3** | ~1153 | `s-bg` + doodle + counter + green bar top + Caveat quote (white, 52px, italic-feel) + green bar bottom + ADADAD 1-line footer + `→` | **USE** — quotable one-liner that deserves its own moment. Philosophical statement or insight. |
| **E4** | ~1184 | `s-bg` + doodle + counter + pill nav row (Part 1 ✓ / Part 2 active / Part 3 inactive) + big headline + ADADAD description + `→` | **USE only** for carousels with 3+ explicitly named parts and 12+ slides. Otherwise use E2. |

#### TYPE F — Pattern Interrupt / Breath (max 1 per carousel)

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **F1** | ~1272 | Full-bleed photo + minimal overlay. No counter, no logo, no text. | **SKIP** — requires real photo. |
| **F2** | ~1338 | `s-bg` + doodle + two massive ALL CAPS Inter lines (line 1 white, line 2 #34C363, 130px). No counter, no brand text. | **USE** — visual pause in long carousels (9+ slides). Words must be a punchy truth ("SLOW IS / SUSTAINABLE."). |
| **F3** | ~1287 | `s-bg` + doodle + two giant Caveat lines (white + #34C363, 170px). No counter, no brand text. | **USE** — breather for motivational carousels. One word-pair truth ("just / start."). |
| **F4** | ~1310 | `s-bg` + doodle + giant faint number background (380px, 7% opacity) + two green horizontal rules + "TIP N OF N" Inter ExtraBold (88px). | **USE** — numbered tip progress marker. Good for listicle carousels where showing progress aids retention. |

#### TYPE G — Recap / Summary (second-to-last slide)

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **G1** | ~1379 | `s-bg` + doodle + counter (←) + "QUICK RECAP" (Permanent Marker, #34C363, 60px) + 3–5 numbered bullet rows (plain digits in green span + white text, 36px) + Caveat "Save this." footer | **USE** — default recap. 3+ distinct tips worth summarising. |
| **G2** | ~1518 | `s-bg` + doodle + counter (←) + "QUICK RECAP" + 2×3 grid of emoji icon cards (green top border, `#292929` bg, emoji + 2-word label) | **USE only** when carousel has exactly 6 clean takeaways that map naturally to a single emoji each. |
| **G3** | ~1417 | `s-bg` + doodle + counter (←) + "REMEMBER THIS —" (Permanent Marker, #34C363) + one giant Inter ExtraBold statement (100px, white) + green bar + Caveat "Save it. Share it." | **USE** — when the whole carousel's lesson distills to one unforgettable line. Use instead of G1. |
| **G4** | ~1449 | `s-bg` + doodle + counter (←) + "YOUR SCORECARD" (Permanent Marker) + 3 habit rows (`#292929` card + habit label + 7 day-dot squares filled/empty) + Caveat footer | **USE only** for habit/routine carousels where the output is a weekly habit tracker. |

#### TYPE H — CTA (always last slide, no counter)

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **H1** | ~1591 | `s-bg` + doodle + centered logo (80px, `mix-blend-mode:screen`) + green bar + CTA text (Inter Bold, 44px, white, 2 lines) + green bar + ADADAD follow line + Permanent Marker `@dadfit.in` (52px, #34C363) | **USE** — default CTA for all carousels. |
| **H2** | ~1694 | `s-bg` + circular founder photo (200px) + logo + name + credentials + green divider + Caveat personal message + Inter `DM "START"` CTA | **SKIP** — requires founder photo. |
| **H3** | ~1631 | `s-bg` + doodle + radial green glow bg + "D A D F I T" (spaced, ADADAD) + big white headline + green pill button ("DM 'DAD'", Permanent Marker) + ADADAD `dadfit.in` | **USE** — only when CTA explicitly asks users to DM a keyword. |
| **H4** | ~1659 | `s-bg` + doodle + `#292929` testimonial card (green top border + Caveat quote + member attribution) + green bar + follow prompt + Permanent Marker `@dadfit.in` | **USE only** when you have a real named testimonial from the script. Never fabricate. |

---

## Design Token Cheat-Sheet

| Token | Value | Usage |
|-------|-------|-------|
| Slide BG | `#1E1E1E` | All slide backgrounds |
| Card BG | `#292929` | Callout boxes, inset cards |
| Brand green | `#34C363` | Section numbers, accents, CTA text |
| Primary text | `#FFFFFF` | Headlines, body |
| Secondary text | `#ADADAD` | Sub-labels, counter, URL, caveat |
| Warning red | `#FF6B6B` | Pain labels ONLY |
| Font body | Inter | All structural text |
| Font warm | Caveat | Annotations, coach asides |
| Font impact | Permanent Marker | Pain labels, challenge text |

---

## STEP 2 — Fill content vars for each slide

For each slide, produce a slide object with type + label + vars. The renderer substitutes every `{{VAR}}` in the matching snippet file.

```json
{
  "type": "A1",
  "slide_no": 1,
  "label": "Cover",
  "vars": {
    "COUNTER": "01 / 10",
    "HEADLINE": "YOU DON'T NEED <span style=\"color:#34C363;\">MORE TIME</span>",
    "SUBTEXT": "You need a smarter 20-minute plan.",
    "DOODLE_SRC": "../doodles/7-d-01.png",
    "DOODLE_ALT": "A cracked hourglass"
  }
}
```

### Global var rules

| Rule | Detail |
|------|--------|
| `DOODLE_SRC` | Always `"../doodles/{running_no}-d-{slide_no:02d}.png"` — e.g. slide 3 of carousel #7 → `"../doodles/7-d-03.png"` |
| `DOODLE_ALT` | 1-line description of the doodle subject |
| `COUNTER` | Format `"0N / NN"` zero-padded — e.g. `"03 / 10"` |
| HTML in values | Allowed — use `<span style="color:#34C363;">word</span>` for green; `<br>` for line breaks. Escape internal quotes with `\"` |
| `LOGO_SRC` | **Never supply** — injected automatically |
| Optional vars | **Omit the key entirely** if not needed — do not pass an empty string |

### Vars reference by slide type

**A1** — Cover: Text-Only  
`COUNTER`, `HEADLINE` (4–7 words ALL CAPS, 1–2 in green span), `SUBTEXT` (1 sentence), `DOODLE_SRC`, `DOODLE_ALT`

**A4** — Cover: Caveat Hook  
`COUNTER`, `CAVEAT_OPENER` (e.g. `"Be honest —"`), `HEADLINE`, `SUBTEXT`, `DOODLE_SRC`, `DOODLE_ALT`

**A5** — Cover: Challenge  
`COUNTER`, `CHALLENGE_WORD` (1 word, e.g. `"CHALLENGE"`), `HEADLINE`, `SUBTEXT`, `DOODLE_SRC`, `DOODLE_ALT`

**B1** — Content: Body  
`COUNTER`, `HEADLINE` (3–5 words), `BODY` (2–3 lines, `<br>` separators), `CAVEAT` *(optional)*, `DOODLE_SRC`, `DOODLE_ALT`

**B4** — Content: Callout Box  
`COUNTER`, `HEADLINE`, `BODY` (1–2 lines context), `CALLOUT` (1 punchy action sentence — replaces body, not alongside it), `DOODLE_SRC`, `DOODLE_ALT`

**B5** — Content: Wrong vs Right  
`COUNTER`, `HEADLINE`, `WRONG_ITEMS` (HTML `<li>` items, 2–3), `RIGHT_ITEMS` (HTML `<li>` items, 2–3), `CAVEAT`, `DOODLE_SRC`, `DOODLE_ALT`

**B6** — Content: 3-Step Protocol  
`COUNTER`, `HEADLINE`, `STEP1`, `STEP2`, `STEP3` (one line each), `CAVEAT`, `DOODLE_SRC`, `DOODLE_ALT`

**C1** — Pain: Empathy Frame  
`COUNTER`, `PAIN_LABEL` (e.g. `"SOUND FAMILIAR?"`), `PAIN_LINE1`, `PAIN_LINE2`, `REFRAME` (1-line green pivot), `DOODLE_SRC`, `DOODLE_ALT`

**C3** — Myth vs Truth  
`COUNTER`, `MYTH_TEXT`, `TRUTH_TEXT`, `DOODLE_SRC`, `DOODLE_ALT`

**C4** — Excuse Checklist  
`COUNTER`, `EXCUSE1`, `EXCUSE2`, `EXCUSE3`, `C4_CAVEAT`, `DOODLE_SRC`, `DOODLE_ALT`

**D1** — Big Stat  
`COUNTER`, `STAT_LABEL` (e.g. `"DID YOU KNOW?"`), `BIG_NUMBER`, `STAT_CONTEXT` (1 sentence), `STAT_CAVEAT`, `DOODLE_SRC`, `DOODLE_ALT`

**D3** — Split Stat  
`COUNTER`, `LEFT_NUMBER`, `LEFT_LABEL`, `RIGHT_NUMBER`, `RIGHT_LABEL`, `D3_FOOTER`, `DOODLE_SRC`, `DOODLE_ALT`

**D4** — Progress Bar Timeline  
`COUNTER`, `D4_LABEL`, `HEADLINE`, `ROW1_LABEL`, `ROW1_RESULT`, `ROW1_PCT` (e.g. `"25%"`), `ROW2_LABEL`, `ROW2_RESULT`, `ROW2_PCT`, `ROW3_LABEL`, `ROW3_RESULT`, `ROW3_PCT`, `DOODLE_SRC`, `DOODLE_ALT`

**E1** — Bridge: Pivot  
`COUNTER`, `E1_OPENER` (Caveat opener line), `E1_STATEMENT` (3–5 words, green), `E1_SUBTEXT` (1-line grey), `DOODLE_SRC`, `DOODLE_ALT`

**E2** — Section Label Card  
`COUNTER`, `E2_PART_LABEL` (e.g. `"PART 2 — NUTRITION"`), `E2_SECTION_TITLE`, `E2_DESCRIPTION` (1 line), `DOODLE_SRC`, `DOODLE_ALT`

**E3** — Green Accent Quote  
`COUNTER`, `E3_QUOTE` (quotable line, Caveat style), `E3_FOOTER` (attribution line), `DOODLE_SRC`, `DOODLE_ALT`

**E4** — Part Intro with Pills  
`COUNTER`, `E4_PILL1`, `E4_PILL2`, `E4_PILL3`, `E4_HEADLINE`, `E4_DESCRIPTION`, `DOODLE_SRC`, `DOODLE_ALT`

**F2** — Bold Statement (no counter/brand)  
`F2_LINE1` (white ALL CAPS, max 2 words), `F2_LINE2` (green ALL CAPS, max 2 words), `DOODLE_SRC`, `DOODLE_ALT`

**F3** — Caveat Poster (no counter/brand)  
`F3_LINE1` (white Caveat word), `F3_LINE2` (green Caveat word), `DOODLE_SRC`, `DOODLE_ALT`

**F4** — Rule Break / Tip Number  
`F4_NUMBER` (decorative digit, e.g. `"2"`), `F4_TIP_LABEL` (e.g. `"TIP 2 OF 5"`), `DOODLE_SRC`, `DOODLE_ALT`

**G1** — Recap: Numbered List  
`COUNTER`, `BULLET1`, `BULLET2`, `BULLET3`, `BULLET4` *(optional)*, `BULLET5` *(optional)*, `G1_CAVEAT` (e.g. `"Save this."`), `DOODLE_SRC`, `DOODLE_ALT`

**G2** — Recap: Icon Grid  
`COUNTER`, `CARD1_EMOJI`–`CARD6_EMOJI`, `CARD1_LABEL`–`CARD6_LABEL` (2–4 words each), `DOODLE_SRC`, `DOODLE_ALT`

**G3** — One-Line Truth  
`COUNTER`, `G3_STATEMENT` (ALL CAPS, max 6 words), `G3_CAVEAT`, `DOODLE_SRC`, `DOODLE_ALT`

**G4** — Habit Scoreboard  
`COUNTER`, `HABIT1`, `HABIT2`, `HABIT3` (2–4 words each), `HABIT1_FILLED`, `HABIT2_FILLED`, `HABIT3_FILLED` (number 0–7 as a string, e.g. `"5"`), `G4_CAVEAT`, `DOODLE_SRC`, `DOODLE_ALT`

**H1** — CTA: Text-Only  
`CTA_TEXT` (1–2 sentences), `H1_FOLLOW` (e.g. `"Join 5,000+ dads who train smarter."`), `DOODLE_SRC`, `DOODLE_ALT`

**H3** — CTA: DM Action  
`H3_HEADLINE` (ALL CAPS 3–4 words), `H3_DM_WORD` (keyword to DM, e.g. `"START"`), `DOODLE_SRC`, `DOODLE_ALT`

**H4** — CTA: Testimonial  
`H4_TESTIMONIAL` (quote text), `H4_ATTRIBUTION` (name + handle), `H4_FOLLOW_TEXT`, `DOODLE_SRC`, `DOODLE_ALT`

### Copy rules

- **Headline**: 3–7 words. ALL CAPS or Title Case. 1–2 words in `<span style="color:#34C363;">`. Never 2 full headline lines AND 3 body lines on the same slide.
- **Body (B1/B4)**: 2–3 lines max, never 4. Use `<br>` between lines.
- **Callout (B4)**: one punchy action sentence only — replaces body, never alongside it.
- **Photo slides (A2, A3, B2, B3, D2, F1)**: require real photos — **never use**. Substitute A1, B1, B4, D1, D3.
- **No decorative section numbers** on B1/B4/B5/B6. The COUNTER already shows position.

---

## STEP 3 — Write doodle prompts

For every slide **except H1** write one doodle prompt entry. These go in the `doodle_prompts` array — **do not write a separate file**.

```json
{"running_no": 7, "image_name": "7-d-02.png", "prompt": "...full prompt text..."}
```

### Prompt guidelines

- **Subject**: the single most concrete object or scene that visualises this slide's idea. Be specific ("a cracked hourglass", "a barbell with a tiny figure underneath"). No abstract nouns.
- **Style**: flat hand-drawn ink-line illustration, `#34C363` green ink only, ~3–5px stroke, no shading, no fill, no gradients.
- **Framing**: subject fills 70–80% of canvas, centered, generous empty space.
- **End every prompt** with exactly: `Pure black background #000000. Flat green line art, #34C363 ink. No fill. No shading. No text. No typography. DadFit doodle style.`

### Example prompts

> *Checklist slide:* "A flat hand-drawn ink-line doodle of a simple notepad or clipboard with three horizontal lines drawn on it, each preceded by a checkbox square with a checkmark inside. The notepad has a clean rectangular outline, a small binding clip at the top center, and the three checkbox + line elements stacked evenly down the page. Rendered in #34C363 green ink lines, stroke weight ~4px, no fill inside shapes. Fills 70–80% of canvas height, centered. Pure black background #000000. Flat green line art, #34C363 ink. No fill. No shading. No text. No typography. DadFit doodle style."

> *Exercise slide:* "A flat hand-drawn ink-line doodle of a person performing the World's Greatest Stretch — a wide lunge position with one arm reaching up and torso rotating. Simple stick-figure style showing the key movement pattern: wide lunge stance, upper body rotation visible, one arm extended upward. Rendered in #34C363 green ink lines, stroke weight ~4px, no fill. Single pose, human figure fills 70–80% of canvas height, centered. Pure black background #000000. Flat green line art, #34C363 ink. No fill. No shading. No text. No typography. DadFit doodle style."

> *Stat slide:* "A flat hand-drawn ink-line doodle showing a simple stopwatch displaying '20' with motion lines indicating speed and urgency. Circular face, button on top, two or three short motion lines radiating from the sides. Rendered in #34C363 green ink lines, stroke weight ~4px, no fill inside shapes. Single object centered on canvas, fills 70–80% of canvas height. Pure black background #000000. Flat green line art, #34C363 ink. No fill. No shading. No text. No typography. DadFit doodle style."

---

## STEP 4 — Write the JSON file and respond DONE

> ⛔ **HARD STOP:** Use **only** `create_file`. Never `run_in_terminal` with `python3 -c` or any multiline command to produce this file. Doing so violates the skill contract.

Use `create_file` to write exactly ONE file: `/tmp/carousel_{uuid}.json`

```json
{
  "uuid": "{uuid}",
  "running_no": {running_no},
  "folder_name": "{running_no}_{uuid}",
  "page_title": "HOOK HEADLINE — DadFit Carousel",
  "slides": [
    {
      "type": "A1",
      "slide_no": 1,
      "label": "Cover",
      "vars": {
        "COUNTER": "01 / 10",
        "HEADLINE": "YOU DON'T NEED <span style=\"color:#34C363;\">MORE TIME</span>",
        "SUBTEXT": "You need a smarter 20-minute plan.",
        "DOODLE_SRC": "../doodles/{running_no}-d-01.png",
        "DOODLE_ALT": "A cracked hourglass"
      }
    }
  ],
  "doodle_prompts": [
    {"running_no": {running_no}, "image_name": "{running_no}-d-01.png", "prompt": "..."},
    {"running_no": {running_no}, "image_name": "{running_no}-d-02.png", "prompt": "..."}
  ]
}
```

**Rules:**
- Include **all slides** in the `slides` array with all required vars for each type (see Step 2 reference)
- Include **all doodle prompts** except H1
- Do **not** write `carousel.html` — the renderer does that
- Do **not** write any other file

After writing the file, respond with **only**:
```
DONE: /tmp/carousel_{uuid}.json — {N} slides
```
