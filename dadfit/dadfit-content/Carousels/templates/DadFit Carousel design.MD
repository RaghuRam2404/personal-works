# DadFit Carousel Design System

> The visual grammar for all DadFit Instagram carousels — built from first principles, grounded in the DadFit brand, and informed by four carousel reference systems.

---

## 0. Design Philosophy

DadFit carousels exist for one reason: **to earn the swipe and hold the dad's attention long enough to change his mind about fitness.**

That means every design decision must serve honesty, clarity, and momentum — never decoration for its own sake.

### The Five Core Principles

| Principle | What it means in practice |
|---|---|
| **Dark & Earned** | Every slide lives on a dark background. Light doesn't mean weak — but DadFit's darkness signals seriousness, not gloom. The green accent is reserved for things that *matter*. |
| **One Idea Per Slide** | Never cram. Headline + body copy combined: **12–15 words maximum** per slide (covers excepted; recap/CTA slides allow up to 20 words). If a dad can't absorb it in 2 seconds, cut it. |
| **Handwritten = Human** | The scribble fonts exist to remind the reader there's a real coach speaking — not a brand machine. Use them for emphasis, never for body text. |
| **Green Is Earned** | `#34C363` appears only on things worth noticing: key stats, action words, the final CTA, section numbers. Overuse kills its power. |
| **Doodle = Concept** | Every non-full-bleed slide carries one AI-generated thematic doodle in the lower-right corner at ~17% opacity. One doodle per carousel — its subject reflects the carousel's topic. It adds warmth and visual continuity without competing with content. Full-bleed image slides (A3, B3, D2, F1) are excluded. |

---

## 1. Color System — Carousel Edition

Pulled directly from the DadFit brand guide, adapted for carousel usage.

### Palette

| Role | Hex | Carousel Usage |
|---|---|---|
| **Slide Background (primary)** | `#1E1E1E` | Default background for all slides |
| **Card / Layer Background** | `#292929` | Inset content cards, callout boxes, stat blocks |
| **Brand Accent (green)** | `#34C363` | Section numbers, keyword highlights, CTA text, divider lines, borders on key elements |
| **Primary Text** | `#FFFFFF` | Headlines, body copy on dark slides |
| **Secondary / Caption Text** | `#9A9A9A` | Subtitles, labels, URL, slide counter |
| **Negative / Warning** | `#FF6B6B` | Bad habits, unhealthy stats, things to AVOID — used sparingly |
| **Overlay / Shadow** | `rgba(0,0,0,0.55)` | Applied on top of photos before placing text |

### Accent Usage Rules

- `#34C363` is used as a **line/bar accent**, **number highlight**, **keyword color in body**, and **CTA button color** — never as a full slide background.
- `#FF6B6B` appears only when the slide is explicitly calling out a mistake, pain, or warning. Never decorative.
- No gradients on text. Gradients allowed only on overlay backgrounds over images.
- The green-to-transparent gradient `linear-gradient(180deg, #34C363 0%, rgba(52,195,99,0) 100%)` is the only allowed gradient on non-photo slides.

### Slide Background Variants

| Variant | Usage |
|---|---|
| Flat `#1E1E1E` | Default — all text-only slides, explainer slides |
| `#1E1E1E` + dark photo overlay | All slides with a background image |
| `#292929` inner card on `#1E1E1E` outer | Content cards, stat blocks, inset info |
| Photo with `rgba(0,0,0,0.55)` + optional green gradient from bottom | Full-bleed image slides |

---

## 2. Typography System

DadFit carousels use **three fonts**. Each has a distinct emotional job.

### Font 1 — Inter (Brand Font)

> Structural, clear, trustworthy. Used for all body copy and most headlines.

| Role | Weight | Size (at 1080px) | Case | Color |
|---|---|---|---|---|
| Slide counter (e.g., `01 / 06`) | 400 Regular | 18–20px | as-is | `#9A9A9A` |
| Logo / Wordmark | 700 Bold | 22–26px | Title Case | `#FFFFFF` |
| Tagline / URL | 400 Regular | 14–16px | lowercase | `#9A9A9A` |
| Sub-headline / label | 500 Medium | 18–22px | Sentence | `#9A9A9A` |
| Body / explanation | 400 Regular | 22–26px | Sentence | `#FFFFFF` |
| Content headline | 700 Bold | 44–56px | Sentence / Title | `#FFFFFF` |
| Cover headline | 800 ExtraBold | 80–110px | ALL CAPS | `#FFFFFF` |
| Section number | 800 ExtraBold | 72–90px | as-is | `#34C363` |
| Stat / big number | 800 ExtraBold | 90–120px | as-is | `#34C363` |

### Font 2 — Caveat (Expression Font — Warmth)

> Casual, handwritten, coach-at-the-whiteboard energy. Use for personal annotations, "dad-to-dad" asides, or underlining a human truth. Never for body text.

| Role | Usage | Color options |
|---|---|---|
| Callout annotation | "This is the one most dads miss ↓" | `#FFFFFF` or `#34C363` |
| Pull-quote emphasis | A single key line from a tip | `#FFFFFF` |
| Slide subtitle on covers | When the tone is reflective or personal | `#9A9A9A` |
| Handwritten label on stat slides | e.g., "that's 3× the average" | `#34C363` |

> Caveat weight: Regular (400) for annotations, Bold (700) for emphasis.
> Size: 24–42px. Never larger — it should feel like a friendly margin note, not a headline.

### Font 3 — Permanent Marker (Expression Font — Impact)

> Bold marker-pen energy. Use for hook words, challenge-style covers, or anything that needs to feel like it was written in a hurry by someone who means it.

| Role | Usage | Color options |
|---|---|---|
| Hook word on cover | Single word or short phrase punched over a headline | `#34C363` or `#FFFFFF` |
| Challenge / dare text | e.g., "YOU READY?" on a challenge cover | `#34C363` |
| Stats with attitude | e.g., "68% of dads quit by week 3" (on a pain slide) | `#FF6B6B` |
| "Cheat label" highlight | A circled or underlined word over body text | `#34C363` |

> Permanent Marker is always uppercase or title-case, never sentence case.
> Size: 36–72px. Used for 1–5 words maximum per slide.

### Font Usage Summary

| Font | Feel | Max usage per slide | Avoid |
|---|---|---|---|
| **Inter** | Clean, structural, trustworthy | Entire slide if needed | Decorative overuse |
| **Caveat** | Personal, human, coach-like | 1 annotation or 1 sub-line | Body copy, headlines |
| **Permanent Marker** | Bold, punchy, raw energy | 1 short phrase (1–5 words) | Body text, captions |

---

## 3. Slide Template Library

Every DadFit carousel is assembled from **8 core slide types**. Each type has multiple layout variants depending on whether an image is present.

---

### TYPE A — Cover / Hook Slide

> **Goal:** Stop the scroll. Promise something the dad wants. Force the first swipe.
> **Narrative job:** Open a loop the reader must close.

---

#### A1 — Cover: Text-Only

*Best for: bold claim carousels, numbered tip series, challenge carousels*

```
┌─────────────────────────────────────────────┐
│  DAD FIT    ·    dadfit.in         [01/07]  │
│                                              │
│                                              │
│  THE REAL REASON                             │
│  YOU'RE NOT                                  │
│  LOSING WEIGHT                               │
│                                              │
│  ← short 1–2 line sub-text here →           │  ← Inter Regular, #9A9A9A
│                                              │
│ ─────────────  (green 4px bar)  ─────────── │  ← #34C363
│                                              │
│  Swipe to find out  →                        │  ← Caveat, #9A9A9A
└─────────────────────────────────────────────┘
```

**Layout rules:**
- Dark `#1E1E1E` background, full slide
- Logo top-left, counter top-right — both in `#9A9A9A`
- Main headline: Inter ExtraBold, 80–110px, ALL CAPS, `#FFFFFF` — key word(s) swapped to `#34C363`
- Sub-text (1–2 lines): Inter Regular, 22px, Sentence case, `#9A9A9A`
- Green 4px horizontal divider line below sub-text, 80–120px wide, left-aligned
- Swipe prompt: Caveat, 28px, `#9A9A9A` or `#FFFFFF`, bottom zone

**Key word color rule:** The single most important word or phrase in the headline gets `#34C363`. Everything else stays `#FFFFFF`. On covers about pain/mistakes, the negative word gets `#FF6B6B` instead.

---

#### A2 — Cover: With Framed / Inset Image

*Best for: transformation reveals, "real story" carousels, before/after setups*

```
┌─────────────────────────────────────────────┐
│  DAD FIT    ·    dadfit.in         [01/07]  │
│                                              │
│  YOU DON'T NEED          ┌──────────────┐   │
│  2 HOURS.                │              │   │  ← framed image
│  YOU NEED                │  [PHOTO]     │   │     border: 1px #34C363
│  20 MINUTES.             │              │   │     radius: 15px
│                          └──────────────┘   │
│                                              │
│  ─────  dadfit.in  ─────────────────  →     │
└─────────────────────────────────────────────┘
```

**Layout rules:**
- Headline occupies left ~55% of the slide, vertically centered
- Image framed in a `15px` border-radius card; thin `1px solid #34C363` border
- Image sits right ~38%, vertically centered to match headline midpoint
- Background remains `#1E1E1E` — the photo doesn't take over the slide

---

#### A3 — Cover: With Full-Bleed Background Image

*Best for: emotional hooks, aspirational covers, "imagine if" setups*

```
┌─────────────────────────────────────────────┐
│  [PHOTO fills entire background]             │
│  [dark overlay: rgba(0,0,0,0.55)]            │
│  [+ green gradient from bottom]              │
│                                              │
│  DAD FIT    ·    dadfit.in         [01/07]  │
│                                              │
│                                              │
│  HOW RAGHU LOST                              │
│  12 KG WITHOUT                               │
│  QUITTING CHAI                               │
│                                              │
│  The real-life playbook for Indian dads.     │
│  ─────────────────────────────────      →   │
└─────────────────────────────────────────────┘
```

**Layout rules:**
- Photo fills 100% of slide; apply `rgba(0,0,0,0.55)` overlay + `linear-gradient(180deg, transparent 0%, rgba(30,30,30,0.7) 60%, #1E1E1E 100%)` from the bottom
- All text sits in the lower 60% of the slide where the gradient is darkest
- Headline: Inter ExtraBold, ALL CAPS, `#FFFFFF` with key word in `#34C363`
- Photo must be authentic — real training/lifestyle, Indian context, not stock-photo gym perfection
- Border-radius on the photo: none (it's a background layer)

---

### TYPE B — Content / Explainer Slide

> **Goal:** Deliver one idea with enough clarity that the reader immediately understands and values it.
> **Narrative job:** Reward the swipe. Make the reader feel informed.

---

#### B1 — Content: Text-Only (Clean)

*Best for: tips, principles, numbered lists, steps*

```
┌─────────────────────────────────────────────┐
│  DAD FIT                           [03/07]  │
│                                              │
│  ③                                          │  ← Inter ExtraBold, #34C363, 90px
│                                              │
│  EAT BEFORE                                  │  ← Inter ExtraBold, #FFFFFF, 52px
│  YOU WORK OUT                                │  ← key word: #34C363
│                                              │
│  Your body burns more fat when it has        │  ← Inter Regular, #FFFFFF, 22px
│  something to burn. Fasted workouts work     │
│  for some — but for most Indian dads         │
│  with morning chai + breakfast habits,       │
│  fuelling up wins every time.                │
│                                              │
│  ─────────  dadfit.in  ──────────────────   │
└─────────────────────────────────────────────┘
```

**Layout rules:**
- `#1E1E1E` background
- Section number: Inter ExtraBold, `#34C363`, 80–90px — top-left of content zone
- Headline: 2–3 lines, Inter ExtraBold, 48–56px, `#FFFFFF` — one key phrase `#34C363`
- Body: Inter Regular, 22–24px, `#FFFFFF`, max 6 lines, relaxed line-height (1.7)
- Optional: Caveat annotation in margin (e.g., "← most dads miss this") in `#34C363` at 28px
- URL: bottom bar, `#9A9A9A`, 14px

---

#### B2 — Content: With Inset / Side Image

*Best for: technique slides, food slides, equipment tips, form demonstrations*

```
┌─────────────────────────────────────────────┐
│  DAD FIT                           [04/07]  │
│                                              │
│  ④                                          │
│                                              │
│  THE 10-MINUTE    ┌──────────────────────┐  │
│  ROUTINE THAT     │                      │  │
│  WORKS ON         │      [PHOTO]         │  │
│  ANY SCHEDULE     │                      │  │
│                   └──────────────────────┘  │
│  3 sentences of body copy here.             │
│                                             │
│  ─────  dadfit.in  ──────────────────────  │
└─────────────────────────────────────────────┘
```

**Layout variants:**
- **Left-text / Right-image** (standard): Headline + body left ~55%; framed photo right ~40%
- **Left-image / Right-text** (alternating): Use on every other content slide to create visual rhythm

**Image frame rules:**
- Border radius: `15px`
- Border: `1px solid rgba(52, 195, 99, 0.3)` (subtle green glow)
- Image sits on a `#292929` card background before the photo
- No text overlaid on the image on content slides — keep it clean

---

#### B3 — Content: With Full-Bleed Background Image

*Best for: emotion-first content, aspirational statements, lifestyle tips*

```
┌─────────────────────────────────────────────┐
│  [PHOTO — full background with overlay]      │
│                                              │
│  DAD FIT                           [05/07]  │
│                                              │
│                                              │
│                                              │
│  ⑤                                          │
│                                              │
│  SLEEP IS YOUR                               │
│  MOST UNDERRATED                             │
│  WORKOUT                                     │
│                                              │
│  2 lines of body text max when photo is bg.  │
│  ────────  dadfit.in  ──────────────  →     │
└─────────────────────────────────────────────┘
```

**Rules:**
- Same photo overlay system as A3 (dark overlay + green gradient from bottom)
- Limit body text to 2 lines maximum — the photo carries emotional weight, don't fight it
- Section number: `#34C363`, bottom-left of content zone
- Never use `#FF6B6B` as an accent when a photo is the background — too chaotic

---

#### B4 — Content: Text-Only with Callout Box

*Best for: tips with a key takeaway that needs to be visually separated*

```
┌─────────────────────────────────────────────┐
│  DAD FIT                           [03/07]  │
│                                              │
│  ③                                          │
│                                              │
│  PROTEIN ISN'T                               │
│  JUST FOR GYMBROS                            │
│                                              │
│  Every Indian dad needs 0.8–1g per kg of     │
│  bodyweight. Most get less than half of      │
│  that from dal-roti-sabzi alone.             │
│                                              │
│ ┌─────────────────────────────────────────┐ │
│ │  ✦  Quick win: add 2 eggs or 100g paneer│ │  ← #292929 card, green left border
│ │      to one meal today.                 │ │
│ └─────────────────────────────────────────┘ │
│  ─────  dadfit.in  ──────────────────────   │
└─────────────────────────────────────────────┘
```

**Callout box rules:**
- Background: `#292929`
- Left border: `4px solid #34C363`
- Border radius: `10px`
- Padding: `16px 20px`
- Text: Inter Regular, 20px, `#FFFFFF`; the checkmark or bullet: `#34C363`

---

### TYPE C — Problem / Pain Slide

> **Goal:** Make the dad feel seen. Name his exact pain before offering the solution.
> **Narrative job:** Build empathy and tension. The solution slide must follow.

---

#### C1 — Pain: Text-Only

```
┌─────────────────────────────────────────────┐
│  DAD FIT                           [02/07]  │
│                                              │
│  SOUND FAMILIAR?                             │  ← Permanent Marker, #FF6B6B, 52px
│                                              │
│  You wake up tired.                          │  ← Inter Bold, #FFFFFF, 42px
│  You skip the workout.                       │
│  You tell yourself                           │
│  "tomorrow."                                 │
│                                              │
│  That's not laziness.                        │  ← Inter Regular, #9A9A9A, 22px
│  That's a system problem.                    │
│  We fix the system.                          │
│                                              │
│  ─────  dadfit.in  ──────────────────────   │
└─────────────────────────────────────────────┘
```

**Rules:**
- The "pain label" (e.g., "SOUND FAMILIAR?") uses Permanent Marker in `#FF6B6B`
- The pain description uses Inter Bold, `#FFFFFF`, at a slightly larger size than normal body — it's not a headline but it needs weight
- The pivot/reframe line at the bottom uses Inter Regular, `#9A9A9A`
- Consider: Caveat annotation near the pain lines (e.g., "← this is 9 out of 10 dads") in `#FF6B6B`

---

#### C2 — Pain: With Image (Relatable Scene)

*Use when a photo can dramatize the pain — e.g., an exhausted dad on a couch*

- Same layout as B2 (left-text, right-image) but headline uses `#FF6B6B` accent word instead of green
- Image frame border switches to `1px solid rgba(255, 107, 107, 0.3)` (red tint)
- Keep the overlay system if using full-bleed background

---

### TYPE D — Proof / Stat Slide

> **Goal:** Back up the claim with a number that is impossible to ignore.
> **Narrative job:** Kill doubt. Make the reader feel the data.

---

#### D1 — Stat: Text-Only (Big Number)

```
┌─────────────────────────────────────────────┐
│  DAD FIT                           [04/07]  │
│                                              │
│                                              │
│           73%                                │  ← Inter ExtraBold, #34C363, 140px
│                                              │
│    of Indian men over 35 have                │  ← Inter Regular, #FFFFFF, 26px
│    inadequate protein intake.                │
│                                              │
│    ────────────────────────────────────  ✦  │  ← #34C363 divider
│                                              │
│    "That's not a gym problem.                │  ← Caveat, #9A9A9A, 30px
│     That's a food awareness problem."        │
│                                              │
│  ─────  dadfit.in  ──────────────────────   │
└─────────────────────────────────────────────┘
```

**Rules:**
- The stat is the entire slide's hero — Inter ExtraBold, `#34C363`, at least 120px, centered or left-anchored
- Supporting context text: Inter Regular, `#FFFFFF`, 24–26px
- Optional Caveat pull-quote below the divider (coach's voice)
- Optional Permanent Marker "label" above the stat (e.g., "THE TRUTH:")
- Use `#FF6B6B` for stats about negative outcomes (e.g., "68% of dads quit within 3 weeks")

---

#### D2 — Stat: With Image

- Stat number overlaid on a darkened photo background (full-bleed image, heavy `rgba(0,0,0,0.65)` overlay)
- Same typography rules as D1
- The photo should relate to the stat (e.g., a sad empty gym for a dropout stat)

---

### TYPE E — Transition / Bridge Slide

> **Goal:** Pivot the narrative. Move from problem to solution, or between two sections.
> **Narrative job:** Give the reader a moment to breathe and orient. Sustain the swipe.

---

#### E1 — Bridge: Minimal Text

```
┌─────────────────────────────────────────────┐
│  DAD FIT                           [04/07]  │
│                                              │
│                                              │
│  But here's the thing —                      │  ← Caveat Bold, #FFFFFF, 44px
│                                              │
│  You already know what to do.               │  ← Inter ExtraBold, #34C363, 58px
│                                              │
│  You just need a system                      │  ← Inter Regular, #9A9A9A, 26px
│  that works with your life.                  │
│                                              │
│                                              │
│  ─────  dadfit.in  ──────────────────────   │
└─────────────────────────────────────────────┘
```

**Rules:**
- No section number — this is a connective slide, not a numbered content beat
- Caveat leads (coach's voice introducing the pivot)
- Inter ExtraBold carries the key re-frame line in `#34C363`
- Keep breathing room — this slide should feel lighter than the content slides

---

#### E2 — Bridge: With Section Label (for multi-section carousels)

```
┌─────────────────────────────────────────────┐
│  DAD FIT                           [05/09]  │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │  PART 2                                │ │  ← #292929 card, top green bar
│  │  The Nutrition Fix                     │ │
│  └────────────────────────────────────────┘ │
│                                              │
│  Now that we've covered movement,            │  ← Inter Regular, #9A9A9A, 22px
│  here's what to eat — without throwing       │
│  out your thali.                             │
│                                              │
│  ─────  dadfit.in  ──────────────────────   │
└─────────────────────────────────────────────┘
```

**Rules:**
- Section label card: `#292929` background, `4px solid #34C363` top border, `10px` radius
- "PART 2" label: Inter ExtraBold, `#34C363`, 18px — ALL CAPS
- Section title: Inter Bold, `#FFFFFF`, 32–36px

---

### TYPE F — Pattern Interrupt / Breath Slide

> **Goal:** Break visual monotony. Re-engage a reader who is starting to skim.
> **Narrative job:** A visual beat — no information delivery, pure atmosphere.

---

#### F1 — Breath: Full-Bleed Image (Brand-Clean)

*No logo, no text, no URL. Just the image.*

```
┌─────────────────────────────────────────────┐
│                                             │
│                                             │
│                                             │
│          [AUTHENTIC LIFESTYLE PHOTO]        │
│                                             │
│          (Dad with kid, home workout,       │
│           meal prep, morning run)           │
│                                             │
│                                             │
│                                             │
└─────────────────────────────────────────────┘
```

**Rules:**
- Use this on slide 3 or 6 in a long carousel (7+ slides)
- Image must be authentic — real Indian dad context, not stock photo
- Minimal `rgba(0,0,0,0.2)` overlay at most — the photo must breathe
- Border radius on image if displayed as a frame inside the slide: `20px`
- This slide deliberately carries no brand marks — the photo IS the brand signal

---

#### F2 — Breath: Single Bold Statement (No Image)

*A single line in massive type — almost typographic art.*

```
┌─────────────────────────────────────────────┐
│                                             │
│                                             │
│                                             │
│     SLOW IS                                 │  ← Inter ExtraBold, #FFFFFF, 110px
│                                             │
│       SUSTAINABLE.                          │  ← Inter ExtraBold, #34C363, 110px
│                                             │
│                                             │
│                                             │
└─────────────────────────────────────────────┘
```

**Rules:**
- No logo, no counter, no URL — clean canvas
- 2–3 words maximum
- Line 1: `#FFFFFF`; Line 2 (or the key word): `#34C363`
- Use Inter ExtraBold at 100–120px
- No body text. The emptiness is intentional.

---

### TYPE G — Recap / Summary Slide

> **Goal:** Compress the full carousel into a scannable takeaway list.
> **Narrative job:** Complete the reading contract. Reward those who made it through.

---

#### G1 — Recap: Numbered List (Text-Only)

```
┌─────────────────────────────────────────────┐
│  DAD FIT                           [07/08]  │
│                                              │
│  QUICK RECAP                                 │  ← Permanent Marker, #34C363, 44px
│                                              │
│  ①  Protein first, always                   │  ← Inter Bold, #FFFFFF, 26px
│  ②  30 min beats 0 min every time          │
│  ③  Sleep is training                       │
│  ④  Track food for 7 days, judge later      │
│  ⑤  Your family is your why                 │
│                                              │
│  Save this. Come back to it.                 │  ← Caveat, #9A9A9A, 28px
│                                              │
│  ─────  dadfit.in  ──────────────────────   │
└─────────────────────────────────────────────┘
```

**Rules:**
- "QUICK RECAP" label: Permanent Marker, `#34C363`, 40–48px
- List items: Inter Bold, `#FFFFFF`, 24–28px
- Numbers/bullets: `#34C363`
- Caveat save-prompt at the bottom is optional but highly recommended for Instagram save-rate
- Vertical spacing between list items: generous — don't cram

---

#### G2 — Recap: With Icons / Visuals

- Replace the list numbers with small icon illustrations (dumbbell, plate, moon, calendar, family) — each `#34C363` icon above the corresponding text
- Layout as a 2×3 or 3×2 grid of icon + label pairs
- Each cell on a `#292929` mini-card, `10px` radius

---

### TYPE H — CTA / Last Slide

> **Goal:** Convert the momentum of completed reading into one single action.
> **Narrative job:** Close the loop. Give the reader exactly ONE thing to do.

---

#### H1 — CTA: Text-Only (Follow / Save)

```
┌─────────────────────────────────────────────┐
│                                              │
│             DAD FIT                          │  ← Inter ExtraBold, #FFFFFF, 36px
│          dadfit.in                           │  ← Inter Regular, #9A9A9A, 18px
│                                              │
│  ─────────────────────────────────────── ─  │  ← #34C363, 4px
│                                              │
│      If this helped, save it.               │  ← Inter Bold, #FFFFFF, 32px
│      Send it to a dad who needs it.         │
│                                              │
│  ─────────────────────────────────────── ─  │
│                                              │
│      Follow for more honest fitness          │  ← Inter Regular, #9A9A9A, 22px
│      advice for Indian dads.                │
│                                              │
│      @dadfit.in                              │  ← Permanent Marker, #34C363, 38px
│                                              │
└─────────────────────────────────────────────┘
```

**Rules:**
- Center-aligned layout — this is the one slide where center alignment is standard
- Logo (`logo.png`) becomes the hero — 80px on H1 (centered), 72px on H2 (identity block); far larger than the 52px header logo on the cover slide
- Two green dividers flank the main action line
- One primary CTA ("save it"). One secondary CTA ("follow"). Never three.
- `@dadfit.in` handle: Permanent Marker, `#34C363` — the handle should feel like a signature

---

#### H2 — CTA: With Founder / Persona Image

*Best for: Trust-building CTAs, coaching offer carousels, personal story carousels*

```
┌─────────────────────────────────────────────┐
│                                              │
│  ┌──────────────┐                            │
│  │   [RAGHU's   │  DAD FIT                   │  ← logo right of image
│  │    PHOTO]    │  Raghu Ram                  │
│  │              │  NASM Certified Trainer     │  ← Inter Medium, #9A9A9A, 18px
│  └──────────────┘  dadfit.in                  │
│  (rounded, green border)                     │
│  ────────────────────────────────────────── │
│                                              │
│  This is exactly what I teach               │  ← Caveat Bold, #FFFFFF, 34px
│  inside the DadFit program.                  │
│                                              │
│  DM me "START" to learn more.               │  ← Inter ExtraBold, #34C363, 32px
│                                              │
│  ─────  dadfit.in  ──────────────────────   │
└─────────────────────────────────────────────┘
```

**Rules:**
- Founder photo: circular crop, `4px solid #34C363` border, ~200–220px diameter
- Founder photo position: left-aligned at top
- Caveat for the personal message ("This is exactly what I teach...") — makes it feel human
- The single action verb is in Inter ExtraBold, `#34C363`
- DM keyword ("START") styled in Permanent Marker inline for emphasis

---

## 4. Navigation System

### Slide Counter

- Format: `01 / 06` — with spaces around the slash
- Position: top-right corner, ~60px from right, ~55px from top
- Font: Inter Regular, 18px, `#9A9A9A`
- Present on ALL slides except Pattern Interrupt (F1, F2) and standalone single-image slides

### Swipe Arrow

- Right-pointing `→` on all slides except the last
- Left-pointing `←` on the last content slide (reverse arrow signals: you've reached the end)
- No arrow on CTA slide — replace with save/follow icons
- Arrow: `#34C363` filled triangle or Inter Regular `→` character, 20–24px
- Position: bottom-right, inside the URL foot zone

### URL Footer

- Every slide (except Pattern Interrupt) carries `dadfit.in` in the footer zone
- Font: Inter Regular, 14px, `#9A9A9A`, lowercase
- Position: bottom-left, ~60px from left, ~50px from bottom
- Paired with arrow on the right side of the same baseline

---

## 5. Branding System

| Element | Slides | Position | Size / Style |
|---|---|---|---|
| "DAD FIT" wordmark | All except F1/F2 | Top-left | Inter Bold, 22–26px, `#FFFFFF` |
| Tagline (rare) | Cover only | Below wordmark | Inter Regular, 14px, `#9A9A9A` |
| URL "dadfit.in" | All except F1/F2 | Bottom-left | Inter Regular, 14px, `#9A9A9A` |
| Green divider line | All CTA + some content | Below key section | `4px × 80px`, `#34C363`, `border-radius: 2px` |
| Top green bar on cards | Bridge section cards | Top edge of card | `4px × 100% card width`, `#34C363` |
| Left green bar on callout | Callout boxes | Left edge | `4px × full height`, `#34C363` |

**Branding density by slide type:**

| Slide Type | Logo | Counter | URL | Arrow |
|---|---|---|---|---|
| Cover — text only (A1) | ✓ header | ✓ | ✓ | ✓ |
| Cover — framed image (A2) | ✗ | ✓ | ✓ | ✓ |
| Cover — full-bleed (A3) | ✗ | ✓ | ✓ | ✓ |
| Content (B1/B2/B3/B4) | ✗ | ✓ | ✓ | ✓ |
| Pain (C1/C2) | ✗ | ✓ | ✓ | ✓ |
| Stat (D1/D2) | ✗ | ✓ | ✓ | ✓ |
| Bridge (E1/E2) | ✗ | ✓ | ✓ | ✓ |
| Breath — full image (F1) | ✗ | ✗ | ✗ | ✗ |
| Breath — bold statement (F2) | ✗ | ✗ | ✗ | ✗ |
| Recap (G1/G2) | ✗ | ✓ | ✓ | ✓ |
| CTA — text only (H1) | ✓ hero (80px) | ✗ | ✓ | ✗ |
| CTA — with founder (H2) | ✓ identity (72px) | ✗ | ✓ | ✗ |

---

## 6. Complete Slide Combination Reference

Every possible slide configuration summarized for quick selection:

| Code | Type | Image | Image Style |
|---|---|---|---|
| **A1** | Cover / Hook | None | Text-only |
| **A2** | Cover / Hook | Yes | Framed / inset |
| **A3** | Cover / Hook | Yes | Full-bleed background |
| **B1** | Content / Explainer | None | Text-only |
| **B2** | Content / Explainer | Yes | Inset / side by side |
| **B3** | Content / Explainer | Yes | Full-bleed background |
| **B4** | Content / Explainer | None | Text + callout card |
| **C1** | Problem / Pain | None | Text-only |
| **C2** | Problem / Pain | Yes | Framed or full-bleed |
| **D1** | Proof / Stat | None | Big number, text-only |
| **D2** | Proof / Stat | Yes | Stat overlaid on photo |
| **E1** | Bridge / Transition | None | Minimal text |
| **E2** | Bridge / Transition | None | Section label card |
| **F1** | Pattern Interrupt | Yes | Full-bleed, no text |
| **F2** | Pattern Interrupt | None | Bold type-art, no brand |
| **G1** | Recap / Summary | None | Numbered list |
| **G2** | Recap / Summary | Icons only | Icon grid |
| **H1** | CTA / Last | None | Text-only |
| **H2** | CTA / Last | Yes | Founder/persona photo |

---

## 7. Canonical Carousel Blueprints

Pre-built arc templates for the most common DadFit content types.

### Blueprint 1 — The Numbered Tips Carousel (5–7 slides)

*Used for: "5 reasons...", "3 habits...", "7 things...", how-to lists*

| Slide | Type | Layout |
|---|---|---|
| 1 | A1 | Cover: bold claim with number promise |
| 2 | B1 | Tip #1: text-only |
| 3 | B2 | Tip #2: text + image |
| 4 | B1 | Tip #3: text-only |
| 5 | B4 | Tip #4: text + callout box |
| 6 | G1 | Recap: numbered summary |
| 7 | H1 | CTA: save / follow |

---

### Blueprint 2 — The Myth-Buster Carousel (6 slides)

*Used for: debunking fitness myths specific to Indian dads*

| Slide | Type | Layout |
|---|---|---|
| 1 | A1 | Cover: "X Things Indian Dads Get WRONG About ___" |
| 2 | C1 | Pain: name the myth and how most dads fall for it |
| 3 | D1 | Stat: hard data that disproves the myth |
| 4 | B3 | Truth #1: with lifestyle photo as background |
| 5 | B4 | Truth #2: with callout box for quick action |
| 6 | H2 | CTA: founder photo + coaching offer |

---

### Blueprint 3 — The Transformation Story Carousel (8–9 slides)

*Used for: client stories, before/after, founder story*

| Slide | Type | Layout |
|---|---|---|
| 1 | A3 | Cover: aspirational full-bleed photo + bold claim |
| 2 | C2 | Pain: before state with relatable image |
| 3 | F1 | Breath: full-bleed lifestyle image |
| 4 | E1 | Bridge: pivot to the solution |
| 5 | B2 | Step 1: text + inset image |
| 6 | B2 | Step 2: text + image (alternate side) |
| 7 | D2 | Result stat: overlaid on transformation photo |
| 8 | G1 | Recap |
| 9 | H2 | CTA: founder sign-off |

---

### Blueprint 4 — The Deep Dive Carousel (9+ slides)

*Used for: "The Ultimate Guide to ___", nutrition deep dives, training primers*

| Slide | Type | Layout |
|---|---|---|
| 1 | A2 | Cover: framed image + bold guide title |
| 2 | C1 | Pain: why this matters for dads specifically |
| 3 | E2 | Bridge: "Part 1 — ___" section card |
| 4 | B1 | Content 1 |
| 5 | B4 | Content 2 with callout |
| 6 | F2 | Breath: bold statement |
| 7 | E2 | Bridge: "Part 2 — ___" section card |
| 8 | B3 | Content 3 with photo bg |
| 9 | B1 | Content 4 |
| 10 | G2 | Recap: icon grid |
| 11 | H1 | CTA |

---

## 8. Green Line / Divider Usage Rules

The green `#34C363` line is the smallest but most loaded brand element in the system. Use it intentionally.

| Context | Width | Height | Radius | Alignment |
|---|---|---|---|---|
| Section title underline | 80–100px | 4px | 2px | Left-aligned (below headline) |
| Top bar on section label card | 100% of card width | 4px | 0px (flush to card top) | Top edge |
| Left bar on callout box | 4px | 100% of box height | 0px | Left edge |
| Footer divider on CTA slide | 100% of text zone width | 2px | 1px | Full-width between logo and copy |
| Decorative accent between logo and URL in footer | 1px | 12px | 0px | Centered vertically, between elements |

---

## 9. Green Emphasis Rule for Headline Text

In every headline, one key word or phrase earns `#34C363`. The rule for choosing which word:

1. **The specific number** — "5 habits" → "5" in green
2. **The outcome word** — "you will LOSE weight" → "LOSE" in green
3. **The identity word** — "for INDIAN DADS" → "INDIAN DADS" in green
4. **The time/effort word** — "in 20 MINUTES" → "20 MINUTES" in green
5. **The contrast word** — "work harder, not SMARTER" → "SMARTER" in green

If using `#FF6B6B` on a pain slide, the same rule applies — the single worst thing, the number that hurts, or the mistake label gets the red.

**Never green more than one phrase per headline.**

---

## 10. Typography Pairing Patterns

How the three fonts work together on a single slide.

### Pattern 1 — All-Inter (Structural)

*Use on: numbered content slides, stat slides, recap slides*

```
[Section number] → Inter ExtraBold, #34C363
[Headline]       → Inter ExtraBold, #FFFFFF (one word: #34C363)
[Body]           → Inter Regular, #FFFFFF
[Caption/URL]    → Inter Regular, #9A9A9A
```

### Pattern 2 — Inter + Caveat (Human)

*Use on: bridge slides, CTA slides, pain slides, transformation slides*

```
[Intro phrase]   → Caveat Bold, #FFFFFF (coach's voice, sets tone)
[Headline]       → Inter ExtraBold, #FFFFFF / #34C363
[Body]           → Inter Regular, #FFFFFF
[Annotation]     → Caveat Regular, #9A9A9A (margin note feel)
```

### Pattern 3 — Inter + Permanent Marker (Punch)

*Use on: cover hooks, myth-buster slides, challenge carousels*

```
[Challenge tag]  → Permanent Marker, #34C363 or #FF6B6B (top label)
[Headline]       → Inter ExtraBold, #FFFFFF
[Body]           → Inter Regular, #FFFFFF
[URL]            → Inter Regular, #9A9A9A
```

### Pattern 4 — All Three (Signature)

*Use on: CTA slides (H2), cover slides for major campaigns — sparingly*

```
[Marker label]   → Permanent Marker, #34C363 (one word/label at top)
[Headline]       → Inter ExtraBold, #FFFFFF
[Body]           → Caveat Bold, #FFFFFF (personal sign-off)
[Handle]         → Permanent Marker, #34C363 (@dadfit.in)
```

---

---

## 12. Word Count & Alignment Rules

### Word Count Per Slide

| Slide category | Max word count (headline + body combined) |
|---|---|
| **Cover / Hook** (A1–A3) | No limit — earn the swipe |
| **Content / Explainer** (B1–B4) | **12–15 words** |
| **Problem / Pain** (C1–C2) | **12–15 words** |
| **Proof / Stat** (D1–D2) | **12–15 words** (stat number is not counted) |
| **Bridge / Transition** (E1–E2) | **12–16 words** |
| **Pattern Interrupt** (F1–F2) | ≤ 3 words (F2) or zero (F1) |
| **Recap / Summary** (G1–G2) | ≤ 20 words — each list item max 4 words |
| **CTA** (H1–H2) | ≤ 15 words |

> **Counting rule:** Count all text a reader is meant to read — headlines, body, annotations, callout text. Do NOT count: slide counter, logo text, URL, arrows, section numbers.

### Alignment Rules

| Zone | Alignment |
|---|---|
| All body / headline / annotation text | **Left-aligned** (`text-align: left`) |
| H1 CTA slide (center-composition) | Center-aligned (intentional exception) |
| G2 recap card labels | Center-aligned within each card cell |
| Swipe prompt (Caveat) | Left-aligned |

> Never orphan a single word on the last line of a body text block. Rephrase or add a manual `<br>` to balance line lengths.

---

*DadFit Carousel Design System — v1.2 — April 2026*
*Built for dadfit.in by Raghu Ram, NASM Certified Trainer*

Every non-full-bleed slide carries an AI-generated thematic doodle in the lower-right corner. One doodle per carousel — generated to match the carousel's topic. Applied at low opacity, it functions as a warm visual accent, not information.

### Which Slides Carry the Doodle

A slide gets a doodle holder only if it has **no other image element** — the doodle fills the visual anchor role. If a photo is already present (framed, inset, circular, or full-bleed), it IS the visual — the doodle is skipped.

| Excluded — has a photo | Excluded — full-bleed bg | Gets doodle holder |
|---|---|---|
| A2 — Cover with framed image | A3 — Cover full-bleed | A1 — Cover text-only |
| B2 — Content with inset image | B3 — Content full-bleed | B1 — Content text-only |
| C2 — Pain with image | D2 — Stat on photo | B4 — Content + callout |
| H2 — CTA with founder photo | F1 — Breath full-bleed | C1 — Pain text-only |
| | | D1 — Stat text-only |
| | | E1, E2 — Bridge slides |
| | | F2 — Bold statement |
| | | G1, G2 — Recap slides |
| | | H1 — CTA text-only |

### Doodle Placement & Specs

| Property | Value |
|---|---|
| **Position** | Absolute, lower-right corner of the slide |
| **Size** | 340 × 340 px (at 1080 px canvas) |
| **Opacity** | 0.175 — always subdued, never dominant |
| **Object-fit** | `contain` — the full illustration is always visible |
| **Layer order** | Above the dark background, below all text content |
| **Pointer events** | None — purely decorative |

### Doodle Style Guidelines

| Principle | In practice |
|---|---|
| **One per carousel** | All slides in a carousel share the same doodle. Consistency creates continuity across the swipe. |
| **Topic-matched** | Protein carousel → sketch of a plate or dumbbell. Sleep carousel → moon + Zs. Time carousel → clock. |
| **Line-art / sketch style** | Clean line-drawing, AI-generated. White or light strokes on transparent PNG background work best at low opacity. |
| **Indian context** | Where possible, the doodle should feel culturally resonant — Indian food items, familiar home objects, Indian body types. |
| **No text in doodle** | Purely visual. No words, labels, or numbers inside the doodle image. |

### How to Activate the Doodle in a Carousel HTML File

The doodle holder is injected automatically via the inline script into every `.s-bg` slide. Set one variable to apply the image to all slides at once:

```javascript
var DOODLE_SRC = 'Images/doodle-protein.png'; // ← set your AI doodle image path here
```

Or set the src directly on each `<div class="doodle-ph">` img element if you need different doodles per slide (rare).

---

*DadFit Carousel Design System — v1.2 — April 2026*
*Built for dadfit.in by Raghu Ram, NASM Certified Trainer*
