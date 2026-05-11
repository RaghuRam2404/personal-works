---
name: landing-page-builder
description: 'Build high-converting lead generation landing pages using a proven framework: hook → value proposition → credibility → CTA. Use for: landing page creation, lead generation page, assessment landing page, quiz landing page, scorecard landing page, lead capture page, online assessment page, conversion-optimized page, frustration hook, results hook, readiness sentence. Triggers: landing page, lead gen page, lead generation landing page, build me a landing page, create landing page, assessment page, quiz funnel, scorecard marketing page, optin page.'
argument-hint: 'Describe your business, who your audience is, and the result or frustration your page should address'
---

# Landing Page Builder

Build conversion-optimized lead generation landing pages based on the framework: **hook → subheading → value proposition → credibility → CTA**. Targets 20–40% opt-in rates by matching the page structure to the audience's emotional state and desired outcome.

## Core Philosophy

> Everything is downstream from lead generation. A landing page for an online assessment or quiz is the single highest-leverage asset for consistent, qualified lead flow — across any industry.

The page does one job: convince someone to start the assessment. Every section earns the next click.

## Dependencies

Before building, ensure these skills are available:

- **brand-strategy** — Establishes or retrieves brand voice, positioning, and messaging. Run this first if no brand profile exists.
- **ui-ux-designer** — Handles layout, visual hierarchy, typography, and responsive design.
  ```
  npx skills add https://github.com/sickn33/antigravity-awesome-skills --skill ui-ux-designer
  ```

---

## Step 1 — Run Intake Interview

Ask ALL of the following before writing a single word of copy. Do not skip or combine questions — each answer shapes a different section of the page. See [./references/intake-questions.md](./references/intake-questions.md) for the complete question set with branching logic.

### Minimum required answers before proceeding:

| # | Question | Feeds Into |
|---|----------|------------|
| 1 | What business / niche is this for? | Overall context |
| 2 | Who is the target audience (be specific)? | Hook, CTA tone |
| 3 | What is the **core frustration** the audience feels even when doing things right? | Frustration hook |
| 4 | What is the **desired result** the audience wants to achieve? | Results hook / readiness sentence |
| 5 | Choose hook type: **frustration** or **results (readiness)**? | Hook variant |
| 6 | What are the **3 specific areas** the assessment will measure and improve? | Value proposition |
| 7 | Who created this assessment? What is their background, credentials, or track record? | Credibility section |
| 8 | Is there any research, data, or statistics to cite? (e.g. "85% of people struggle with X") | Credibility section |
| 9 | What does the user DO after the page? (Start quiz, book call, download?) | CTA |
| 10 | How long does it take? Is it free? What do they get immediately? | CTA micro-copy |
| 11 | Does a brand profile exist? (colors, fonts, tone of voice) | Design & copy style |
| 12 | Any competitor pages or inspiration references? | Design direction |

> If a brand profile does **not** exist, pause and invoke the **brand-strategy** skill to build one before continuing.

---

## Step 2 — Establish Brand Foundation

Invoke **brand-strategy** skill to:
- Define or load the brand's tone of voice (authoritative? warm? urgent? conversational?)
- Lock in primary colors, font pairing, and logo usage
- Extract the brand's core promise and differentiator

Pass the brand profile output into Step 4 (copy) and Step 5 (design).

---

## Step 3 — Build the Page Architecture

The page has exactly **5 sections** in a fixed order. See [./references/landing-page-structure.md](./references/landing-page-structure.md) for detailed copy formulas, character limits, and examples for each section.

| Section | Purpose | Target Outcome |
|---------|---------|----------------|
| **1. Hook** | Stop the scroll, name the frustration or the dream | Emotional resonance in < 2 seconds |
| **2. Subheading** | Direct them to act and tell them what they'll learn | Clarity on what the quiz delivers |
| **3. Value Proposition** | Name the 3 areas they'll improve | Desire to see their score |
| **4. Credibility** | Prove the assessment is worth 3 minutes | Trust earned, objections pre-empted |
| **5. CTA** | One button, five reasons to click it | 20–40% conversion to quiz start |

---

## Step 4 — Write Copy

Using intake answers and brand voice, generate copy for each section following the formulas in [./references/landing-page-structure.md](./references/landing-page-structure.md).

**Copy rules:**
- Mobile-first: the hero (sections 1–2) must be readable without scrolling on a phone
- Every sentence either builds desire or removes friction — no neutral sentences
- Never use jargon the audience hasn't already used themselves
- The CTA button text must start with a verb: "Start", "Get", "Find out", "Discover"

---

## Step 5 — Design with ui-ux-designer

Hand off the copy and brand profile to the **ui-ux-designer** skill with this brief:

```
Page type: Lead generation / quiz opt-in
Sections: Hero (hook + subheading) → Value prop (3 pillars) → Credibility (bio + stats) → CTA (single button)
Mobile-first layout
Single column, no nav distractions
CTA button above the fold on desktop AND mobile
Social proof / credibility block below the fold
```

---

## Step 6 — Review Checklist

Before finalizing, verify:

- [ ] Hook names the exact frustration OR uses a readiness sentence ("Are you ready to…?")
- [ ] Subheading contains the number of questions and the outcome ("Answer 15 questions to find out…")
- [ ] Value prop lists exactly 3 named areas to measure/improve
- [ ] Credibility section answers: who made this, what's their track record, what research backs it
- [ ] CTA covers all 5 micro-copy elements: action verb · time ("3 minutes") · free · immediate results · no risk
- [ ] Single primary CTA — no competing links or navigation
- [ ] Mobile hero fits above the fold (headline + subhead + button visible without scrolling)
- [ ] Brand voice is consistent throughout

---

## Output

Deliver:
1. **Full page copy** — headline, subhead, value prop, credibility block, CTA text
2. **HTML/CSS mockup** (via ui-ux-designer) — mobile-first, single-page layout
3. **A/B test variants** — one frustration hook version, one readiness/results hook version
4. **Recommended next step** — what quiz/assessment builder to connect this to (e.g. ScoreApp, Typeform, Outgrow)
