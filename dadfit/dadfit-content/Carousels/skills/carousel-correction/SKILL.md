# SKILL: Carousel Correction

## Purpose

Apply user-described corrections to an open `carousel.html` file — fixing typos, copy errors, layout mistakes, styling violations, design-system deviations, and structural issues in one or more slides. This skill never rewrites entire files; it makes the smallest precise change that resolves each mistake.

> ⛔ **TERMINAL RULE — READ BEFORE ANYTHING ELSE:**
> Multi-line patterns are **forbidden in any `run_in_terminal` call**. The following are illegal everywhere — zero exceptions:
> - `python3 -c "..."`
> - `python3 - <<'PYEOF' ... PYEOF`
> - Any heredoc (`<<`, `<<'EOF'`, `<<-EOF`, etc.)
> - Any shell command containing a newline or line continuation
>
> All `run_in_terminal` calls must use `mode=sync`. Always set `cwd` to `/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content` and wait for output before the next command. If a Python helper is needed, write it with `create_file` and run `python3 /path/to/file.py`. Violating this rule means the step has failed.

---

## Inputs

- **Carousel HTML path** — the currently open `carousel.html` file (e.g., `Carousels/data/batch_1/1_35f5556f-.../carousel.html`). Read the full file before touching it.
- **User-described mistakes** — free-text description of what is wrong. May include:
  - Copy / wording errors ("slide 3 headline says X but should say Y")
  - Styling violations ("the stat number is white, should be green")
  - Layout / structure issues ("slide 5 is a B2 but should be a B4")
  - Design system deviations (wrong color, wrong font, overcrowded slide, missing doodle)
  - Slide count changes ("add a new slide after slide 4", "remove slide 6")

---

## Step 1 — Read the file

Read the full `carousel.html` in one call. Do not guess at line numbers; always read first.

---

## Step 2 — Parse the mistakes

For each mistake the user described:

1. Identify the **slide number** and **element** affected (headline, body, callout, stat, footer, etc.).
2. Classify the correction type:
   - **Copy** — text content change
   - **Style** — CSS property / class change
   - **Structure** — adding, removing, or reordering HTML blocks
   - **Design-system** — restoring compliance with the rules below

3. Locate the **exact string** in the HTML that must change. Never edit by line number — always use an exact literal match with 3–5 lines of context.

---

## Step 3 — Validate against the design system

Before editing, confirm the intended fix is compliant with the DadFit design system. Key rules (condensed):

### Colors
| Token | Hex | Correct usage |
|---|---|---|
| `--primary-bg` | `#1E1E1E` | Default slide background |
| `--secondary-bg` | `#292929` | Cards, callout boxes, stat blocks |
| `--primary-green` | `#34C363` | Section numbers, key stats, CTA button, green-bar accents |
| `--danger-red` | `#FF6B6B` | Mistakes, warnings, things to AVOID only — never decorative |
| `--text-primary` | `#FFFFFF` | All body copy and headlines |
| `--text-secondary` | `#ADADAD` | Subtitles, labels, URL, slide counter |

> Use CSS custom properties (`var(--primary-green)`) — never hard-code hex values in inline styles unless the token doesn't exist.

### Typography
| Role | Font | Weight | Size (at 1080px) | Case |
|---|---|---|---|---|
| Cover headline | Inter | 800 | 80–110px | ALL CAPS |
| Content headline | Inter | 700 | 44–56px | Title / Sentence |
| Body copy | Inter | 400 | 22–26px | Sentence |
| Section number / Big stat | Inter | 800 | 72–120px | as-is |
| Annotation / aside | Caveat | 400–700 | 24–42px | Sentence |
| Hook / challenge word | Permanent Marker | — | contextual | ALL CAPS |

### Content density (non-negotiable)
- **One idea per slide.** Two separable thoughts → two slides.
- **Headline: 3–6 words max** (covers may run longer). Punchy, ALL CAPS on content slides.
- **Body copy: 2–3 lines max.** Never 4 lines.
- **Never stack more than 3 content blocks** in a single slide zone.
- If a slide is overcrowded after the fix, propose splitting it rather than cramming.

### Slide header / footer anatomy (every slide)
Every slide has:
- **Header**: Logo left (`DadFit`), slide counter right (`01 / 08`), green bar below.
- **Footer**: URL left (`dadfit.in`), swipe arrow right (→), brand bar above (optional).
- Do not alter header/footer unless the mistake is explicitly in those elements.

---

## Step 4 — Apply corrections

Use `replace_string_in_file` (or `multi_replace_string_in_file` for multiple edits) to apply each fix with **exact literal context strings**.

Rules:
- Include at least 3 lines of unchanged context before and after the changed text.
- Never replace an entire slide block unless the mistake spans the whole slide.
- One edit per distinct mistake. Do not batch unrelated changes into a single replacement.
- After each edit, mentally re-read the surrounding HTML to confirm the change is coherent.

### Adding a slide
1. Identify the slide immediately before and after the new one.
2. Copy the closest matching slide type from the existing HTML as a template.
3. Insert with `replace_string_in_file` — replace the closing `</div>` of the preceding slide's `.slide-item` wrapper with that closing tag **plus** the new slide block.
4. Update the slide counter (`01 / 08` → `01 / 09`, etc.) in all affected slides.

### Removing a slide
1. Locate the full `.slide-item` block (from opening `<div class="slide-item">` to its closing `</div>`).
2. Remove it.
3. Update slide counters in all remaining slides.

---

## Step 5 — Verify

After all edits, re-read the changed sections of the file and confirm:
- [ ] Each corrected element matches the user's intent.
- [ ] No design-system violation was introduced.
- [ ] Slide counters are still sequential and correct.
- [ ] No HTML tags were left unclosed or orphaned.

Report back to the user with a brief summary: which slides were changed, what was fixed, and any design notes (e.g., "Slide 5 body was 4 lines — trimmed to 3 to comply with density rule").

---

## Quick-reference: common mistake patterns

| Symptom | Likely cause | Fix |
|---|---|---|
| Stat number is white | Missing `color: var(--primary-green)` | Add/fix the color property |
| Body text 4 lines | Overcrowding | Cut to 3 lines max; split into two slides if meaning is lost |
| Headline too long | Hook / content headline exceeds 6 words | Trim or rephrase |
| Wrong slide counter | Slide added/removed without updating all counters | Increment/decrement all affected `s-counter` spans |
| Hard-coded hex instead of CSS var | Inline style uses `#34C363` directly | Replace with `var(--primary-green)` |
| Caveat font on body text | Wrong font family | Change to Inter |
| Green used decoratively | `--primary-green` on non-data, non-CTA element | Revert to `--text-primary` or `--text-secondary` |
| Doodle missing | `<img class="s-doodle">` block absent | Do NOT add placeholder — note it for the doodle processor step |
| Footer URL wrong | Typo in `dadfit.in` | Fix the text node |
