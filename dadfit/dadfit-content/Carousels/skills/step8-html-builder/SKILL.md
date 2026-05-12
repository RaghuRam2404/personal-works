# SKILL: Step 8 — HTML Builder (Orchestrator)

## Purpose

Orchestrate HTML generation for all 100 `CAPTION_WRITTEN` carousels. Spawns subagents (one per carousel) that use `step8-slide-writer/SKILL.md` to produce content JSON; a renderer script converts that JSON to HTML using snippet templates.

> **TERMINAL RULE:** Never use `run_in_terminal` with `mode=async` or `isBackground=true`. All terminal commands must run in the foreground (`mode=sync`).

---

## Output Spec

For each carousel (identified by `uuid` and `running_no`):

| Item | Path |
|------|------|
| Output folder | `Carousels/data/batch_{batch_no}/{running_no}_{uuid}/` |
| Carousel HTML | `Carousels/data/batch_{batch_no}/{running_no}_{uuid}/carousel.html` |
| Doodle prompts (batch-level) | `Carousels/data/batch_{batch_no}/doodle_prompts.json` |
| Shared doodle images | `Carousels/data/batch_{batch_no}/doodles/{running_no}-d-01.png` … |

DB columns set after all 100 are done:
- `folder_name = '{running_no}_{uuid}'`
- `current_stage = 'HTML_CREATED'`

---

## Instructions

> **This skill is recursive.** Each invocation handles exactly 10 carousels, then re-invokes itself for the next 10. Never process more than 10 carousels per invocation.
>
> **Every invocation must start by reading `Carousels/skills/step8-html-builder/SKILL.md` in full before doing anything else.**

---

### Step A — Check state

The state checker is at `Carousels/scripts/step8_stepa.py`. Do **not** re-create it.

**Before round 1 only** — create files if missing:
```bash
echo "[]" > Carousels/data/batch_1/html_checkpoint.json
echo "[]" > Carousels/data/batch_1/doodle_prompts.json
mkdir -p Carousels/data/batch_1/doodles
```

```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 Carousels/scripts/step8_stepa.py --batch 1
```

- `ALL DONE` → skip to Step E
- `PROCESS running_no X to Y` → proceed to Step B with those carousels

---

### Step B — Spawn 10 subagents simultaneously

Read `/tmp/batch_1_html_round.json` to get the 10 carousels for this round. Spawn all 10 **simultaneously** — one per carousel, no exceptions.

Use this prompt for every subagent (fill in all `{placeholders}` from the carousel data):

---

> **Read `Carousels/skills/step8-slide-writer/SKILL.md` in full as your FIRST action. Follow it exactly.**
>
> Carousel data:
> - UUID: `{uuid}`
> - Running No: `{running_no}`
> - Batch: `1`
> - Title: `{title}`
> - Keyword: `{keyword}`
> - Category: `{category}` (TOFU = educate cold audience | MOFU = deepen trust | BOFU = convert)
> - Hook: `{hook}`
> - CTA: `{cta}`
> - Script:
> ```
> {script_content}
> ```

---

### Step C — Render HTML, then validate and checkpoint

For each of the 10 carousels, run the renderer which reads the subagent content JSON and generates HTML from snippet templates:

```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && \
  python3 Carousels/scripts/step8_renderer.py --input /tmp/carousel_{uuid}.json --batch 1 \
  >> /tmp/html_round_results.json
```

Run this for all 10 carousels. The renderer appends one result JSON line per run. Then wrap into an array and validate:

```bash
# Wrap renderer output lines into a JSON array
python3 -c "
import sys, json
lines = open('/tmp/html_round_results.json').read().strip().splitlines()
results = [json.loads(l) for l in lines if l.strip()]
json.dump(results, open('/tmp/html_round_results.json','w'), indent=2)
"
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 Carousels/scripts/step8_validate.py --batch 1
```

The validator checks all 10 rendered HTMLs and, if all pass, updates `html_checkpoint.json` and `doodle_prompts.json`.

**Before the first carousel render**, reset the results file:
```bash
> /tmp/html_round_results.json
```

**For any failed entry**: re-spawn a fresh subagent for that carousel only. Re-run the renderer for it, re-validate.

**Common failure modes and fixes:**

| Failure | Fix |
|---------|-----|
| `/tmp/carousel_{uuid}.json` not found | Subagent failed to write — re-spawn |
| `snippet not found` in stderr | Wrong type code used — check SKILL.md vars reference, re-spawn |
| `MISSING: VAR` warnings in stderr | Subagent omitted a required var — re-spawn with complete vars |
| HTML < 10 KB | `slides` array was empty/truncated — re-spawn |
| `../doodles/` not in HTML | Wrong `DOODLE_SRC` format in vars — re-spawn |
| `doodle_prompts` missing | Subagent omitted from JSON — re-spawn |

---

### Step D — Re-invoke this skill (recursive call)

After checkpointing, **re-invoke this skill**. The new invocation must:
1. **Read `Carousels/skills/step8-html-builder/SKILL.md` in full** — first action, no exceptions
2. Run `Carousels/scripts/step8_stepa.py --batch 1` to get the next round
3. Process as a fresh round with a clean context

Repeat until Step A reports `ALL DONE`.

**Terminal condition**: When Step A reports `ALL DONE`, proceed to Step E.

---

### Step E — Insert into DB

The insert script is permanently stored at `Carousels/scripts/step8_insert.py`. Do **not** re-create it — just run it:

```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 Carousels/scripts/step8_insert.py --batch 1
```

---

### Step F — Verify

```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 Carousels/scripts/orchestrator.py status --batch 1
```

Confirm `HTML_CREATED = 100`.

---

### Step E — Insert into DB

The insert script is permanently stored at `Carousels/scripts/step8_insert.py`. Do **not** re-create it — just run it:

```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 Carousels/scripts/step8_insert.py --batch 1
```

---

### Step F — Verify

```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 Carousels/scripts/orchestrator.py status --batch 1
```

Confirm `HTML_CREATED = 100`.

---

## Path Reference (Quick Look-Up)

| Item | Relative path from `carousel.html` |
|------|-------------------------------------|
| Logo | `../../../../Resources/Images/logo.png` |
| Doodle slide 1 of carousel #7 | `../doodles/7-d-01.png` |
| Doodle slide 5 of carousel #42 | `../doodles/42-d-05.png` |

The output file lives at:
```
Carousels/data/batch_1/{running_no}_{uuid}/carousel.html
```
To reach the project root: go up 4 levels → `../../../../`
To reach `doodles/`: go up one level to `batch_1/`, then into `doodles/` → `../doodles/`

**Doodle image naming:** `{running_no}-d-{slide_no:02d}.png` — unique across the batch since all images share one folder.

---

## Slide Type Quick Reference

All 36 types from the template. Search for the slide comment tag in the template to get the exact line, then `read_file` that range. **Never write slide markup from memory.**

### TYPE A — Cover (Slide 1 only)

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **A1** | ~318 | `s-bg` + doodle + counter top-left + logo top-right + big headline (Inter ExtraBold, white) + 1-line sub-text (ADADAD) + `→` footer | **USE** — default cover. Bold declarative hook or urgent promise. |
| **A2** | ~361 | `s-bg` + counter top-left + logo top-right + left headline + right framed photo (`photo-ph`) + `→` footer | **SKIP** — requires real photo. Replace with A1. |
| **A3** | ~476 | Full-bleed photo + dark overlay + green gradient bottom + counter + logo + bottom headline + `→` footer | **SKIP** — requires real photo. Replace with A1. |
| **A4** | ~402 | `s-bg` + doodle + counter top-left + logo top-right + Caveat opener line (white, 42px) + big Inter headline (white) + `→` footer | **USE** — personal/reflective hook. Sounds like a coach talking directly to the dad. |
| **A5** | ~438 | `s-bg` + doodle + counter top-left + logo top-right + Permanent Marker challenge word (#34C363) + big Inter headline + `→` footer | **USE** — dare / protocol / challenge hook. Use when carousel is a 30-day challenge. |

### TYPE B — Content / Explainer

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **B1** | ~527 | `s-bg` + doodle + counter + headline (Inter ExtraBold, 68px, white) + 2-line body (Inter Regular, 33px, white) + optional green bar + Caveat footer + `→` | **USE** — workhorse. Single tip, principle, or insight. ~60% of content slides. |
| **B2** | ~568 | `s-bg` + counter + left headline + body + right framed `photo-ph` (340×490px) + `→` | **SKIP** — requires real photo. Replace with B1. |
| **B3** | ~707 | Full-bleed photo + overlays + counter + bottom headline + body + `→` | **SKIP** — requires real photo. Replace with B1, B4, or E1. |
| **B4** | ~737 | `s-bg` + doodle + counter + headline (68px) + body (31px, 2 lines) + `callout` box (✦ + one action sentence) + `→` | **USE** — tip with a concrete boxed action. Callout **replaces** body — never use both. |
| **B5** | ~605 | `s-bg` + doodle + counter + headline (60px) + two-column grid: WRONG card (red border) + RIGHT card (green border) + Caveat footer + `→` | **USE** — explicit wrong vs right contrast. Only when script directly compares two behaviours. |
| **B6** | ~653 | `s-bg` + doodle + counter + headline (62px) + 3 step rows (64px green circle with digit + step text) + green bar + Caveat footer + `→` | **USE** — 3-step ordered protocol. Only when script is literally "step 1, step 2, step 3". |

### TYPE C — Problem / Pain

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **C1** | ~790 | `s-bg` + doodle + counter + Permanent Marker pain label (#FF6B6B, 52px) + Inter Bold pain statement (white, 46px, 2 lines) + Inter Regular reframe (ADADAD, 30px) + `→` | **USE** — name the dad's exact frustration. Use once near the start to establish empathy before the solution. |
| **C2** | ~904 | `s-bg` + counter + Permanent Marker label (#FF6B6B) + left headline (red accent) + right `photo-ph` (red-tint border) + body + `→` | **SKIP** — requires real photo. Replace with C1. |
| **C3** | ~832 | `s-bg` + doodle + counter + MYTH card (red top border, Permanent Marker label + 2-line myth text) + TRUTH card (green top border, label + truth text) + `→` | **USE** — busting a specific named misconception. Only when script explicitly contradicts a common belief. |
| **C4** | ~868 | `s-bg` + doodle + counter + "BE REAL." (Permanent Marker, #FF6B6B) + 3-item checklist (checkmark + excuse text, ADADAD) + Caveat footer + `→` | **USE** — listing excuses the dad makes by name. Only when script calls out avoidance behaviours. |

### TYPE D — Proof / Stat

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **D1** | ~956 | `s-bg` + doodle + counter + Permanent Marker label (#34C363) + giant number (Inter ExtraBold, #34C363, 200–230px) + 1-line context (white, 28px) + Caveat quote (ADADAD) + `→` | **USE** — one real stat from the script. Number must exist in script — never invent data. |
| **D2** | ~1075 | Full-bleed photo + heavy overlay + counter + Permanent Marker label + giant number overlaid + context text + `→` | **SKIP** — requires real photo. Replace with D1 or D3. |
| **D3** | ~995 | `s-bg` + doodle + counter + two-column split: left (red, negative stat) + right (green, positive stat) + 1-line footer + `→` | **USE** — two opposing numbers. Only when script has explicit contrasting figures (e.g. "68% quit vs 94% with a system"). |
| **D4** | ~1031 | `s-bg` + doodle + counter + headline + 3 progress bar rows (label + filled bar + milestone text) + `→` | **USE** — measurable timeline with specific checkpoints. Only when script describes a progression (Week 1 / Week 4 / Week 12). |

### TYPE E — Transition / Bridge

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **E1** | ~1117 | `s-bg` + doodle + counter + Caveat opener line (white, 44px) + Inter ExtraBold pivot statement (#34C363, 64px) + optional ADADAD sub-line + `→` | **USE** — pivot from problem to solution. 2–3 text elements only. No numbered section. |
| **E2** | ~1220 | `s-bg` + doodle + counter + centered `#292929` card (green top border + "PART 2" label + section title) + ADADAD description below + `→` | **USE** — section divider for multi-section carousels (2+ clearly distinct parts). |
| **E3** | ~1153 | `s-bg` + doodle + counter + green bar top + Caveat quote (white, 52px, italic-feel) + green bar bottom + ADADAD 1-line footer + `→` | **USE** — quotable one-liner that deserves its own moment. Philosophical statement or insight. |
| **E4** | ~1184 | `s-bg` + doodle + counter + pill nav row (Part 1 ✓ / Part 2 active / Part 3 inactive) + big headline + ADADAD description + `→` | **USE only** for carousels with 3+ explicitly named parts and 12+ slides. Otherwise use E2. |

### TYPE F — Pattern Interrupt / Breath (max 1 per carousel)

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **F1** | ~1272 | Full-bleed photo + minimal overlay. No counter, no logo, no text. | **SKIP** — requires real photo. |
| **F2** | ~1338 | `s-bg` + doodle + two massive ALL CAPS Inter lines (line 1 white, line 2 #34C363, 130px). No counter, no brand text. | **USE** — visual pause in long carousels (9+ slides). Words must be a punchy truth ("SLOW IS / SUSTAINABLE."). |
| **F3** | ~1287 | `s-bg` + doodle + two giant Caveat lines (white + #34C363, 170px). No counter, no brand text. | **USE** — breather for motivational carousels. One word-pair truth ("just / start."). |
| **F4** | ~1310 | `s-bg` + doodle + giant faint number background (380px, 7% opacity) + two green horizontal rules + "TIP N OF N" Inter ExtraBold (88px). | **USE** — numbered tip progress marker. Good for listicle carousels where showing progress aids retention. |

### TYPE G — Recap / Summary (second-to-last slide)

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **G1** | ~1379 | `s-bg` + doodle + counter (←) + "QUICK RECAP" (Permanent Marker, #34C363, 60px) + 3–5 numbered bullet rows (plain digits in green span + white text, 36px) + Caveat "Save this." footer | **USE** — default recap. 3+ distinct tips worth summarising. |
| **G2** | ~1518 | `s-bg` + doodle + counter (←) + "QUICK RECAP" + 2×3 grid of emoji icon cards (green top border, `#292929` bg, emoji + 2-word label) | **USE only** when carousel has exactly 6 clean takeaways that map naturally to a single emoji each. |
| **G3** | ~1417 | `s-bg` + doodle + counter (←) + "REMEMBER THIS —" (Permanent Marker, #34C363) + one giant Inter ExtraBold statement (100px, white) + green bar + Caveat "Save it. Share it." | **USE** — when the whole carousel's lesson distills to one unforgettable line. Use instead of G1. |
| **G4** | ~1449 | `s-bg` + doodle + counter (←) + "YOUR SCORECARD" (Permanent Marker) + 3 habit rows (`#292929` card + habit label + 7 day-dot squares filled/empty) + Caveat footer | **USE only** for habit/routine carousels where the output is a weekly habit tracker. |

### TYPE H — CTA (always last slide, no counter)

| Type | Line | Layout | Use / Skip |
|------|------|--------|-----------|
| **H1** | ~1591 | `s-bg` + doodle + centered logo (80px, `mix-blend-mode:screen`) + green bar + CTA text (Inter Bold, 44px, white, 2 lines) + green bar + ADADAD follow line + Permanent Marker `@dadfit.in` (52px, #34C363) | **USE** — default CTA for all carousels. |
| **H2** | ~1694 | `s-bg` + circular founder photo (200px) + logo + name + credentials + green divider + Caveat personal message + Inter `DM "START"` CTA | **SKIP** — requires founder photo. |
| **H3** | ~1631 | `s-bg` + doodle + radial green glow bg + "D A D F I T" (spaced, ADADAD) + big white headline + green pill button ("DM 'DAD'", Permanent Marker) + ADADAD `dadfit.in` | **USE** — only when CTA explicitly asks users to DM a keyword. |
| **H4** | ~1659 | `s-bg` + doodle + `#292929` testimonial card (green top border + Caveat quote + member attribution) + green bar + follow prompt + Permanent Marker `@dadfit.in` | **USE only** when you have a real named testimonial from the script. Never fabricate. |

> **Finding slide HTML**: Search the template file for the `slide-label` text (e.g. `B1 — Content`) using grep, get the exact line, then `read_file` that range. **Never write slide markup from memory** — always copy verbatim.

---

## Design Token Cheat-Sheet (from design.MD)

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

## Pitfalls & Solutions

| Pitfall | Solution |
|---------|---------|
| Logo path `../../Brand assets/logo.png` copied from template | Rewrite to `../../../../Resources/Images/logo.png` always |
| CSS block truncated | Must be copied verbatim — no truncation |
| Script block omitted | Must be the last thing before `</body>` |
| Doodle `src` uses `images/d-01.png` or `src=""` | Must be `../doodles/{running_no}-d-{slide_no:02d}.png` — shared batch folder, named by carousel |
| Slide has headline + long body + callout all at once | Remove body OR callout — never both. Callout replaces, not adds. |
| Body copy is 4+ lines | Cut to 2–3 lines. If the idea needs more, split to a second slide. |
| Font reduced below 30px to fit copy | Wrong direction — reduce the copy, not the font. |
| Slide type chosen for convenience not fit | Re-read the type selection table. C1 is pain, not just any problem. D1 needs a real number. |
| More than 11 slides | Merge bridge ideas, cut recap if not needed |
| Fewer than 7 slides | Script has enough ideas for 8–10 slides — don't merge excessively |
| Subagent returns text instead of JSON | Re-spawn with explicit instruction to respond ONLY with JSON |
| `python3 -c "..."` fails with unicode | Always use `create_file` + `python3 /tmp/file.py` |

---

## Success Criteria

- All 100 carousels have `carousel.html` in their folder
- Every HTML is >10 KB (full CSS + script included)
- Logo path is `../../../../Resources/Images/logo.png` in all files
- No `Brand assets` path in any output file
- `buildSVG` scribble generator present in all HTML files
- All doodle `src` values use `../doodles/{running_no}-d-{slide_no:02d}.png` format
- `Carousels/data/batch_1/doodle_prompts.json` exists with all doodle prompt entries for all 100 carousels
- All 100 DB rows have `folder_name` set and `current_stage = HTML_CREATED`
- Orchestrator confirms `HTML_CREATED = 100`
