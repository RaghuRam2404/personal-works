# SKILL: Trim Carousel Slides

## Purpose

Read a `carousel.html` file, count the slides, and trim down to **10 or fewer** by removing slides according to the priority order defined in the DadFit slide-writer rules. Never touch the Cover (A-type) or CTA (H-type). Always update every counter and the subtitle after removal.

> ⛔ **TERMINAL RULE — READ BEFORE ANYTHING ELSE:**
> Multi-line patterns are **forbidden in any `run_in_terminal` call**. The following are illegal everywhere — zero exceptions:
> - `python3 -c "..."`
> - Any heredoc (`<<`, `<<'EOF'`, `<<-EOF`, etc.)
> - Any shell command containing a newline or line continuation
>
> All edits are made with `replace_string_in_file` or `multi_replace_string_in_file`. If a Python helper is needed write it with `create_file` first, then run it. Violating this rule means the step has failed.

---

## Inputs

- **Carousel HTML path** — absolute path to the `carousel.html` to trim.

---

## Step 1 — Read the file

Read the **full** `carousel.html` in one `read_file` call. Do not estimate line numbers before reading.

---

## Step 2 — Count and catalogue slides

Parse the `<h2>` inside each `<div class="section-header">`. Every such block is one slide. Extract:

| Field | Source in HTML |
|---|---|
| Slide number | `Slide N` in `<h2>` text |
| Slide type | The type code after `—` (e.g., `C1`, `B4`, `G1`) |
| Label | The descriptive label after the second `—` |

Also read the `<p class="page-subtitle">` for the current declared slide count.

Build a numbered list:

```
1. A1  — Cover
2. C1  — Pain
3. B1  — Skinny-Fat Defined
...
N. H1  — CTA
```

If total slides ≤ 10 → report "Already within limit" and stop.

---

## Step 3 — Build the removal plan

Remove slides in this **strict priority order** until total ≤ 10. Stop removing as soon as the count hits 10 — do not over-trim.

### Removal priority (highest → lowest)

| Priority | Types | Rule |
|---|---|---|
| 1 | **G** (G1, G2, G3, G4) | Recap/summary slides. Always the first to cut. Remove all G-type slides before touching anything else. |
| 2 | **F** (F2, F3, F4) | Pattern interrupt slides. Remove the second F-type if more than one exists; remove the only F if count is still > 10. |
| 3 | **E2** | Section-divider cards. Remove E2 slides if the carousel no longer has a true multi-section structure after earlier removals, or if removing them doesn't break story flow. |
| 4 | **E1 / E3** | Bridge or quote slides. Remove the weakest bridge (one that restate what the adjacent slide already says). Never remove a bridge that is the only transition from problem to solution. |
| 5 | **B1 / B4 / B5 / B6** | Content slides. Remove the one with the least unique information — e.g., a B4 whose callout restates the same point as an adjacent B1. Only remove one content slide per pass; re-evaluate after each removal. |
| **Never remove** | **A** (any) | Cover — always slide 1. |
| **Never remove** | **H** (any) | CTA — always last slide. |
| **Never remove** | **C1** (first occurrence) | The first/only pain/empathy frame. Removing it kills the story arc. |
| **Never remove** | **D1 / D3** | Real stat slides with a unique number — the data is rare and credible. |

If after applying all priorities the count is still > 10 (edge case: e.g., 12-slide carousel with no G/F/E slides), remove the **least informative B1/B4** — the one whose headline/body most closely mirrors an adjacent slide — and annotate the decision in your report.

### Output of this step

Write a numbered removal plan before touching the file:

```
REMOVAL PLAN (11 → 10 slides):
  Remove slide 10 — G1 — Quick Recap  [reason: G-type, priority 1]
```

---

## Step 4 — Apply removals

For each slide to remove:

1. Locate the full `<div class="section">…</div>` block for that slide. It starts with the matching `<div class="section">` and ends at the `</div>` that closes it, just before the next `<div class="section">` or `</body>`.
2. Remove the entire block using `replace_string_in_file`. Include at least 3 lines of context before the opening `<div class="section">` and after its closing `</div>`.
3. Do **not** remove the blank line between sections if one exists — keep the file tidy.

Remove slides one at a time. Re-read the file between removals if more than one slide needs to go.

---

## Step 5 — Update all counters

After all removals, update every counter reference in the file:

### 5a — `s-counter` spans

Every remaining slide (except the A-type cover and H-type CTA which typically omit counters) has:

```html
<span class="s-counter">07 / 11</span>
```

The new total is `N` (the final slide count after trimming). Renumber **every** `s-counter` span sequentially. Cover is `01 / N`, second slide is `02 / N`, etc.

Use `multi_replace_string_in_file` to update all counter spans in a single call when the old and new values are distinct per slide.

### 5b — Page subtitle

Find the `<p class="page-subtitle">` line, e.g.:

```html
<p class="page-subtitle">11 slides &middot; DadFit &middot; Batch 1</p>
```

Replace the slide count number only — do not alter batch info.

### 5c — Section header `<h2>` slide numbers

Each section header reads `Slide N — TYPE — Label`. Renumber `N` for every section that shifted. Use `multi_replace_string_in_file` for all renumbering in one call.

---

## Step 6 — Verify

After all edits, re-read the relevant sections of the file and confirm:

- [ ] Total `<div class="section">` blocks = 10 or fewer.
- [ ] First slide is A-type (Cover). Last slide is H-type (CTA).
- [ ] All `s-counter` values are sequential and the denominator matches the new total.
- [ ] `<p class="page-subtitle">` shows the new slide count.
- [ ] Section header `<h2>` numbers are sequential (Slide 1 … Slide N).
- [ ] No unclosed `<div>` tags or orphaned HTML.

---

## Step 7 — Report

Respond with a brief summary:

```
TRIMMED: 11 → 10 slides

Removed:
  • Slide 10 — G1 — Quick Recap  [priority 1: recap]

Counters updated: all s-counter spans, subtitle, section headers.
```

If no trimming was needed: `Already at 10 slides or fewer — no changes made.`
