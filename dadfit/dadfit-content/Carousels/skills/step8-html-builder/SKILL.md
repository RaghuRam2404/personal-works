# SKILL: Step 8 — HTML Builder (Orchestrator)

## Purpose

Orchestrate HTML generation for all 100 `CAPTION_WRITTEN` carousels. Each round spawns **5 subagents in parallel** (one per carousel) that use `step8-slide-writer/SKILL.md` to produce content JSON; a renderer script converts that JSON to HTML using snippet templates.

> **CLI RULE:** All pipeline commands must be run using `run_in_terminal` with `mode=sync` (foreground). Never use VS Code tasks or background execution. Always set `cwd` to `/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content` and wait for the command to complete before reading its output.
> **STRICTLY NO MULTILINE COMMANDS.** Every command sent to the terminal must be a single line. No `\n` inside strings, no here-docs, no multiline `python3 -c`. If a Python snippet is needed, write it to a `.py` file using `create_file`, then run `python3 /path/to/file.py`.

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

> **This skill is recursive.** Each invocation handles exactly 1 round of 5 carousels, spawns 5 parallel subagents, then re-invokes itself for the next round. This loops until all carousels are done.
>
> **Every invocation must start by reading `Carousels/skills/step8-html-builder/SKILL.md` in full before doing anything else.**

---

### Step A — Check state

The state checker is at `Carousels/scripts/step8_stepa.py`. Do **not** re-create it.

**Before round 1 only** — create files if missing. Run:

```
[ -f Carousels/data/batch_1/html_checkpoint.json ] || echo '[]' > Carousels/data/batch_1/html_checkpoint.json && [ -f Carousels/data/batch_1/doodle_prompts.json ] || echo '[]' > Carousels/data/batch_1/doodle_prompts.json && mkdir -p Carousels/data/batch_1/doodles && echo 'Init done.'
```

Then run the state checker requesting 5 carousels for this round:

```
python3 Carousels/scripts/step8_stepa.py --batch 1 --count 5
```

Read the task output:
- `ALL DONE` → skip to Step E
- `STATUS: PROCESS running_no X, Y, Z, …` → proceed to Step B with those carousels

The script writes up to 5 carousel objects to `/tmp/batch_1_html_round.json`.

---

### Step B — Spawn 5 parallel subagents

Read `/tmp/batch_1_html_round.json` — it contains an array of up to 5 carousel objects. Spawn **all of them in parallel** using `runSubagent` in a single `<function_calls>` block. Do not wait for one to finish before spawning the next.

Use this prompt for every subagent (fill in all `{placeholders}` from each carousel's data):

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

### Step C — Render HTML, validate, and checkpoint

**Before the first carousel render in this round**, clear the results file:

```
echo '' > /tmp/html_round_results.json && echo 'Reset done.'
```

Run the renderer for each of the up-to-5 carousels **one at a time** (the renderer appends one result JSON line per run):

```
python3 Carousels/scripts/step8_renderer.py --input /tmp/carousel_{uuid}.json --batch 1 >> /tmp/html_round_results.json
```

(Repeat this command for each UUID from Step B.)

Wrap the output lines into a JSON array:

```
python3 -c "import json; lines=open('/tmp/html_round_results.json').read().strip().splitlines(); results=[json.loads(l) for l in lines if l.strip()]; json.dump(results,open('/tmp/html_round_results.json','w'),indent=2); print(f'Wrapped {len(results)} result(s).')"
```

> If this single-line command fails for any reason, use `create_file` to write the logic to `/tmp/wrap_results.py` (as a normal multi-line Python file), then run `python3 /tmp/wrap_results.py`.

Validate the rendered HTMLs and checkpoint:

```
python3 Carousels/scripts/step8_validate.py --batch 1
```

The validator checks every rendered HTML and, if it passes, updates `html_checkpoint.json` and `doodle_prompts.json`.

**For any failed entry**: re-spawn a fresh subagent for that carousel only. Re-run the renderer command for it, then re-run the wrap and validate commands.

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
2. Run `python3 Carousels/scripts/step8_stepa.py --batch 1 --count 5` to get the next round of 5
3. Process as a fresh round with a clean context

Repeat until Step A reports `ALL DONE`.

**Terminal condition**: When Step A reports `ALL DONE`, proceed to Step E.

---

### Step E — Insert into DB

The insert script is permanently stored at `Carousels/scripts/step8_insert.py`. Do **not** re-create it — just run:

```
python3 Carousels/scripts/step8_insert.py --batch 1
```

---

### Step F — Verify

```
python3 Carousels/scripts/orchestrator.py status --batch 1
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
| Wrap step fails for any reason | Use `create_file` to write the wrap logic to `/tmp/wrap_results.py`, then run `python3 /tmp/wrap_results.py` — never use a multiline `python3 -c` |

## Purpose

Orchestrate HTML generation for all 100 `CAPTION_WRITTEN` carousels. Spawns subagents (one per carousel) that use `step8-slide-writer/SKILL.md` to produce content JSON; a renderer script converts that JSON to HTML using snippet templates.

> **CLI RULE:** All pipeline commands must be run using `run_in_terminal` with `mode=sync` (foreground). Never use VS Code tasks or background execution. Always set `cwd` to `/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content` and wait for the command to complete before reading its output.
> **STRICTLY NO MULTILINE COMMANDS.** Every command sent to the terminal must be a single line. No `\n` inside strings, no here-docs, no multiline `python3 -c`. If a Python snippet is needed, write it to a `.py` file using `create_file`, then run `python3 /path/to/file.py`.

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

> **This skill is recursive.** Each invocation handles exactly 1 carousel, spawns 1 subagent, then re-invokes itself for the next carousel. This loops until all 100 carousels are done.
>
> **Every invocation must start by reading `Carousels/skills/step8-html-builder/SKILL.md` in full before doing anything else.**

---

### Step A — Check state

The state checker is at `Carousels/scripts/step8_stepa.py`. Do **not** re-create it.

**Before round 1 only** — create files if missing. Run:

```
[ -f Carousels/data/batch_1/html_checkpoint.json ] || echo '[]' > Carousels/data/batch_1/html_checkpoint.json && [ -f Carousels/data/batch_1/doodle_prompts.json ] || echo '[]' > Carousels/data/batch_1/doodle_prompts.json && mkdir -p Carousels/data/batch_1/doodles && echo 'Init done.'
```

Then run the state checker:

```
python3 Carousels/scripts/step8_stepa.py --batch 1
```

Read the task output:
- `ALL DONE` → skip to Step E
- `PROCESS running_no X` → proceed to Step B with that carousel

---

### Step B — Spawn 1 subagent

Read `/tmp/batch_1_html_round.json` to get the 1 carousel for this round. Spawn **1 subagent** for it — do not spawn more than one at a time.

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

**Before the first carousel render**, clear the results file:

```
echo '' > /tmp/html_round_results.json && echo 'Reset done.'
```

Run the renderer for the 1 carousel (the renderer appends one result JSON line):

```
python3 Carousels/scripts/step8_renderer.py --input /tmp/carousel_{uuid}.json --batch 1 >> /tmp/html_round_results.json
```

(Replace `{uuid}` with the actual UUID from Step B.)

Wrap the output lines into a JSON array:

```
python3 -c "import json; lines=open('/tmp/html_round_results.json').read().strip().splitlines(); results=[json.loads(l) for l in lines if l.strip()]; json.dump(results,open('/tmp/html_round_results.json','w'),indent=2); print(f'Wrapped {len(results)} result(s).')"
```

> If this single-line command fails for any reason, use `create_file` to write the logic to `/tmp/wrap_results.py` (as a normal multi-line Python file), then run `python3 /tmp/wrap_results.py`.

Validate the rendered HTML and checkpoint:

```
python3 Carousels/scripts/step8_validate.py --batch 1
```

The validator checks the rendered HTML and, if it passes, updates `html_checkpoint.json` and `doodle_prompts.json`.

**For any failed entry**: re-spawn a fresh subagent for that carousel only. Re-run the renderer command for it, then re-run the wrap and validate commands.

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
2. Run `python3 Carousels/scripts/step8_stepa.py --batch 1` to get the next round
3. Process as a fresh round with a clean context

Repeat until Step A reports `ALL DONE`.

**Terminal condition**: When Step A reports `ALL DONE`, proceed to Step E.

---

### Step E — Insert into DB

The insert script is permanently stored at `Carousels/scripts/step8_insert.py`. Do **not** re-create it — just run:

```
python3 Carousels/scripts/step8_insert.py --batch 1
```

---

### Step F — Verify

```
python3 Carousels/scripts/orchestrator.py status --batch 1
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
| Wrap step fails for any reason | Use `create_file` to write the wrap logic to `/tmp/wrap_results.py`, then run `python3 /tmp/wrap_results.py` — never use a multiline `python3 -c` |

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
