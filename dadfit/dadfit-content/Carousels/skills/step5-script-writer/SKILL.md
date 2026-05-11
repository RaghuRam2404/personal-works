# SKILL: Step 5 — Script Writer

## UUID Rule
Never type or fabricate UUIDs. Query the DB. See `Carousels/skills/generate-uuid/SKILL.md`.

## Purpose
Write the **slide-by-slide script** for each of the 100 `HOOK_WRITTEN` carousels. Each script is the full spoken/visual content of the carousel — one sentence per slide. The script must weave SPCL (Status → Power → Credibility → Likeness), follow the correct content-type flow from `Resources/Content based flows.md`, and stay grounded in the DadFit ideology from `Resources/DadFit Ideology.md`.

## Required Reading (before spawning any subagent)
The orchestrating agent must read these resources before building subagent prompts:
- `Resources/SPCL.md` — influence framework to weave into every script
- `Resources/Content based flows.md` — structural templates by content type
- `Resources/DadFit Ideology.md` — DadFit's core concepts, principles, and vocabulary for diet, workout, and lifestyle. Each subagent will read this file directly; the orchestrator should be familiar with it to verify outputs.

## Output
- All 100 rows updated: `script_content` set, `current_stage = SCRIPT_WRITTEN`
- `scripts.json` saved to `Carousels/data/batch_{batch_no}/scripts.json` for audit

---

## What the Script Is

Each carousel has **8–10 slides** (worst case 12). Each slide carries exactly **one sentence** — short, punchy, conversational. These are not bullet points. They are spoken lines, like a person talking directly to the viewer.

The script is stored as a single `\n`-delimited string in `script_content` (one line per slide).

**Slide structure:**
1. Hook slide — restate or reinforce the hook from the cover (1 sentence)
2. Body slides — 6–8 slides that deliver the content
3. CTA slide — a soft close that leads into the CTA (written in Step 6)

The CTA slide text is NOT the CTA itself. It's the bridge sentence that makes the viewer ready for the CTA (e.g. "Here's exactly how to start this today.").

---

## SPCL Weaving (Read `Resources/SPCL.md` before writing)

Every script must contain at least 3 of the 4 SPCL elements, woven naturally into the body slides:

| Element | What it looks like in a carousel script |
|---------|------------------------------------------|
| **Status** | Reference to a scarce resource the audience wants (energy, a lean body, time, financial security) — frame DadFit as controlling access to it |
| **Power** | Give a directive: "Do X" → imply the good outcome follows. Say-do correspondence. The script itself is the power move — you told them what to do |
| **Credibility** | A specific number, a named result, a named protocol, or a reference to verified outcomes ("clients who did this lost X in Y weeks") |
| **Likeness** | Language that mirrors the audience's life: desk job, 9-to-5, evening exhaustion, chai break, family dinners, Indian diet, budget constraints |

SPCL does not need to appear in order. Embed the elements naturally across the body slides.

---

## DadFit Ideology (Read `Resources/DadFit Ideology.md` before writing)

Every script must stay within DadFit's ideological boundaries. Before spawning subagents, read `Resources/DadFit Ideology.md` and extract the sections relevant to the carousel's topic (diet, workout, or lifestyle). Pass those excerpts into the subagent prompt.

**Hard rules derived from DadFit ideology:**
- Never recommend advice that contradicts DadFit's stance on diet, training, or lifestyle
- Use DadFit's vocabulary and framing — not generic fitness influencer language
- If the carousel topic is about diet, the script must reflect DadFit's position on Indian food, protein targets, and sustainable eating — not keto/paleo/Western frameworks unless the carousel explicitly compares them
- If the topic is about training, reflect DadFit's minimum-effective-dose philosophy — not bro-split or high-volume-for-its-own-sake language
- If the topic is lifestyle/financial, reflect DadFit's grounded, family-first, salaried-income framing

---

## Content-Type Flows (Read `Resources/Content based flows.md` before writing)

Determine the content type for each carousel from its `title`, `keyword`, and `category`, then apply the matching flow:

### EDUCATIONAL (most TOFU and MOFU)
Pick one of:
- **Myth-Busting**: Open with the wrong belief → reveal the truth → explain the mechanism → close with the fix
- **Step-by-Step**: Open with the problem → deliver 3–5 numbered steps → close with what happens when they follow it
- **Problem-Agitation-Solution**: Name the pain → show why it gets worse → deliver the fix
- **Do's and Don'ts**: Show the wrong approach → show the right approach → explain why → close with action

### STORYTELLING (select MOFU, any topic with a journey arc)
Use the **Win/Loss/Lesson** arc:
- Set up the situation → show what was tried → reveal the unexpected outcome → extract the lesson

Do NOT write first-person "I" or "My" story arcs — content is AI-generated, not personal. Reframe as "most dads do X" or "here's what changed everything for desk workers".

### AUTHORITY / PROTOCOL (most BOFU)
Use the **Before → After → Process → CTA bridge** structure:
- State the gap (before) → state the result (after) → deliver the protocol as numbered steps → bridge to the CTA

---

## Pitfalls & Solutions (Learned from First Full Run)

| Pitfall | Root cause | Solution |
|---------|-----------|----------|
| `python3 -c "..."` fails when scripts contain `—` em-dashes | Unicode breaks shell inline scripts | Always write save/validate scripts to `/tmp/save_round{N}.py` using `create_file` tool, then run with `python3`. Never paste scripts with em-dashes into an inline `-c` command. |
| Checkpoint JSON has escaped unicode (`\u2014`) instead of literal `—` | `json.dump` defaults to `ensure_ascii=True` | Always pass `ensure_ascii=False` to every `json.dump` call in this workflow |
| Subagents produce generic scripts | Audience line in prompt is boilerplate | Replace the audience line with a specific description derived from each carousel's `trait` and `title` before spawning the subagent |
| Validation logic re-invented every round | SKILL.md describes validation but gives no code | Use the canonical validation block in Step C — never write validation from scratch |

---

## Instructions

> **This skill is recursive.** Each invocation of this skill handles exactly one round of 10 carousels, then re-invokes itself for the next round. A fresh invocation = a fresh context = no drift. Never try to process more than 10 carousels in a single invocation of this skill.

> **Every invocation — including re-invocations — must start by reading `Carousels/skills/step5-script-writer/SKILL.md` in full before doing anything else.** Do not rely on memory from a previous invocation. Re-reading this file is the first action of every round, without exception. This ensures all rules, constraints, and the recursive structure are fresh in context before any subagent is spawned.

### Step A — Check state and decide what to do this invocation

Run this first. It tells you exactly what this invocation must do.

**First invocation only**: Use `create_file` to write `/tmp/step5_stepa.py` with the content below. On all subsequent invocations (rounds 2–10), the file already exists — just run it directly.

```python
# /tmp/step5_stepa.py
import sqlite3, json, os

BATCH_NO = 1  # change per batch

checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/scripts_checkpoint.json'

conn = sqlite3.connect('Carousels/data/db.sqlite')
all_rows = conn.execute(
    'SELECT uuid, running_no, title, keyword, trait, category, hook '
    'FROM Carousel WHERE batch_no = ? AND current_stage = "HOOK_WRITTEN" '
    'ORDER BY running_no',
    (BATCH_NO,)
).fetchall()
conn.close()

all_carousels = [{
    'uuid': r[0], 'running_no': r[1], 'title': r[2],
    'keyword': r[3], 'trait': r[4], 'category': r[5], 'hook': r[6]
} for r in all_rows]

done_uuids = set()
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, encoding='utf-8') as f:
        done_uuids = {e['uuid'] for e in json.load(f)}

remaining = [c for c in all_carousels if c['uuid'] not in done_uuids]
this_round = remaining[:10]

print(f'Done: {len(done_uuids)}/100  |  Remaining: {len(remaining)}  |  This round: {len(this_round)}')

if not this_round:
    print('STATUS: ALL DONE — skip to Step D')
else:
    print(f'STATUS: PROCESS running_no {this_round[0]["running_no"]} to {this_round[-1]["running_no"]}')
    for c in this_round:
        print(f'  # {c["running_no"]} [{c["category"]}] {c["title"]}')
    with open(f'/tmp/batch_{BATCH_NO}_this_round.json', 'w', encoding='utf-8') as f:
        json.dump(this_round, f, indent=2, ensure_ascii=False)
    print(f'\nWritten to /tmp/batch_{BATCH_NO}_this_round.json')
```

Then run it from the project root:
```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 /tmp/step5_stepa.py
```

**Read the STATUS line:**
- `ALL DONE` → skip to Step D
- `PROCESS running_no X to Y` → this invocation handles only those carousels. Proceed to Step B.

### Step B — Spawn 10 subagents for this round (one per carousel)

Take the 10 carousels from `/tmp/batch_BATCH_NO_this_round.json`. Spawn all 10 subagents simultaneously — one per carousel, no exceptions.

**STRICT RULE — ONE SUBAGENT, ONE CAROUSEL, ONE SCRIPT. NO EXCEPTIONS.**

This rule applies identically to every invocation of this skill, whether it is the 1st or the 10th:
- Each subagent call contains the data for **exactly one carousel** and must return **exactly one script**.
- The prompt must contain one `uuid`, one `title`, one `hook`. If you find yourself writing a prompt with two carousels in it, stop — split them.
- Do not ask a subagent to "handle the rest", "do the remaining ones", or "write scripts for the following". Each invocation is isolated.
- If a subagent returns a JSON array or multiple `uuid` keys: **reject it entirely**, re-run a fresh dedicated subagent for that carousel.

**Why the recursive structure enforces this**: Each skill invocation only ever sees 10 carousels — the next 10 are handled by the next invocation of the skill, with a clean context. There is no "later in the same session" where batching can creep in.

Use this prompt per subagent (one invocation per carousel):

---

> You are writing the slide-by-slide script for a **single** Instagram carousel for DadFit — a fitness, lifestyle, and financial brand for salaried Indian fathers aged 30–45.
>
> **This prompt is for exactly one carousel. Write exactly one script. Do not ask for more carousels. Do not produce output for any other carousel.**
>
> **Audience**: [Write a specific 1-sentence description derived from the carousel's `trait` field and `title`. Do NOT use the generic boilerplate. Example: "Salaried Indian father who skips breakfast every morning and reaches his desk already behind on energy."] This specificity makes the script resonate — generic audience descriptions produce generic scripts.
>
> **What the script is**: One sentence per slide. 8–10 slides total (worst case 12). Each sentence is a spoken, conversational line — like talking directly to the viewer. No bullet points. No headers. No markdown.
>
> **Slide structure**:
> 1. Reinforce the hook — restate or deepen the opening tension (1 sentence)
> 2. Body slides — 6–8 sentences that deliver the content
> 3. Bridge to CTA — 1 sentence that closes the content and sets up the action step
>
> **SPCL Rules** (weave at least 3 of 4 naturally):
> - Status: Frame the insight as a scarce resource the viewer is getting access to
> - Power: Give a clear directive — "do X" and let the implied outcome follow
> - Credibility: Use at least one specific number, named result, or verified protocol
> - Likeness: Mirror the audience's life — desk job, Indian diet, family, chai breaks, tight schedule
>
> **Content-type flow** (pick the right one for this carousel):
> - TOFU myth-bust or educational: Myth → Truth → Mechanism → Fix
> - MOFU step-by-step or system: Problem → Steps → Result
> - BOFU protocol/blueprint: Gap → Outcome → Protocol steps → Bridge
>
> **Rules**:
> - Never start any sentence with "I" or "My" — content is not first-person storytelling
> - Each sentence must be ≤20 words — short, punchy, slide-ready
> - No vague sentences like "This is important" or "Many people struggle" — be specific
> - Include at least one specific number (kg, minutes, rupees, weeks, reps, etc.) in the script
> - The last sentence must be a bridge to action, not a summary
>
> **Carousel to script**:
> - Title: {title}
> - Keyword: {keyword}
> - Trait: {trait}
> - Category: {category}
> - Hook: {hook}
>
> **DadFit Ideology** (apply this — do not contradict it):
> Read `Resources/DadFit Ideology.md` in full. Extract the sections relevant to this carousel's topic domain (diet, workout, or lifestyle). Your script must stay within those ideological boundaries — use DadFit's vocabulary, framing, and positions. Never use generic fitness advice that conflicts with what that file says.
>
> **Before responding, run this self-scrutiny checklist on your draft. Fix any failure before outputting.**
>
> | Check | Pass condition |
> |-------|----------------|
> | Sentence count | Between 8 and 12 sentences total |
> | Sentence length | Every sentence is ≤20 words |
> | No first-person | Zero sentences start with "I" or "My" |
> | Specific number | At least one sentence contains a number (kg, mins, rupees, weeks, reps, %, etc.) |
> | SPCL coverage | At least 3 of 4 elements (Status, Power, Credibility, Likeness) are present |
> | Content-type flow | Flow matches the category (TOFU → myth-bust/educational, MOFU → step-by-step/system, BOFU → protocol/blueprint) |
> | Slide 1 | Reinforces the hook — does not just repeat it word for word |
> | Last sentence | Is a bridge to action, not a summary |
> | No vague filler | Zero sentences like "This is important" or "Many people struggle" — every sentence is specific |
> | Topic alignment | Every sentence is about the carousel topic — no tangents |
>
> Fix all failures, then output the final script.
>
> **Respond ONLY with a JSON object**:
> ```json
> {
>   "uuid": "{uuid}",
>   "script": "Sentence one.\nSentence two.\nSentence three.\n..."
> }
> ```
> No explanation. No extra text. Use `\n` between sentences (not actual newlines in JSON).

---

### Step C — Validate this round's results, then checkpoint

**DO NOT use inline `python3 -c` for this step.** Scripts contain em-dashes and other unicode that break shell inline commands. Always write a file and run it.

Using the `create_file` tool, create `/tmp/save_round{N}.py` with this exact structure:

```python
import json

# ── Paste the 10 {uuid, script} dicts here ────────────────────────────
# Use \u2014 for em-dashes in string literals (e.g. \u2014 not —)
scripts = [
    {"uuid": "...", "script": "Sentence one.\nSentence two.\n..."},
    # ... 9 more
]
# ──────────────────────────────────────────────────────────────────────

BATCH_NO = 1  # change per batch
START_NO = 51  # first running_no in this round

# Validate
errors = []
for i, entry in enumerate(scripts):
    uuid = entry['uuid']
    sentences = [s.strip() for s in entry['script'].split('\n') if s.strip()]
    n = len(sentences)
    label = f"#{i + START_NO} ({uuid[:8]})"

    if n < 8 or n > 12:
        errors.append(f"{label}: sentence count {n} (need 8-12)")

    for j, s in enumerate(sentences):
        wc = len(s.split())
        if wc > 20:
            errors.append(f"{label} slide {j+1}: {wc} words > 20 — '{s[:60]}'")
        if s.startswith("I ") or s.startswith("My "):
            errors.append(f"{label} slide {j+1}: starts with I/My — '{s[:60]}'")

    print(f"{label}: {n} sentences")

if errors:
    print("\nFAILURES:")
    for e in errors:
        print(" ", e)
else:
    print("\nAll scripts PASSED validation.")

# Append to checkpoint only if all pass
if not errors:
    checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/scripts_checkpoint.json'
    with open(checkpoint_path, encoding='utf-8') as f:
        checkpoint = json.load(f)
    checkpoint.extend(scripts)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    print(f'Checkpoint updated: {len(checkpoint)} entries total')
else:
    print('\nCheckpoint NOT updated — fix failures first.')
```

Then run it from the project root:
```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 /tmp/save_round{N}.py 2>&1 | tail -20
```

For any failed entry: re-run a fresh dedicated subagent for that carousel, fix the script in the save file, run again. Do not update the checkpoint until all 10 pass.

### Step D — Re-invoke this skill (recursive call)

After checkpointing, **re-invoke this skill**. The new invocation must:
1. **Read `Carousels/skills/step5-script-writer/SKILL.md` in full** — this is the first action, before anything else
2. Read the updated checkpoint
3. Compute the next 10 remaining carousels
4. Process them as a fresh round with a clean context

Repeat until Step A reports `ALL DONE`.

**Terminal condition**: When Step A reports `ALL DONE`, run:

```bash
cp Carousels/data/batch_1/scripts_checkpoint.json Carousels/data/batch_1/scripts.json
```

(Replace `batch_1` with the actual batch number.)

Then proceed to Step E.

### Step E — Insert into DB

```bash
python3 Carousels/scripts/step5_script_writer.py --batch {batch_no} --scripts-file Carousels/data/batch_{batch_no}/scripts.json
```

### Step F — Verify

```bash
python3 Carousels/scripts/orchestrator.py status --batch {batch_no}
```

Confirm `SCRIPT_WRITTEN = 100`.

---

## Success Criteria
- All 100 rows have `script_content` set and `current_stage = SCRIPT_WRITTEN`
- Every script has 8–12 sentences
- No sentence exceeds 20 words
- No sentence starts with "I" or "My"
- Every script contains at least one specific number
- TOFU scripts use myth-bust or educational flow; BOFU scripts use protocol/blueprint flow
