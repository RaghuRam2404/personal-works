# SKILL: Step 4 — Hook Writer

## UUID Rule
Never type or fabricate UUIDs. Query the DB to obtain UUIDs. See `Carousels/skills/generate-uuid/SKILL.md`.

## Purpose
Write a **carousel cover hook** for each of the 100 `ORDER_SET` carousels in the batch. The hook is the only text the viewer sees before deciding whether to swipe. Its one job: create an open loop so compelling they cannot scroll past.

## Inputs
- All `ORDER_SET` carousels for the batch (uuid, title, keyword, trait, category from DB)
- `Resources/1000-Viral-Hooks.md` — reference for hook patterns (read fresh each run)
- Previous batch hooks from DB (to avoid repeats)

## Output
- All 100 rows updated: `hook` set, `current_stage = HOOK_WRITTEN`
- `hooks.json` saved to `Carousels/data/batch_{batch_no}/hooks.json` for audit

---

## What Makes a Hook Work for Carousels

The hook appears on the **cover slide** — not as a video caption. The viewer is scrolling their Instagram feed. They see the cover image for less than 1 second. If the hook doesn't stop them, the carousel never gets seen.

### The only goal: create an open loop
An open loop is an unresolved tension in the viewer's mind. The hook raises a question, reveals a problem, or teases a specific insight — and makes them swipe to resolve it.

**Open loops that work for DadFit:**
- Surfaces a hidden truth they didn't know about themselves
- Challenges something they believe is correct
- Names a specific pain they feel but haven't articulated
- Promises a surprising number or result they want to know

### What "5–7 words" means in practice
The hook must fit on a single cover slide and be readable in under 1 second. That means:
- 5–7 words is the hard ceiling — count every word
- No punctuation hedging ("maybe", "possibly", "might") — hooks are declarative or interrogative
- No filler words ("the", "a", "of" count toward the limit — use them only if necessary)
- Start with a strong first word: a number, "Why", "Stop", "Your", "This"

### Hook patterns by funnel stage

**TOFU hooks** (awareness — viewer may not know the problem exists)
- Pattern: Name the hidden problem / Myth-bust a common belief / Reveal a surprising fact
- Examples:
  - "Your Rice Is Keeping You Fat" (names the unseen cause)
  - "Busy Dads Lose Muscle, Not Fat" (counter-intuitive truth)
  - "Sitting Is Worse Than Smoking" (shocking fact with specificity)
  - "Why 10000 Steps Isn't Enough" (challenges common belief)

**MOFU hooks** (consideration — viewer knows the problem, wants the fix)
- Pattern: Promise a specific method / System / Number of steps that solves a known pain
- Examples:
  - "The 3-Lift Plan For Time-Strapped Dads" (specific method for known constraint)
  - "Fix Back Pain In 10 Minutes Daily" (specific time + outcome)
  - "Eat 100g Protein Without Cooking Extra" (specific number + constraint solved)

**BOFU hooks** (conversion — viewer is ready to act, needs the exact plan)
- Pattern: Name the exact program / Blueprint / Result with a specific number
- Examples:
  - "The 12-Week Indian Dad Recomposition" (named program + specific duration)
  - "Your 5-Day High-Protein Meal Plan" (ownership + specific deliverable)
  - "30-Day Sleep Protocol That Works" (named protocol + specific duration)
- **Hooks must never start with "My" or "I"** across all stages — content is not first-person storytelling. Use named protocols, blueprints, or "Your" ownership instead.

---

## Instructions

### Step A — Load context

**Get previous batch hooks (to avoid repeats):**
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('Carousels/data/db.sqlite')
rows = conn.execute('SELECT hook FROM Carousel WHERE batch_no < BATCH_NO AND hook IS NOT NULL').fetchall()
for r in rows: print(r[0])
conn.close()
"
```

**Get current batch carousels sorted by running_no:**
```bash
python3 -c "
import sqlite3, json
conn = sqlite3.connect('Carousels/data/db.sqlite')
rows = conn.execute('''
    SELECT uuid, running_no, title, keyword, trait, category
    FROM Carousel WHERE batch_no = BATCH_NO AND current_stage = \"ORDER_SET\"
    ORDER BY running_no
''').fetchall()
print(json.dumps([{
    \"uuid\": r[0], \"running_no\": r[1], \"title\": r[2],
    \"keyword\": r[3], \"trait\": r[4], \"category\": r[5]
} for r in rows], indent=2))
conn.close()
" > /tmp/batch_BATCH_NO_carousels.json
```

### Step B — Spawn 10 subagents

Split the 100 carousels into 10 groups of 10. Group by `running_no` order (1–10, 11–20, …).

For each group spawn a subagent with this prompt:

---

> You are writing Instagram carousel cover hooks for DadFit — a fitness, lifestyle, and financial brand for salaried Indian fathers aged 30–45.
>
> **The hook is the first line on the carousel cover slide.** It must make the viewer stop scrolling and swipe right within 1 second. The hook creates an open loop — an unresolved tension or question — that can only be resolved by reading the carousel.
>
> **Rules:**
> - Exactly 5–7 words (count every word — no exceptions)
> - No colons, ellipses, or question marks unless the hook is interrogative — one punctuation style max
> - Do not repeat any hook from the previous batch (list provided below)
> - Do not use vague hooks like "Fitness Tips for Busy Dads" — be specific and punchy
> - TOFU hooks: name a hidden problem or myth-bust ("Your Chai Is Wrecking Your Sleep")
> - MOFU hooks: promise a specific method or number ("Fix Posture In 10 Minutes Daily")
> - BOFU hooks: name the exact plan or result ("The 12-Week Dad Recomposition Blueprint")
> - The hook must be strictly about the carousel's topic — do not wander
> - **Never start with "I" or "My"** across all funnel stages — content is not first-person storytelling. Use named protocols, blueprints, or "Your" framing instead
> - Do not start with "You should" or "Here are"
> - Strong opening words: a number, "Why", "Stop", "Your", "The", "How", "Fix", or a direct noun
>
> **Previous batch hooks (do not repeat):**
> {list of previous batch hooks, one per line — empty if batch 1}
>
> **Carousels to write hooks for (include the uuid in your response for each):**
> {list of 10 carousels with running_no, uuid, title, category, keyword}
>
> **Respond ONLY with a JSON array:**
> ```json
> [
>   {"uuid": "...", "hook": "5 to 7 word hook here"},
>   ...
> ]
> ```
> No explanation. No extra text.

---

Collect all 10 subagent responses. Merge into a single list of 100 `{uuid, hook}` objects.

### Step C — Validate before saving

Check every hook manually or with a script:
- Word count is between 5 and 7 (inclusive)
- No hook is identical to another in this batch or the previous batch
- No hook is a direct paraphrase of the carousel's title (the hook must feel different from the title)
- uuid matches a real carousel in the batch

Fix any violations before proceeding.

### Step D — Save hooks.json

```
Carousels/data/batch_{batch_no}/hooks.json
```

Format:
```json
[
  {"uuid": "...", "hook": "Why Sitting Is Wrecking Your Back"},
  ...
]
```

### Step E — Insert into DB

```bash
python3 Carousels/scripts/step4_hook_writer.py --batch {batch_no} --hooks-file Carousels/data/batch_{batch_no}/hooks.json
```

### Step F — Verify

```bash
python3 Carousels/scripts/orchestrator.py status --batch {batch_no}
```

Confirm `HOOK_WRITTEN = 100`. Show the full hook list sorted by `running_no` before proceeding.

---

## Success Criteria
- All 100 rows have `hook` set and `current_stage = HOOK_WRITTEN`
- Every hook is 5–7 words (no exceptions)
- No hook duplicates any hook from this batch or the previous batch
- No hook is a word-for-word paraphrase of the carousel title
- TOFU hooks open a loop, MOFU hooks promise a method, BOFU hooks name a plan
