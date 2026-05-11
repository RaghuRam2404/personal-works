# SKILL: Step 7 — Caption Writer

## Purpose
Write the Instagram caption for each of the 100 `CTA_WRITTEN` carousels.

Each caption:
- Is **≤2,200 characters** (Instagram's hard limit — count includes spaces, line breaks, punctuation)
- Directly **answers the question posed by the carousel title** — GEO-optimized so AI models (Gemini, ChatGPT, Perplexity) can understand, cite, and surface this content
- **Repeats the primary keyword naturally 3–4 times** throughout the body — not stuffed, woven in contextually
- Uses **no hashtags** — zero `#` characters anywhere in the caption
- Is **conversational and direct** — no bullet points, no teases, no "swipe to find out"
- Opens with the carousel's `hook`, closes with the carousel's `cta`

The caption is saved to the `caption` column. No other column is modified.

---

## Caption Structure

```
{hook}

{body paragraph 1 — answer the question directly, set the context}

{body paragraph 2 — go deeper, name the mechanism or root cause}

{body paragraph 3 — give the practical takeaway or actionable insight}

{body paragraph 4 (optional) — relate it to the dad's daily reality, add credibility}

{cta}
```

**Rules for the body:**
- 3–4 short paragraphs. Each paragraph is 2–4 sentences.
- The **primary keyword** (from the `keyword` field) appears **3–4 times** across the body. Use it as it would appear in a natural sentence — not forced.
- Write as if answering a Google search or an AI assistant query. "If someone searched '{title}', this caption should be the answer."
- No teasing. No "save this for later." No "swipe left." Say the thing directly.
- Language is plain, warm, grounded — a knowledgeable friend talking to a 35-year-old Indian dad.

---

## GEO-Optimization Principles

Generative Engine Optimization (GEO) means writing so AI models can extract and cite your answer. Apply these:

1. **Answer first** — the first paragraph states the answer, not a buildup to it
2. **Name the concept** — use the keyword as it would appear in a search query (e.g., "fat loss for Indian men", "skinny fat Indian diet")
3. **Be specific** — numbers, timeframes, named mechanisms beat vague advice
4. **Sound authoritative but human** — cite what works, not what's trendy
5. **Repeat the keyword contextually** — each repeat should feel earned, not pasted in

---

## Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| Caption exceeds 2,200 characters | Re-spawn the subagent — no manual trimming |
| Hashtags appear (`#`) | Re-spawn — zero tolerance |
| Caption opens with body instead of hook | Re-spawn — hook must be the first line |
| Caption closes with anything other than the CTA | Re-spawn — CTA must be the last line |
| Keyword appears 0–1 times | Re-spawn with explicit count instruction |
| `python3 -c "..."` fails with unicode/dash characters | Always write scripts to `/tmp/` via `create_file`, run with `python3` |
| `json.dump` stores `\u2014` escapes | Always pass `ensure_ascii=False` |

---

## Instructions

> **This skill is recursive.** Each invocation handles exactly 10 carousels, then re-invokes itself for the next 10. Never process more than 10 carousels per invocation.

> **Every invocation must start by reading `Carousels/skills/step7-caption-writer/SKILL.md` in full before doing anything else.**

---

### Step A — Check state and decide what to do this invocation

**First invocation only**: Use `create_file` to write `/tmp/step7_stepa.py`. On all subsequent invocations, the file already exists — just run it.

```python
# /tmp/step7_stepa.py
import sqlite3, json, os

BATCH_NO = 1  # change per batch

checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/caption_checkpoint.json'

conn = sqlite3.connect('Carousels/data/db.sqlite')
all_rows = conn.execute(
    'SELECT uuid, running_no, title, keyword, category, hook, script_content, cta '
    'FROM Carousel WHERE batch_no = ? AND current_stage = "CTA_WRITTEN" '
    'ORDER BY running_no',
    (BATCH_NO,)
).fetchall()
conn.close()

all_carousels = [{
    'uuid': r[0], 'running_no': r[1], 'title': r[2], 'keyword': r[3],
    'category': r[4], 'hook': r[5], 'script_content': r[6], 'cta': r[7]
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
    with open(f'/tmp/batch_{BATCH_NO}_caption_round.json', 'w', encoding='utf-8') as f:
        json.dump(this_round, f, indent=2, ensure_ascii=False)
    print(f'\nWritten to /tmp/batch_{BATCH_NO}_caption_round.json')
```

**Before round 1**: Create the checkpoint file if it doesn't exist:
```bash
echo "[]" > Carousels/data/batch_1/caption_checkpoint.json
```

Run from project root:
```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 /tmp/step7_stepa.py
```

**Read the STATUS line:**
- `ALL DONE` → skip to Step D
- `PROCESS running_no X to Y` → this invocation handles only those carousels. Proceed to Step B.

---

### Step B — Spawn 10 subagents for this round (one per carousel)

Take the 10 carousels from `/tmp/batch_1_caption_round.json`. Spawn all 10 subagents **simultaneously** — one per carousel, no exceptions.

**STRICT RULE — ONE SUBAGENT, ONE CAROUSEL, ONE CAPTION. NO EXCEPTIONS.**

Use this prompt per subagent (fill in the placeholders):

---

> You are writing the **Instagram caption** for a single carousel post for DadFit — a fitness brand for salaried Indian fathers aged 30–45.
>
> **This prompt is for exactly one carousel. Write exactly one caption. Do not produce output for any other carousel.**
>
> ---
>
> **What the caption must do:**
> This carousel's title is the exact question your audience searches for. The caption must **directly answer that question** — so that when someone reads it (or when an AI model like ChatGPT, Gemini, or Perplexity processes it), it reads as a clear, authoritative, citable answer.
>
> Write as if answering: *"If someone searched '{title}', what is the complete, honest answer?"*
>
> ---
>
> **Caption structure (follow this exactly):**
>
> ```
> {hook}
> [blank line]
> [body paragraph 1 — answer the question directly, set the context, 2–4 sentences]
> [blank line]
> [body paragraph 2 — go deeper: name the mechanism, root cause, or key insight, 2–4 sentences]
> [blank line]
> [body paragraph 3 — the practical takeaway: what should the dad actually do, 2–4 sentences]
> [blank line]
> [body paragraph 4 — optional: ground it in the dad's daily reality, add credibility, 2–3 sentences]
> [blank line]
> {cta}
> ```
>
> **Rules:**
> - The caption opens with the `hook` (provided below) — copy it exactly as the first line
> - The caption closes with the `cta` (provided below) — copy it exactly as the last line
> - Body is 3–4 paragraphs separated by blank lines — no bullet points, no numbered lists
> - The primary keyword `{keyword}` must appear **3–4 times** in the body — use it naturally, in context, as it would appear in a real sentence
> - Total caption length: **≤2,200 characters** — count every character including spaces, line breaks, and punctuation. Do not exceed this under any circumstances.
> - **Zero hashtags** — no `#` character anywhere in the caption
> - Language: conversational, warm, direct — a knowledgeable friend talking to a busy 35-year-old dad. Not a textbook. Not a listicle.
> - **Answer first** — don't build up to the answer. State it in paragraph 1.
> - Be specific — numbers, timeframes, named mechanisms are better than vague advice
>
> **Self-check before responding:**
> 1. Does the caption start with the hook (exact text)?
> 2. Does the caption end with the cta (exact text)?
> 3. Does the keyword appear 3–4 times in the body?
> 4. Is the total character count ≤2,200? (count carefully — err on the side of shorter)
> 5. Are there zero hashtags?
> 6. Are there zero bullet points or numbered lists?
>
> ---
>
> **Carousel data:**
> - UUID: {uuid}
> - Title: {title}
> - Keyword: {keyword}
> - Category: {category}
> - Hook: {hook}
> - CTA: {cta}
> - Script content (read this to understand the carousel's angle and key points):
>
> {script_content}
>
> ---
>
> **Respond ONLY with a JSON object:**
> ```json
> {"uuid": "{uuid}", "caption": "Full caption text here.\n\nWith blank lines between paragraphs.\n\nEnding with the CTA."}
> ```
> No explanation. No extra text. The caption value must be a single JSON string with `\n\n` between paragraphs.

---

### Step C — Validate this round's results, then checkpoint

**DO NOT use inline `python3 -c` for this step.** Always write a file and run it.

Using the `create_file` tool, create `/tmp/save_caption_round{N}.py`:

```python
import json

# ── Paste the 10 {uuid, caption} dicts here ──────────────────────────────
scripts = [
    {"uuid": "...", "caption": "..."},
    # ... 9 more
]
# ─────────────────────────────────────────────────────────────────────────

BATCH_NO = 1   # change per batch
START_NO = 1   # first running_no in this round

errors = []
for i, entry in enumerate(scripts):
    uuid = entry['uuid']
    caption = entry['caption'].strip()
    label = f"#{i + START_NO} ({uuid[:8]})"
    char_count = len(caption)

    if char_count > 2200:
        errors.append(f"{label}: caption is {char_count} chars > 2200")

    if '#' in caption:
        errors.append(f"{label}: hashtag found — '#' not allowed")

    print(f"{label}: {char_count} chars | OK" if not any(uuid[:8] in e for e in errors) else f"{label}: {char_count} chars | FAIL")

if errors:
    print("\nFAILURES:")
    for e in errors:
        print(" ", e)
else:
    print("\nAll captions PASSED validation.")

if not errors:
    checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/caption_checkpoint.json'
    with open(checkpoint_path, encoding='utf-8') as f:
        checkpoint = json.load(f)
    checkpoint.extend(scripts)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    print(f'Checkpoint updated: {len(checkpoint)} entries total')
else:
    print('\nCheckpoint NOT updated — fix failures first.')
```

Run it:
```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 /tmp/save_caption_round{N}.py 2>&1 | tail -20
```

For any failed entry: re-spawn a fresh dedicated subagent for that carousel only. Do not update the checkpoint until all 10 pass.

---

### Step D — Re-invoke this skill (recursive call)

After checkpointing, **re-invoke this skill**. The new invocation must:
1. **Read `Carousels/skills/step7-caption-writer/SKILL.md` in full** — first action, no exceptions
2. Run `/tmp/step7_stepa.py` to get the next round
3. Process as a fresh round with a clean context

Repeat until Step A reports `ALL DONE`.

**Terminal condition**: When Step A reports `ALL DONE`, proceed to Step E.

---

### Step E — Insert into DB

Use `create_file` to write `/tmp/step7_insert.py`, then run it:

```python
import sqlite3, json

BATCH_NO = 1  # change per batch

checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/caption_checkpoint.json'
with open(checkpoint_path, encoding='utf-8') as f:
    entries = json.load(f)

conn = sqlite3.connect('Carousels/data/db.sqlite')

updated = 0
for e in entries:
    conn.execute(
        'UPDATE Carousel SET caption = ?, current_stage = "CAPTION_WRITTEN" WHERE uuid = ?',
        (e['caption'], e['uuid'])
    )
    updated += 1

conn.commit()
conn.close()

print(f'Updated {updated} rows to CAPTION_WRITTEN for batch {BATCH_NO}')

# Verify
conn = sqlite3.connect('Carousels/data/db.sqlite')
count = conn.execute(
    'SELECT COUNT(*) FROM Carousel WHERE batch_no = ? AND current_stage = "CAPTION_WRITTEN"',
    (BATCH_NO,)
).fetchone()[0]
conn.close()
print(f'Verified: {count} rows with current_stage = CAPTION_WRITTEN')
```

```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 /tmp/step7_insert.py
```

---

### Step F — Verify

```bash
python3 Carousels/scripts/orchestrator.py status --batch 1
```

Confirm `CAPTION_WRITTEN = 100`.

---

## Success Criteria
- All 100 rows have `caption` set and `current_stage = CAPTION_WRITTEN`
- Every caption is ≤2,200 characters
- Every caption opens with the carousel's `hook` (exact text)
- Every caption closes with the carousel's `cta` (exact text)
- The primary keyword appears 3–4 times naturally in the body
- Zero hashtags (`#`) in any caption
- No bullet points or numbered lists in any caption
