# SKILL: Step 6 — CTA Writer

## Purpose
Write the CTA for each of the 100 `SCRIPT_WRITTEN` carousels.

Each CTA is a **single sentence ≤12 words** that:
- Bridges naturally from the **second-to-last sentence** of the carousel's `script_content`
- Replaces what was the placeholder bridge sentence written in Step 5
- Matches the allowed CTA types for the carousel's `category`
- Requires **zero automation** to execute — no auto-DMs, no comment-keyword triggers

The CTA is saved to the `cta` column. `script_content` is **never modified**.

---

## Allowed CTA Types (No Automation Required)

These CTAs work without ManyChat, Zapier, or any backend trigger. Use only these.

| CTA action | Use for | Example |
|---|---|---|
| Follow | TOFU (primary) | "Follow DadFit for a new dad tip every day." |
| Save | TOFU + MOFU (primary) | "Save this before your next grocery run." |
| Share / Repost | TOFU (primary) | "Share this with a dad who needs to hear it." |
| Tag a friend | TOFU / MOFU | "Tag the dad in your life who skips breakfast." |
| Like | Any (light) | "Like this if part 2 would help." |
| Comment your experience | MOFU (strong) | "Drop your biggest challenge in the comments." |
| Visit profile | MOFU / BOFU | "Take the next step at dadfit.in." |
| Watch Stories | MOFU / BOFU | "Your next move starts at dadfit.in." |
| Link in bio | MOFU / BOFU | "Head to dadfit.in to get started." |

## Banned CTA Patterns (Require Automation — Never Use)

- "Comment X and I'll DM you…"
- "Comment X to get the template"
- "DM me X for…"
- "Type X below and I'll send…"
- Any keyword-triggered auto-response

---

## CTA Mapping by Category

| Category | Primary goal | Primary CTAs | Secondary CTAs |
|---|---|---|---|
| **TOFU** | Reach and audience growth | Follow, Save, Share/Repost | Tag a friend, Like |
| **MOFU** | Trust, engagement, relationship | Save, Comment your experience | Visit profile, Watch Stories, Link in bio |
| **BOFU** | Conversion and commitment | Save, Link in bio, Visit profile | Share, Watch Stories |

---

## Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| `python3 -c "..."` fails when scripts contain `—` | Always write scripts to `/tmp/` using `create_file`, run with `python3` |
| CTA exceeds 12 words | Re-spawn the subagent for that carousel — no manual trimming |
| CTA doesn't bridge from the second-to-last sentence | The subagent must read the full script; re-spawn if the bridge feels disconnected |
| Automation language slips in ("comment X for…") | Reject the entry; re-spawn with explicit "no automation" reinforcement |
| `json.dump` stores `\u2014` escapes | Always pass `ensure_ascii=False` |

---

## Instructions

> **This skill is recursive.** Each invocation handles exactly 10 carousels, then re-invokes itself for the next 10. Never process more than 10 carousels per invocation.

> **Every invocation must start by reading `Carousels/skills/step6-cta-writer/SKILL.md` in full before doing anything else.**

---

### Step A — Check state and decide what to do this invocation

**First invocation only**: Use `create_file` to write `/tmp/step6_stepa.py`. On all subsequent invocations, the file already exists — just run it.

```python
# /tmp/step6_stepa.py
import sqlite3, json, os

BATCH_NO = 1  # change per batch

checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/cta_checkpoint.json'

conn = sqlite3.connect('Carousels/data/db.sqlite')
all_rows = conn.execute(
    'SELECT uuid, running_no, title, category, script_content '
    'FROM Carousel WHERE batch_no = ? AND current_stage = "SCRIPT_WRITTEN" '
    'ORDER BY running_no',
    (BATCH_NO,)
).fetchall()
conn.close()

all_carousels = [{
    'uuid': r[0], 'running_no': r[1], 'title': r[2],
    'category': r[3], 'script_content': r[4]
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
    with open(f'/tmp/batch_{BATCH_NO}_cta_round.json', 'w', encoding='utf-8') as f:
        json.dump(this_round, f, indent=2, ensure_ascii=False)
    print(f'\nWritten to /tmp/batch_{BATCH_NO}_cta_round.json')
```

Run from project root:
```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 /tmp/step6_stepa.py
```

**Read the STATUS line:**
- `ALL DONE` → skip to Step D
- `PROCESS running_no X to Y` → this invocation handles only those carousels. Proceed to Step B.

---

### Step B — Spawn 10 subagents for this round (one per carousel)

Take the 10 carousels from `/tmp/batch_1_cta_round.json`. Spawn all 10 subagents **simultaneously** — one per carousel, no exceptions.

**STRICT RULE — ONE SUBAGENT, ONE CAROUSEL, ONE CTA. NO EXCEPTIONS.**
- Each subagent receives exactly one carousel's data and returns exactly one CTA sentence.
- If a subagent returns multiple CTAs or multiple `uuid` keys: reject entirely and re-spawn.

Use this prompt per subagent (fill in the placeholders):

---

> You are writing the **CTA slide** for a single Instagram carousel for DadFit — a fitness brand for salaried Indian fathers aged 30–45.
>
> **This prompt is for exactly one carousel. Write exactly one CTA sentence. Do not produce output for any other carousel.**
>
> **What the CTA is:**
> The CTA is the final sentence of the carousel — the slide that appears after all the content slides. It must:
> - Flow naturally from the **second-to-last sentence** of the script (provided below)
> - Tell the viewer what to do next — one clear action
> - Be ≤12 words — short, direct, no filler
>
> **Allowed CTA types** (choose the best fit for the category):
> - TOFU → Follow, Save, Share/Repost, Tag a friend, Like
> - MOFU → Save, Comment your experience/opinion, Visit profile, Watch Stories, Link in bio
> - BOFU → Save, Link in bio, Visit profile, Share, Watch Stories
>
> **BANNED — never use these patterns:**
> - "Comment X and I'll DM you…"
> - "Type X below and I'll send…"
> - "DM me X for…"
> - Any conditional "do X to get Y" pattern — these require automation that does not exist
>
> **The CTA must stand alone as one punchy sentence.** It should feel like the natural conclusion of the carousel, not a disconnected instruction.
>
> **Carousel data:**
> - UUID: {uuid}
> - Title: {title}
> - Category: {category}
> - Full script (read carefully — especially the second-to-last and last sentences):
>
> {script_content}
>
> **Self-check before responding:**
> - Is the CTA ≤12 words? If not, shorten it.
> - Does it bridge naturally from the second-to-last sentence of the script?
> - Is it free of any automation-dependent pattern?
> - Does it match the allowed CTA types for {category}?
>
> **Respond ONLY with a JSON object:**
> ```json
> {"uuid": "{uuid}", "cta": "Your CTA sentence here."}
> ```
> No explanation. No extra text.

---

### Step C — Validate this round's results, then checkpoint

**DO NOT use inline `python3 -c` for this step.** Always write a file and run it.

Using the `create_file` tool, create `/tmp/save_cta_round{N}.py`:

```python
import json

# ── Paste the 10 {uuid, cta} dicts here ──────────────────────────────────
scripts = [
    {"uuid": "...", "cta": "Your CTA sentence here."},
    # ... 9 more
]
# ─────────────────────────────────────────────────────────────────────────

BATCH_NO = 1   # change per batch
START_NO = 1   # first running_no in this round

BANNED_PATTERNS = [
    "comment", "type ", "dm me", "dm ", "i'll send", "i will send",
    "and i'll", "and i will", "to get the", "to receive"
]

errors = []
for i, entry in enumerate(scripts):
    uuid = entry['uuid']
    cta = entry['cta'].strip()
    label = f"#{i + START_NO} ({uuid[:8]})"
    wc = len(cta.split())

    if wc > 12:
        errors.append(f"{label}: CTA is {wc} words > 12 — '{cta}'")

    cta_lower = cta.lower()
    for pat in BANNED_PATTERNS:
        if pat in cta_lower:
            errors.append(f"{label}: banned pattern '{pat}' found — '{cta}'")

    print(f"{label}: {wc} words | {cta}")

if errors:
    print("\nFAILURES:")
    for e in errors:
        print(" ", e)
else:
    print("\nAll CTAs PASSED validation.")

if not errors:
    checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/cta_checkpoint.json'
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
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 /tmp/save_cta_round{N}.py 2>&1 | tail -20
```

For any failed entry: re-spawn a fresh dedicated subagent for that carousel. Do not update the checkpoint until all 10 pass.

**Before round 1**: The checkpoint file must exist. Create it if it doesn't:
```bash
echo "[]" > Carousels/data/batch_1/cta_checkpoint.json
```

---

### Step D — Re-invoke this skill (recursive call)

After checkpointing, **re-invoke this skill**. The new invocation must:
1. **Read `Carousels/skills/step6-cta-writer/SKILL.md` in full** — first action, no exceptions
2. Run `/tmp/step6_stepa.py` to get the next round
3. Process as a fresh round with a clean context

Repeat until Step A reports `ALL DONE`.

**Terminal condition**: When Step A reports `ALL DONE`, proceed to Step E.

---

### Step E — Insert into DB

Use `create_file` to write `/tmp/step6_insert.py`, then run it:

```python
import sqlite3, json

BATCH_NO = 1  # change per batch

checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/cta_checkpoint.json'
with open(checkpoint_path, encoding='utf-8') as f:
    entries = json.load(f)

conn = sqlite3.connect('Carousels/data/db.sqlite')

updated = 0
for e in entries:
    conn.execute(
        'UPDATE Carousel SET cta = ?, current_stage = "CTA_WRITTEN" WHERE uuid = ?',
        (e['cta'], e['uuid'])
    )
    updated += 1

conn.commit()
conn.close()

print(f'Updated {updated} rows to CTA_WRITTEN for batch {BATCH_NO}')

# Verify
conn = sqlite3.connect('Carousels/data/db.sqlite')
count = conn.execute(
    'SELECT COUNT(*) FROM Carousel WHERE batch_no = ? AND current_stage = "CTA_WRITTEN"',
    (BATCH_NO,)
).fetchone()[0]
conn.close()
print(f'Verified: {count} rows with current_stage = CTA_WRITTEN')
```

```bash
cd "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content" && python3 /tmp/step6_insert.py
```

---

### Step F — Verify

```bash
python3 Carousels/scripts/orchestrator.py status --batch 1
```

Confirm `CTA_WRITTEN = 100`.

---

## Success Criteria
- All 100 rows have `cta` set and `current_stage = CTA_WRITTEN`
- Every CTA is ≤12 words
- No CTA contains automation-dependent language
- Every CTA matches the allowed types for its `category`
- Every CTA bridges naturally from the second-to-last sentence of `script_content`
