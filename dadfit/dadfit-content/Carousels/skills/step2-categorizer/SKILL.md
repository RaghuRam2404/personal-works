# SKILL: Step 2 — TOFU / MOFU / BOFU Categorizer

## UUID Rule

Never type, guess, or copy UUIDs from memory. Always query the DB to obtain UUIDs for `categories.json`. See `Carousels/skills/generate-uuid/SKILL.md` for the exact commands.

**To export all UUIDs for a batch before building categories.json:**
```bash
python3 -c "
import sqlite3, json
conn = sqlite3.connect('Carousels/data/db.sqlite')
rows = conn.execute('SELECT uuid, title FROM Carousel WHERE batch_no=BATCH_NO AND current_stage=\'TOPIC_FETCHED\' ORDER BY rowid').fetchall()
print(json.dumps([{'uuid': r[0], 'title': r[1]} for r in rows], indent=2))
conn.close()
" > /tmp/batch_BATCH_NO_uuids.json
```
Then build `categories.json` by copy-pasting from that query output — never retype UUID strings.

## Purpose
Assign a funnel stage (`TOFU`, `MOFU`, or `BOFU`) to every carousel in a batch that is at `current_stage = TOPIC_FETCHED`, then update the DB to `current_stage = CATEGORIZED`.

## Inputs
- Batch number (passed at runtime)
- All rows in `Carousel` table with `current_stage = TOPIC_FETCHED` for that batch
- No external resource files needed — categorization is based on topic intent alone

## Output
- All 100 rows updated: `category = TOFU | MOFU | BOFU`, `current_stage = CATEGORIZED`
- A `categories.json` file saved to `Carousels/data/batch_{batch_no}/categories.json` for audit

---

## Definitions

| Stage | Meaning | Signals in the title |
|---|---|---|
| **TOFU** | Top of Funnel — Awareness | "Why...", "5 Signs...", myth-busting, problem identification, curiosity hooks |
| **MOFU** | Middle of Funnel — Consideration | "How to...", "The X-Step Fix", frameworks, systems, protocols with education |
| **BOFU** | Bottom of Funnel — Conversion | Specific named programs, "The Exact...", transformation stories, direct action plans with numbers |

---

## Instructions

### Step A — Load topics from DB

Query the DB for all carousels in the batch at `TOPIC_FETCHED`:
```sql
SELECT uuid, title FROM Carousel
WHERE batch_no = {batch_no} AND current_stage = 'TOPIC_FETCHED'
ORDER BY rowid;
```

### Step B — Spawn 10 subagents

Split the 100 topics into 10 groups of 10 (by DB row order). For each group, spawn a subagent with this prompt:

> You are a DadFit content strategist. Assign each of the following carousel topics a funnel stage: TOFU, MOFU, or BOFU.
>
> Rules:
> - TOFU = Awareness. Content that educates, reveals a problem, debunks a myth, or sparks curiosity. The viewer may not know they have this problem yet. Example signals: "Why...", "5 Signs...", "The Truth About...", myth-busting phrases.
> - MOFU = Consideration. Content that offers a framework, system, how-to, or step-by-step approach to a problem the viewer already knows they have. Example signals: "How to...", "The X-Step Fix", "A Protocol For...", "The Guide To...".
> - BOFU = Conversion. Specific, named, actionable programs or plans with concrete outputs (numbers, weeks, results). Content that asks the viewer to commit to a specific method. Example signals: "The Exact...", "The [Name] Program", "Built in X Weeks", transformation stories with systems.
>
> Topics:
> {list of 10 titles with their UUIDs}
>
> Respond with a JSON array: [{"uuid": "...", "tofu_mofu_bofu": "TOFU|MOFU|BOFU"}, ...]

Collect all 10 subagent responses. Merge into a single list of 100 `{uuid, tofu_mofu_bofu}` objects.

### Step C — Save categories.json

Save the merged list to:
```
Carousels/data/batch_{batch_no}/categories.json
```

Format:
```json
[
  {"uuid": "...", "tofu_mofu_bofu": "TOFU"},
  {"uuid": "...", "tofu_mofu_bofu": "MOFU"},
  ...
]
```

### Step D — Insert into DB

Run the categorizer script:
```bash
python3 Carousels/scripts/step2_categorizer.py --batch {batch_no} --categories-file Carousels/data/batch_{batch_no}/categories.json
```

The script validates (exactly 100 entries, valid values, all UUIDs match the batch) and updates in a single transaction.

### Step E — Verify

After updating, run:
```bash
python3 Carousels/scripts/orchestrator.py status --batch {batch_no}
```

Confirm `CATEGORIZED = 100`. Show the user the TOFU / MOFU / BOFU distribution before proceeding.

---

## Success Criteria
- All 100 rows have `category` set to `TOFU`, `MOFU`, or `BOFU`
- All 100 rows have `current_stage = CATEGORIZED`
- `categories.json` saved to `Carousels/data/batch_{batch_no}/categories.json`
- Distribution is reasonable (no single stage dominates at >60% of total)
