# SKILL: Step 1 — Topic Fetcher

## UUID Rule

Never type or fabricate UUID strings. The insertion script (`step1_topic_fetcher.py`) generates all UUIDs via `uuid.uuid4()` automatically — do not put UUIDs in `topics.json`. See `Carousels/skills/generate-uuid/SKILL.md` for the full rule.

## Purpose
Generate exactly 100 unique carousel topics for a new batch and insert them into the `Carousel` table in `Carousels/data/db.sqlite`.

## Inputs
- `Resources/SubNiche-Keyword-Traits.md` — source of subniches, keywords, and trait mappings (read fresh each run)
- User's own topic ideas (collected at runtime via prompt)
- Batch number (collected at runtime via prompt)
- Existing topics in the DB (to avoid repeats across batches)

## Output
100 new rows in the `Carousel` table with `current_stage = TOPIC_FETCHED`.
A `topics.json` file saved to `Carousels/data/batch_{batch_no}/topics.json` with all generated topics for audit.

---

## Instructions

### Step A — Collect runtime inputs

Ask the user:
1. "What is the batch number for this run?" (integer)
2. "Do you have any personal topic ideas to include? If yes, list them now (one per line). Type DONE when finished." Collect all ideas. These are injected as mandatory topics — include every one of them.

### Step B — Read the resource file

Read `Resources/SubNiche-Keyword-Traits.md` in full. Extract:
- All subniches (under `# Subniches`)
- All keywords grouped by category (under `# Keywords`)
- The keyword-to-trait mappings (under `# Keyword vs Traits`)

Do not rely on hardcoded data — read the file fresh every time.

### Step C — Load existing topics

Query the DB to fetch titles from the most recent completed batch only (to avoid repeating the last 100):
```sql
SELECT title FROM Carousel WHERE batch_no = (SELECT MAX(batch_no) FROM Carousel);
```
Store these titles. No new topic title may match any of them.

### Step D — Determine topic distribution

Apply the following percentage split to 100 topics:

| Category | Subniches | Target % | Target count |
|---|---|---|---|
| **Diet & Workout** | Skinny-fat recomposition, Desi diet protein optimization, Time-crunched strength training, Desk-worker posture & mobility | **70%** | **70** |
| **Lifestyle** | Sleep architecture & recovery, Stress & cortisol management, Habit systems & discipline | **20%** | **20** |
| **Financial** | Automated wealth building, Family risk management, Lifestyle inflation control | **10%** | **10** |

Distribute within each category evenly across its subniches (±1 is acceptable). For example:
- Diet & Workout: 70 topics ÷ 4 subniches ≈ 17–18 each
- Lifestyle: 20 topics ÷ 3 subniches ≈ 6–7 each
- Financial: 10 topics ÷ 3 subniches ≈ 3–4 each

If the user supplies personal ideas, assign them to their most appropriate subniche and count them toward that subniche's quota.

### Step E — Generate topics

For each subniche's allocated count, generate that many unique topics. For each topic produce:
- `title` — a specific, carousel-friendly title (e.g. "5 High-Protein Indian Breakfasts Under 10 Minutes")
- `keyword` — the closest matching keyword from the `Keywords` section of the resource file
- `trait` — the most relevant traits from the `Keyword vs Traits` section (comma-separated, max 3)

**Rules:**
- No `title` or `keyword` may duplicate any existing DB record
- Titles must be specific and actionable — not generic (e.g. avoid "Fitness Tips for Dads")
- Titles should feel like carousel hooks: punchy, specific, curiosity-driving
- Do not repeat the same keyword across more than 3 topics in the same batch
- Draw keywords and traits directly from the resource file — do not invent them

### Step F — Save topics.json

Before inserting into DB, save the generated topics to:
```
Carousels/data/batch_{batch_no}/topics.json
```

Format:
```json
[
  {
    "title": "...",
    "keyword": "...",
    "trait": "...",
    "subniche": "...",
    "category": "Diet & Workout | Lifestyle | Financial"
  },
  ...
]
```

### Step G — Insert into DB

Run the insertion script:
```bash
python3 Carousels/scripts/step1_topic_fetcher.py --batch {batch_no} --topics-file Carousels/data/batch_{batch_no}/topics.json
```

The script validates and inserts all rows in a single transaction. If it fails, fix the issue and re-run — do not insert partial data.

### Step H — Verify

After inserting, run:
```bash
python3 Carousels/scripts/orchestrator.py status --batch {batch_no}
```
Confirm `TOPIC_FETCHED = 100`. Show the user a summary grouped by category and subniche before proceeding.

---

## Success Criteria
- Exactly 100 rows in DB with `batch_no = {batch_no}` and `current_stage = TOPIC_FETCHED`
- Distribution: ~70 Diet & Workout, ~20 Lifestyle, ~10 Financial (±2 per subniche acceptable)
- No duplicate titles or keywords vs any previous batch
- All user-supplied ideas are included
- `topics.json` saved to `Carousels/data/batch_{batch_no}/topics.json`
