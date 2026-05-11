# SKILL: Step 3 — Posting Order Setter

## UUID Rule
Never type or fabricate UUIDs. The script reads all UUIDs directly from the DB — no manual UUID handling required in this step. See `Carousels/skills/generate-uuid/SKILL.md`.

## Purpose
Assign `running_no` 1–100 to all `CATEGORIZED` carousels in a batch using a funnel-aware, content-pillar-interleaved ordering strategy, then update DB to `current_stage = ORDER_SET`.

---

## Ordering Philosophy

### Why not a rigid 3 TOFU → 3 MOFU → 4 BOFU pattern?
A mechanical fixed sequence ignores the actual TOFU/MOFU/BOFU distribution in the batch (which rarely divides cleanly into thirds), and it ignores content pillar variety. Posting 10 Diet & Workout carousels in a row — even if TOFU → MOFU → BOFU — causes audience fatigue and reduces reach diversity.

### The Principles

#### 1. Each 10-post content-day follows an awareness → consideration → conversion arc
- **TOFU posts open the day** — they hook new viewers, surface a pain point, or challenge a myth. They build reach with people who don't yet know they need DadFit.
- **MOFU posts fill the middle** — they educate the warm audience who engaged with the TOFU content. They build authority and desire.
- **BOFU posts close the day** — they ask warm, educated viewers to take a specific action. Placed at the end when trust has been built within the day's rhythm.

#### 2. Day templates vary strategically across the batch
The batch is not uniform. The opening days need more awareness content (new audience). The closing days need more conversion content (warm audience built over the month).

| Days | Template | TOFU | MOFU | BOFU | Rationale |
|---|---|---|---|---|---|
| 1–3 | B | 3 | 5 | 2 | Launch: build awareness first |
| 4–9 | A | 2 | 5 | 3 | Standard arc: educate and convert |
| 10 | C | 2 | 4 | 4 | Close: push conversion with warm audience |

Totals: **23 TOFU + 49 MOFU + 28 BOFU = 100** (matches batch distribution exactly).

#### 3. Content pillars interleave within each funnel stage
Within each funnel stage (TOFU, MOFU, BOFU), the script cycles through content pillars in this order: **Diet & Workout → Lifestyle → Financial → repeat**.

This means:
- No pillar runs more than one post consecutively within the same funnel stage
- Financial posts (10% of batch) appear periodically — after fitness content has warmed the audience — not in isolation
- Lifestyle posts break up the dominant Diet & Workout content to prevent monotony

#### 4. The 3-3-4 posting calendar
Each 10-post content-day maps to **3 actual calendar days**: the first 3 posts (running_no N+1 to N+3) post on calendar day 1, the next 3 (N+4 to N+6) on day 2, the last 4 (N+7 to N+10) on day 3. So posts 1–10 → calendar days 1–3, posts 11–20 → days 4–6, and so on.

---

## Instructions

### Step A — Dry-run first

```bash
python3 Carousels/scripts/step3_order_setter.py --batch {batch_no} --dry-run
```

Review the per-day distribution table. Confirm:
- Each day has TOFU items before MOFU/BOFU items
- Pillars vary within each day (no 3+ Diet & Workout in a row)
- Financial posts appear scattered, not clustered

### Step B — Apply the order

```bash
python3 Carousels/scripts/step3_order_setter.py --batch {batch_no}
```

### Step C — Verify

```bash
python3 Carousels/scripts/orchestrator.py status --batch {batch_no}
```

Confirm `ORDER_SET = 100`.

---

## Success Criteria
- All 100 rows have a unique `running_no` (1–100)
- All 100 rows have `current_stage = ORDER_SET`
- Each content-day (10 posts) has at least 2 TOFU, 4 MOFU, and 2 BOFU
- No content pillar appears in more than 3 consecutive slots within a day-group
- Financial posts are never the first post of a day-group
