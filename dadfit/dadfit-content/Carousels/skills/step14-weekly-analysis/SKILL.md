---
name: step14-weekly-analysis
description: "Generates a weekly performance report for a batch of published carousels. Reads all CarouselPerformance rows, outputs top 10 by saves/reach/follows, TOFU vs MOFU vs BOFU engagement comparison, and week-over-week growth per carousel. Saves report to Carousels/out/batch_{n}_week_{w}_report.md."
argument-hint: "Batch number and/or week number (both optional — auto-detected from DB)"
---

# Step 14 — Weekly Analysis

Generates a Markdown performance report for a batch, broken down by week. Run this once per week (or any time you want a snapshot).

---

## Prerequisites

Activate the project venv:

```bash
source Carousels/.venv/bin/activate
```

Step 13 must have run at least once — `CarouselPerformance` needs data.

---

## Usage

```bash
# Auto-detect latest batch and latest week
python3 Carousels/scripts/step14_analysis.py

# Specific batch
python3 Carousels/scripts/step14_analysis.py --batch 1

# Specific batch + week
python3 Carousels/scripts/step14_analysis.py --batch 1 --week 2
```

Output is saved to:

```
Carousels/out/batch_{N}_week_{W}_report.md
```

---

## Week bucketing

Week buckets are calculated **relative to each carousel's `published_date`**, not the calendar week:

| Week | Days since publish |
|---|---|
| Week 1 | Days 0 – 6 |
| Week 2 | Days 7 – 13 |
| Week 3 | Days 14 – 20 |

Within the 15-day monitoring window (Step 13), you'll typically have data for weeks 1 and 2.

---

## Report sections

### 1 — Top 10 by Saves
Ranks all carousels in the batch by total **saves** (summed across all their performance entries). Best proxy for content value.

### 2 — Top 10 by Reach
Ranks by total **reach** (unique accounts that saw the post). Best proxy for distribution.

### 3 — Top 10 by Follows
Ranks by total **follows_from_post**. Measures growth impact.

### 4 — TOFU vs MOFU vs BOFU Engagement
Average per-metric comparison across all carousels in each funnel stage:
- Avg Views, Reach, Likes, Saves, Profile Visits, Follows

### 5 — Week-over-Week Growth
For each carousel that has data in both week `W-1` and week `W`, shows the current week's totals and a `▲/▼ %` delta for Views, Saves, Reach, and Follows.

> ⚠️ Week-over-week requires `--week 2` or later. Week 1 reports show top performers and category comparison only.

---

## Notes

- The report is **additive** — re-running for the same batch/week overwrites the previous file
- All metrics are **summed** across all performance entries in the week bucket (not averaged)
- For top-10 tables, metrics are summed across **all weeks**, giving lifetime totals
- `--week` auto-detects the highest week bucket seen across all performance entries if omitted

---

## Script location

`Carousels/scripts/step14_analysis.py`
