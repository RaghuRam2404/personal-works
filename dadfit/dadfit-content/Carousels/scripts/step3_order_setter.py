"""
step3_order_setter.py
Assigns running_no 1-100 to CATEGORIZED carousels using a funnel-aware,
content-pillar-interleaved ordering strategy.

Ordering strategy:
  - 100 carousels split into 10 "content-day groups" of 10 posts each
  - Each content-day follows a TOFU → MOFU → BOFU arc (awareness first, conversion last)
  - Day templates are computed dynamically from the actual TOFU/MOFU/BOFU counts
    so the script works correctly for any batch distribution, not just 23T/49M/28B
  - Launch days (1-3) get more TOFU slots; close day (10) gets more BOFU slots
  - Within each funnel stage, content pillars cycle: Diet & Workout → Lifestyle → Financial
  - The 3-3-4 posting calendar maps each 10-post content-day to 3 actual posting days

Usage:
  python3 Carousels/scripts/step3_order_setter.py --batch 1 [--dry-run]
"""

import sqlite3
import os
import json
import math
import argparse

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "db.sqlite")

FUNNEL_ORDER = ["TOFU", "MOFU", "BOFU"]

# Preferred fallback order per slot stage when that stage is exhausted.
# Avoids placing TOFU in late-day (BOFU) positions which would break the arc.
_FALLBACK = {
    "TOFU": ["MOFU", "BOFU"],   # if TOFU exhausted, prefer MOFU over BOFU
    "MOFU": ["TOFU", "BOFU"],   # if MOFU exhausted, either side is acceptable
    "BOFU": ["MOFU", "TOFU"],   # if BOFU exhausted, prefer MOFU over reverting to TOFU
}

# Pillar cycle order: Diet & Workout first (70% of batch),
# then Lifestyle (20%), Financial last (10%) so it appears after warm-up.
PILLAR_ORDER = ["Diet & Workout", "Lifestyle", "Financial"]


# ── Dynamic template computation ───────────────────────────────────────────

def _round_to_sum(floats, target):
    """Floor-then-largest-remainder rounding that guarantees sum == target."""
    floors = [math.floor(x) for x in floats]
    deficit = target - sum(floors)
    fracs = sorted(range(len(floats)), key=lambda i: floats[i] - floors[i], reverse=True)
    for i in range(deficit):
        floors[fracs[i]] += 1
    return floors


def _compute_day_templates(T, M, B, n_days=10):
    """
    Compute a per-day slot sequence for each of the n_days content-day groups.
    Each day has exactly 10 slots. Totals across all days equal T, M, B.

    Strategy:
      - TOFU is front-loaded (more slots in early days, fewer in late days)
      - BOFU is back-loaded (fewer slots in early days, more in late days)
      - MOFU fills the remainder, biased toward middle days
      - Within each day the slot order is: TOFU first, then MOFU/BOFU interleaved
        with BOFU at the very end so the arc holds
    """
    # Ramp factor: 0.0 = no ramp, 0.5 = large swing. Applied as fraction of 10.
    # TOFU ramp: day 1 gets +ramp_T extra, day 10 gets -ramp_T extra
    # BOFU ramp: day 1 gets -ramp_B extra, day 10 gets +ramp_B extra
    ramp_T = min(0.8, (T / 100) * 3)   # proportional to batch share
    ramp_B = min(0.8, (B / 100) * 3)

    T_float = []
    B_float = []
    for d in range(n_days):
        alpha = d / (n_days - 1)  # 0.0 on day 1, 1.0 on day 10
        T_float.append(10 * (T / 100) - ramp_T + alpha * 2 * ramp_T)
        B_float.append(10 * (B / 100) - ramp_B + alpha * 2 * ramp_B)

    # Invert ramp for TOFU (more early) and BOFU (more late)
    T_float = [10 * (T / 100) + ramp_T - alpha * 2 * ramp_T
               for alpha, _ in [(d / (n_days - 1), d) for d in range(n_days)]]
    B_float = [10 * (B / 100) - ramp_B + (d / (n_days - 1)) * 2 * ramp_B
               for d in range(n_days)]

    T_alloc = _round_to_sum(T_float, T)
    B_alloc = _round_to_sum(B_float, B)
    M_alloc = [10 - T_alloc[d] - B_alloc[d] for d in range(n_days)]

    # Guard: if any M < 0, trim T/B proportionally for that day
    for d in range(n_days):
        if M_alloc[d] < 0:
            excess = -M_alloc[d]
            # Trim from whichever is larger (T or B) to stay closest to arc intent
            if T_alloc[d] >= B_alloc[d]:
                T_alloc[d] -= excess
            else:
                B_alloc[d] -= excess
            M_alloc[d] = 0

    # Build the slot sequence for each day: TOFU first, then MOFU/BOFU interleaved
    # ending with BOFU slots so closing positions are always conversion content.
    templates = []
    for d in range(n_days):
        t, m, b = T_alloc[d], M_alloc[d], B_alloc[d]
        # Interleave MOFU and BOFU in the middle+end block, BOFU biased to end.
        # Pattern: MOFU MOFU BOFU MOFU BOFU BOFU ... (MOFU before each BOFU pair)
        mb = []
        m_left, b_left = m, b
        while m_left > 0 or b_left > 0:
            # Place MOFU first if available, then BOFU
            if m_left > 0:
                mb.append("MOFU")
                m_left -= 1
            if b_left > 0:
                mb.append("BOFU")
                b_left -= 1
        templates.append(["TOFU"] * t + mb)

    return templates, T_alloc, M_alloc, B_alloc

# ── Keyword sets for pillar inference (fallback if topics.json missing) ────
_FINANCIAL_KEYWORDS = {
    "emergency fund (3\u20136 months)", "term life insurance (pure protection)",
    "health insurance top-up / super top-up", "sip (systematic investment plan)",
    "index funds (nifty 50 / sensex)", "epf / ppf contribution strategy",
    "debt snowball (loan payoff plan)", "credit score improvement",
    "budgeting (50/30/20 rule)", "retirement corpus planning",
}
_LIFESTYLE_KEYWORDS = {
    "sleep quality tips for parents", "deep sleep optimization hacks",
    "how to workout with little sleep", "stress management for working men",
    "nervous system regulation for men", "corporate burnout recovery",
    "men's mental health and fitness", "5 am morning routine for dads",
    "atomic habits for fitness", "sustainable fitness routines",
    "work-life balance for indian fathers", "juggling corporate jobs and family",
    "making time for exercise with kids", "time management for working parents",
    "active weekend hobbies with family", "overcoming daily fatigue and low energy",
}


def _infer_pillar_from_keyword(keyword):
    kw = keyword.strip().lower()
    if kw in _FINANCIAL_KEYWORDS:
        return "Financial"
    if kw in _LIFESTYLE_KEYWORDS:
        return "Lifestyle"
    return "Diet & Workout"


def _load_topics_lookup(batch_no):
    path = os.path.join(
        os.path.dirname(__file__), "..", "data",
        "batch_%d" % batch_no, "topics.json",
    )
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {t["title"].strip().lower(): t for t in json.load(f)}


def _enrich_pillars(carousels, batch_no):
    """Add 'pillar' key to each carousel dict using topics.json (primary) or keyword (fallback)."""
    lookup = _load_topics_lookup(batch_no)
    for c in carousels:
        entry = lookup.get(c["title"].strip().lower())
        if entry:
            c["pillar"] = entry.get("category", "Diet & Workout")
        else:
            c["pillar"] = _infer_pillar_from_keyword(c["keyword"])
    return carousels


def _pick(stage, buckets, cycles):
    """
    Pick next carousel for `stage`, cycling through pillars to interleave variety.
    Advances the pillar cycle to avoid same-pillar runs within the same funnel stage.
    Returns None if the stage is exhausted.
    """
    start = cycles[stage]
    for offset in range(len(PILLAR_ORDER)):
        idx = (start + offset) % len(PILLAR_ORDER)
        pillar = PILLAR_ORDER[idx]
        key = (stage, pillar)
        if buckets.get(key):
            item = buckets[key].pop(0)
            cycles[stage] = (idx + 1) % len(PILLAR_ORDER)
            return item
    return None


def build_order(carousels, batch_no):
    """
    Return carousels in posting order with 'running_no' assigned 1-100.
    """
    carousels = _enrich_pillars(carousels, batch_no)

    # Count actual funnel distribution
    counts = {"TOFU": 0, "MOFU": 0, "BOFU": 0}
    for c in carousels:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    T, M, B = counts["TOFU"], counts["MOFU"], counts["BOFU"]

    # Compute templates dynamically from actual counts
    templates, T_alloc, M_alloc, B_alloc = _compute_day_templates(T, M, B)
    print("  Template allocation (T/M/B per content-day group):")
    for d, (t, m, b) in enumerate(zip(T_alloc, M_alloc, B_alloc)):
        print("    Day %2d: %dT %dM %dB" % (d + 1, t, m, b))
    print("  Totals: %dT %dM %dB (batch has %dT %dM %dB)" % (
        sum(T_alloc), sum(M_alloc), sum(B_alloc), T, M, B))

    # Build buckets {(funnel_stage, pillar): [ordered list of carousels]}
    buckets = {}
    for c in carousels:
        key = (c["category"], c["pillar"])
        buckets.setdefault(key, []).append(c)

    # Pillar cycle position per funnel stage (persists across day-groups)
    cycles = {s: 0 for s in FUNNEL_ORDER}

    ordered = []
    for template in templates:
        for slot_stage in template:
            item = _pick(slot_stage, buckets, cycles)
            if item is None:
                # Stage exhausted — use stage-aware fallback priority to preserve arc:
                # BOFU slot → prefer MOFU over TOFU (don't open close-of-day with TOFU)
                # TOFU slot → prefer MOFU over BOFU (don't fill open-of-day with conversion)
                for fallback in _FALLBACK[slot_stage]:
                    item = _pick(fallback, buckets, cycles)
                    if item:
                        break
            if item:
                ordered.append(item)

    # Assign running_no
    for i, c in enumerate(ordered):
        c["running_no"] = i + 1

    return ordered


# ── Output ─────────────────────────────────────────────────────────────────

def _print_summary(ordered, batch_no):
    sep = "-" * 76
    print("\n%s\n  Posting Order — Batch %d\n%s" % (sep, batch_no, sep))
    print("  %-5s  %-4s  %-6s  %-16s  %s" % ("Day", "#", "Stage", "Pillar", "Title"))
    print(sep)
    for c in ordered:
        rn = c["running_no"]
        day = ((rn - 1) // 10) + 1
        print("  D%-4d  %-4d  %-6s  %-16s  %s" % (
            day, rn, c["category"], c.get("pillar", "?")[:16], c["title"][:52],
        ))
    print(sep)

    print("\n  Per-day distribution:")
    print("  %-5s  %-4s  %-4s  %-4s  |  %-14s  %-10s  %-10s" % (
        "Day", "TOF", "MOF", "BOF", "Diet&Workout", "Lifestyle", "Financial",
    ))
    print("  " + "-" * 68)
    for day in range(1, 11):
        items = [c for c in ordered if ((c["running_no"] - 1) // 10) + 1 == day]
        t = sum(1 for x in items if x["category"] == "TOFU")
        m = sum(1 for x in items if x["category"] == "MOFU")
        b = sum(1 for x in items if x["category"] == "BOFU")
        dw = sum(1 for x in items if x.get("pillar") == "Diet & Workout")
        ls = sum(1 for x in items if x.get("pillar") == "Lifestyle")
        fi = sum(1 for x in items if x.get("pillar") == "Financial")
        print("  D%-4d  %-4d  %-4d  %-4d  |  %-14d  %-10d  %-10d" % (
            day, t, m, b, dw, ls, fi,
        ))
    print()


# ── Main ───────────────────────────────────────────────────────────────────

def run(batch_no, dry_run):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT uuid, title, keyword, category FROM Carousel "
        "WHERE batch_no = ? AND current_stage = 'CATEGORIZED' ORDER BY rowid",
        (batch_no,),
    ).fetchall()
    rows = [dict(r) for r in rows]
    conn.close()

    print("\n  Loaded %d CATEGORIZED carousels for batch %d" % (len(rows), batch_no))
    if len(rows) != 100:
        print("  ERROR: Expected 100 CATEGORIZED carousels, got %d. Aborting." % len(rows))
        return

    ordered = build_order(rows, batch_no)
    _print_summary(ordered, batch_no)

    if dry_run:
        print("  DRY RUN — no changes made.\n")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("BEGIN")
        for item in ordered:
            conn.execute(
                "UPDATE Carousel SET running_no = ?, current_stage = 'ORDER_SET' "
                "WHERE uuid = ? AND batch_no = ?",
                (item["running_no"], item["uuid"], batch_no),
            )
        conn.commit()
        print("  Updated %d rows to ORDER_SET for batch %d.\n" % (len(ordered), batch_no))
    except Exception as exc:
        conn.rollback()
        print("  ERROR: %s" % exc)
        raise
    finally:
        conn.close()

    # Verify
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute(
        "SELECT COUNT(*) FROM Carousel WHERE batch_no = ? AND current_stage = 'ORDER_SET'",
        (batch_no,),
    ).fetchone()[0]
    conn.close()
    print("  Verified: %d rows with current_stage = ORDER_SET\n" % n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, required=True, help="Batch number")
    p.add_argument("--dry-run", action="store_true", help="Preview order without writing to DB")
    args = p.parse_args()
    run(args.batch, args.dry_run)


if __name__ == "__main__":
    main()
