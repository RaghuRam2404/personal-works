#!/usr/bin/env python3
"""
step14_analysis.py — Weekly Batch Analysis

Reads all CarouselPerformance rows for a batch and generates a Markdown report.

Report sections:
  1. Top 10 by saves
  2. Top 10 by reach
  3. Top 10 by follows_from_post
  4. TOFU vs MOFU vs BOFU engagement comparison (averages)
  5. Week-over-week growth per carousel (week 1 = days 1–7, week 2 = days 8–14, etc.)

Week bucketing is relative to each carousel's published_date.
The --week argument selects which report week to generate (default: auto-detect latest).

Usage:
  python3 Carousels/scripts/step14_analysis.py [--batch N] [--week W]

Output:
  Carousels/out/batch_{N}_week_{W}_report.md
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

WORKSPACE = Path(__file__).resolve().parents[2]
DB_PATH   = WORKSPACE / "Carousels" / "data" / "db.sqlite"
OUT_DIR   = WORKSPACE / "Carousels" / "out"

WEEK_DAYS = 7   # days per week bucket

# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_conn():
    if not DB_PATH.exists():
        sys.exit(f"Database not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def get_latest_batch(conn):
    row = conn.execute("SELECT MAX(batch_no) FROM Carousel").fetchone()
    return row[0] if row and row[0] is not None else None


def load_carousels(conn, batch_no) -> dict:
    """Returns {uuid: {running_no, title, category, published_date}} for PUBLISHED carousels."""
    rows = conn.execute(
        """
        SELECT uuid, running_no, title, category, published_date
        FROM   Carousel
        WHERE  batch_no = ? AND upload_status = 'PUBLISHED'
        ORDER  BY running_no ASC
        """,
        (batch_no,),
    ).fetchall()
    cols = ["uuid", "running_no", "title", "category", "published_date"]
    return {r[0]: dict(zip(cols[1:], r[1:])) for r in rows}


def load_performance(conn, batch_no) -> list[dict]:
    """Returns all CarouselPerformance rows for carousels in this batch."""
    rows = conn.execute(
        """
        SELECT cp.carousel_uuid, cp.performance_taken_time,
               cp.views, cp.likes, cp.comments, cp.shares,
               cp.saves, cp.reach, cp.profile_visits, cp.follows_from_post
        FROM   CarouselPerformance cp
        JOIN   Carousel c ON c.uuid = cp.carousel_uuid
        WHERE  c.batch_no = ?
        ORDER  BY cp.carousel_uuid, cp.performance_taken_time ASC
        """,
        (batch_no,),
    ).fetchall()
    cols = ["carousel_uuid", "performance_taken_time",
            "views", "likes", "comments", "shares",
            "saves", "reach", "profile_visits", "follows_from_post"]
    return [dict(zip(cols, r)) for r in rows]


# ── Analysis helpers ───────────────────────────────────────────────────────────

def parse_dt(s: str) -> datetime:
    """Parse a UTC datetime string in either 'YYYY-MM-DD HH:MM:SS' or ISO 8601 format."""
    s = s.strip().replace("Z", "").replace("T", " ").split(".")[0]
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def week_bucket(entry_dt: datetime, published_dt: datetime) -> int:
    """Return 1-based week number relative to published_date (week 1 = days 0–6, etc.)."""
    delta = (entry_dt - published_dt).days
    if delta < 0:
        return 0   # before publish — shouldn't happen
    return (delta // WEEK_DAYS) + 1


def aggregate_entries(entries: list[dict]) -> dict:
    """Sum all metric fields across a list of performance entries."""
    totals = dict(views=0, likes=0, comments=0, shares=0,
                  saves=0, reach=0, profile_visits=0, follows_from_post=0)
    for e in entries:
        for k in totals:
            totals[k] += e.get(k, 0) or 0
    return totals


def avg_metrics(grouped: dict[str, dict]) -> dict:
    """Average aggregated metrics across multiple carousels (keyed by uuid)."""
    if not grouped:
        return dict(views=0, likes=0, comments=0, shares=0,
                    saves=0, reach=0, profile_visits=0, follows_from_post=0)
    keys = ["views", "likes", "comments", "shares", "saves", "reach",
            "profile_visits", "follows_from_post"]
    n = len(grouped)
    return {k: round(sum(v[k] for v in grouped.values()) / n, 1) for k in keys}


def detect_latest_week(perf_rows: list[dict], carousels: dict) -> int:
    """Return the highest week bucket seen across all performance entries."""
    max_week = 1
    for row in perf_rows:
        c = carousels.get(row["carousel_uuid"])
        if not c or not c["published_date"]:
            continue
        pub_dt   = parse_dt(c["published_date"])
        entry_dt = parse_dt(row["performance_taken_time"])
        w = week_bucket(entry_dt, pub_dt)
        if w > max_week:
            max_week = w
    return max_week


# ── Report builder ─────────────────────────────────────────────────────────────

def build_report(batch_no: int, week_no: int, carousels: dict, perf_rows: list[dict]) -> str:
    """Generate the full Markdown report string."""

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Bucket performance entries by carousel + week ──────────────────────────
    # by_carousel[uuid][week] = [entry, ...]
    by_carousel: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for row in perf_rows:
        c = carousels.get(row["carousel_uuid"])
        if not c or not c["published_date"]:
            continue
        pub_dt   = parse_dt(c["published_date"])
        entry_dt = parse_dt(row["performance_taken_time"])
        w = week_bucket(entry_dt, pub_dt)
        by_carousel[row["carousel_uuid"]][w].append(row)

    # ── Aggregate ALL entries per carousel (for top-10 tables) ────────────────
    totals_by_carousel: dict[str, dict] = {}
    for uuid, weeks in by_carousel.items():
        all_entries = [e for entries in weeks.values() for e in entries]
        totals_by_carousel[uuid] = aggregate_entries(all_entries)

    # ── Top 10 helpers ─────────────────────────────────────────────────────────
    def top10_table(metric: str, label: str) -> str:
        ranked = sorted(
            totals_by_carousel.items(),
            key=lambda kv: kv[1].get(metric, 0),
            reverse=True,
        )[:10]
        if not ranked:
            return "_No data yet._\n"
        lines = [f"| # | Title | Category | {label} |",
                 "|---|---|---|---|"]
        for rank, (uuid, m) in enumerate(ranked, 1):
            c = carousels[uuid]
            lines.append(
                f"| {rank} | #{c['running_no']} {c['title']} | {c['category']} | {m.get(metric, 0):,} |"
            )
        return "\n".join(lines) + "\n"

    # ── TOFU / MOFU / BOFU split ───────────────────────────────────────────────
    by_category: dict[str, dict[str, dict]] = defaultdict(dict)
    for uuid, totals in totals_by_carousel.items():
        cat = carousels[uuid]["category"]
        by_category[cat][uuid] = totals

    def category_row(cat: str) -> str:
        avgs = avg_metrics(by_category.get(cat, {}))
        n    = len(by_category.get(cat, {}))
        return (
            f"| {cat} | {n} | {avgs['views']:,.1f} | {avgs['reach']:,.1f} | "
            f"{avgs['likes']:,.1f} | {avgs['saves']:,.1f} | "
            f"{avgs['profile_visits']:,.1f} | {avgs['follows_from_post']:,.1f} |"
        )

    cat_table = (
        "| Category | Count | Avg Views | Avg Reach | Avg Likes | Avg Saves | Avg Profile Visits | Avg Follows |\n"
        "|---|---|---|---|---|---|---|---|\n"
        + "\n".join(category_row(c) for c in ("TOFU", "MOFU", "BOFU"))
        + "\n"
    )

    # ── Week-over-week growth ──────────────────────────────────────────────────
    def wow_table() -> str:
        if week_no < 2:
            return "_Not enough data yet — week-over-week available from week 2 onwards._\n"

        rows_wow = []
        for uuid, weeks in by_carousel.items():
            c = carousels[uuid]
            prev_entries = weeks.get(week_no - 1, [])
            curr_entries = weeks.get(week_no, [])
            if not prev_entries and not curr_entries:
                continue
            prev = aggregate_entries(prev_entries)
            curr = aggregate_entries(curr_entries)

            def delta(key):
                p, c_ = prev.get(key, 0), curr.get(key, 0)
                if p == 0:
                    return "—"
                pct = ((c_ - p) / p) * 100
                arrow = "▲" if pct >= 0 else "▼"
                return f"{arrow} {abs(pct):.0f}%"

            rows_wow.append((
                c["running_no"],
                f"#{c['running_no']} {c['title']}",
                c["category"],
                curr.get("views", 0), delta("views"),
                curr.get("saves", 0), delta("saves"),
                curr.get("reach", 0), delta("reach"),
                curr.get("follows_from_post", 0), delta("follows_from_post"),
            ))

        if not rows_wow:
            return "_No week-over-week data available._\n"

        rows_wow.sort(key=lambda r: r[0])
        lines = [
            "| Title | Cat | Views (W{w}) | Δ | Saves (W{w}) | Δ | Reach (W{w}) | Δ | Follows (W{w}) | Δ |"
            .format(w=week_no),
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows_wow:
            lines.append(
                f"| {r[1]} | {r[2]} | {r[3]:,} | {r[4]} | {r[5]:,} | {r[6]} | {r[7]:,} | {r[8]} | {r[9]:,} | {r[10]} |"
            )
        return "\n".join(lines) + "\n"

    # ── Assemble report ────────────────────────────────────────────────────────
    report = f"""# Batch {batch_no} — Week {week_no} Performance Report

**Generated:** {now_str}  
**Carousels published:** {len(carousels)}  
**Performance entries:** {len(perf_rows)}  
**Week window:** {WEEK_DAYS} days per week bucket (relative to each carousel's publish date)

---

## Top 10 by Saves

{top10_table("saves", "Saves")}
---

## Top 10 by Reach

{top10_table("reach", "Reach")}
---

## Top 10 by Follows

{top10_table("follows_from_post", "Follows")}
---

## TOFU vs MOFU vs BOFU — Average Engagement

{cat_table}
---

## Week-over-Week Growth (Week {week_no - 1} → Week {week_no})

{wow_table()}
---

_Report saved to: `Carousels/out/batch_{batch_no}_week_{week_no}_report.md`_
"""
    return report


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Weekly Analysis Report for DadFit Carousel Batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch number (auto-detects latest if omitted)")
    parser.add_argument("--week", type=int, default=None,
                        help="Week number to report on (auto-detects latest if omitted)")
    args = parser.parse_args()

    conn     = get_conn()
    batch_no = args.batch or get_latest_batch(conn)
    if batch_no is None:
        conn.close()
        sys.exit("No batches found in DB.")

    carousels = load_carousels(conn, batch_no)
    perf_rows = load_performance(conn, batch_no)
    conn.close()

    if not carousels:
        sys.exit(f"No published carousels found for batch {batch_no}.")

    if not perf_rows:
        sys.exit(
            f"No CarouselPerformance rows found for batch {batch_no}.\n"
            "Run Step 13 (step13_monitor.py fetch/log) first to record metrics."
        )

    week_no = args.week or detect_latest_week(perf_rows, carousels)

    report = build_report(batch_no, week_no, carousels, perf_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"batch_{batch_no}_week_{week_no}_report.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"\n✓  Report generated: {out_path}")
    print(f"   Batch {batch_no} | Week {week_no} | {len(carousels)} carousel(s) | {len(perf_rows)} performance entry(ies)\n")


if __name__ == "__main__":
    main()
