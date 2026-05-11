"""
orchestrator.py
Manages pipeline state for the Carousel batch system.

Usage:
  python scripts/orchestrator.py status [--batch BATCH_NO]
  python scripts/orchestrator.py next   [--batch BATCH_NO]
  python scripts/orchestrator.py stuck  [--batch BATCH_NO]

Commands:
  status  — Show per-stage counts for the batch
  next    — Show exactly which step to run next and how many carousels are pending it
  stuck   — List carousels that have not moved stage in a suspicious way (DOODLES_DONE blockers)
"""

import sqlite3
import os
import argparse
from collections import Counter

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "db.sqlite")

STAGE_ORDER = [
    "TOPIC_FETCHED",
    "CATEGORIZED",
    "ORDER_SET",
    "HOOK_WRITTEN",
    "SCRIPT_WRITTEN",
    "CTA_WRITTEN",
    "CAPTION_WRITTEN",
    "HTML_CREATED",
    "DOODLES_DONE",
    "HTML_APPROVED",
    "IMAGES_CREATED",
    "MUSIC_CHOSEN",
    "READY_TO_PUBLISH",
    "PUBLISHED",
    "MONITORED",
]

STAGE_TO_STEP = {
    "TOPIC_FETCHED":    "Step 2  — Run categorizer skill (10 subagents)",
    "CATEGORIZED":      "Step 3  — Run order setter skill",
    "ORDER_SET":        "Step 4  — Run hook writer skill (10 agents)",
    "HOOK_WRITTEN":     "Step 5  — Run script writer skill (10 parallel agents)",
    "SCRIPT_WRITTEN":   "Step 6  — Run CTA writer skill",
    "CTA_WRITTEN":      "Step 7  — Run caption writer skill",
    "CAPTION_WRITTEN":  "Step 8  — Run HTML builder skill",
    "HTML_CREATED":     "Step 9  — MANUAL: Place doodles in doodles/ folder, then mark DOODLES_DONE via web viewer",
    "DOODLES_DONE":     "Step 8b — Open web viewer (carousel_viewer.py), review carousel + doodles, click Approve → HTML_APPROVED",
    "HTML_APPROVED":    "Step 10 — Run HTML-to-images skill (Puppeteer)",
    "IMAGES_CREATED":   "Step 11 — Run music chooser skill (10 agents)",
    "MUSIC_CHOSEN":     "→ All production done. Set current_stage = READY_TO_PUBLISH",
    "READY_TO_PUBLISH": "Step 12 — Run publish queue skill, upload manually",
    "PUBLISHED":        "Step 13 — Run daily monitor skill after upload day",
    "MONITORED":        "Step 14 — Run weekly analysis skill",
}


def get_conn():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run `python scripts/init_db.py` first."
        )
    return sqlite3.connect(DB_PATH)


def cmd_status(batch_no):
    conn = get_conn()
    c = conn.cursor()
    query = "SELECT current_stage, COUNT(*) FROM Carousel"
    params = []
    if batch_no:
        query += " WHERE batch_no = ?"
        params.append(batch_no)
    query += " GROUP BY current_stage"
    c.execute(query, params)
    rows = dict(c.fetchall())
    conn.close()

    label = f"Batch {batch_no}" if batch_no else "All batches"
    total = sum(rows.values())
    print(f"\n{'─' * 50}")
    print(f"  Pipeline Status — {label}  (total: {total})")
    print(f"{'─' * 50}")
    for stage in STAGE_ORDER:
        count = rows.get(stage, 0)
        bar = "█" * min(count, 40)
        print(f"  {stage:<22}  {count:>4}  {bar}")
    print(f"{'─' * 50}\n")


def cmd_next(batch_no):
    conn = get_conn()
    c = conn.cursor()
    query = "SELECT current_stage, COUNT(*) FROM Carousel"
    params = []
    if batch_no:
        query += " WHERE batch_no = ?"
        params.append(batch_no)
    query += " GROUP BY current_stage"
    c.execute(query, params)
    rows = dict(c.fetchall())
    conn.close()

    # Find the earliest stage that still has carousels pending
    for stage in STAGE_ORDER[:-1]:  # skip MONITORED — that's terminal
        count = rows.get(stage, 0)
        if count > 0:
            action = STAGE_TO_STEP.get(stage, "—")
            print(f"\n  ► Next action  : {action}")
            print(f"    Carousels    : {count} at stage '{stage}'")
            if batch_no:
                print(f"    Batch        : {batch_no}")
            print()
            return

    print("\n  ✓ No pending carousels found.\n")


def cmd_stuck(batch_no):
    conn = get_conn()
    c = conn.cursor()
    # Show carousels stuck at HTML_CREATED (need approval via web viewer)
    # or HTML_APPROVED (need doodles placed)
    for stuck_stage, hint in [
        ("HTML_CREATED",  "needs doodles placed in doodles/ folder, then marked DOODLES_DONE via web viewer"),
        ("DOODLES_DONE",  "needs review + approval in web viewer (Step 8b) → HTML_APPROVED"),
    ]:
        query = (
            "SELECT uuid, title, batch_no, folder_name FROM Carousel "
            f"WHERE current_stage = '{stuck_stage}'"
        )
        params = []
        if batch_no:
            query += " AND batch_no = ?"
            params.append(batch_no)
        c.execute(query, params)
        rows = c.fetchall()
        if rows:
            print(f"\n  \u26a0 {len(rows)} carousel(s) at {stuck_stage} — {hint}:\n")
            for uuid, title, batch, folder in rows[:10]:
                print(f"    [{batch}] {title}")
                print(f"           folder → data/batch_{batch}/{folder}/")
                print(f"           uuid   → {uuid}\n")
            if len(rows) > 10:
                print(f"    ... and {len(rows) - 10} more.\n")


def main():
    parser = argparse.ArgumentParser(description="Carousel pipeline orchestrator")
    parser.add_argument("command", choices=["status", "next", "stuck"])
    parser.add_argument("--batch", type=int, default=None, help="Filter by batch number")
    args = parser.parse_args()

    if args.command == "status":
        cmd_status(args.batch)
    elif args.command == "next":
        cmd_next(args.batch)
    elif args.command == "stuck":
        cmd_stuck(args.batch)


if __name__ == "__main__":
    main()
