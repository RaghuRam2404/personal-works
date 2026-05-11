"""
reset_batch.py
Resets or deletes carousel rows in a batch.

Usage:
  python3 scripts/reset_batch.py --batch 1 --to TOPIC_FETCHED
  python3 scripts/reset_batch.py --batch 1 --to HOOK_WRITTEN --dry-run
  python3 scripts/reset_batch.py --batch 1 --to CATEGORIZED --from SCRIPT_WRITTEN
  python3 scripts/reset_batch.py --batch 1 --uuid abc-123 --to HOOK_WRITTEN
  python3 scripts/reset_batch.py --batch 1 --delete
  python3 scripts/reset_batch.py --batch 1 --delete --dry-run

Options:
  --batch    Batch number (required)
  --to       Target stage to reset TO (required unless --delete)
  --from     Only reset rows currently AT this stage (optional)
  --uuid     Reset a single carousel by UUID instead of the whole batch
  --delete   Permanently DELETE all rows for this batch (cannot combine with --to)
  --dry-run  Preview what would change without making any changes
  --yes      Skip confirmation prompt
"""

import sqlite3
import os
import argparse

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
    "IMAGES_CREATED",
    "MUSIC_CHOSEN",
    "READY_TO_PUBLISH",
    "PUBLISHED",
    "MONITORED",
]

# Columns to clear when resetting to a given stage
# When you reset TO stage X, clear all columns written AFTER stage X
STAGE_CLEARS = {
    "TOPIC_FETCHED":    ["category", "running_no", "hook", "script_content", "cta", "caption", "folder_name", "upload_status"],
    "CATEGORIZED":      ["running_no", "hook", "script_content", "cta", "caption", "folder_name", "upload_status"],
    "ORDER_SET":        ["hook", "script_content", "cta", "caption", "folder_name", "upload_status"],
    "HOOK_WRITTEN":     ["script_content", "cta", "caption", "folder_name", "upload_status"],
    "SCRIPT_WRITTEN":   ["cta", "caption", "folder_name", "upload_status"],
    "CTA_WRITTEN":      ["caption", "folder_name", "upload_status"],
    "CAPTION_WRITTEN":  ["folder_name", "upload_status"],
    "HTML_CREATED":     ["upload_status"],
    "DOODLES_DONE":     ["upload_status"],
    "IMAGES_CREATED":   ["upload_status"],
    "MUSIC_CHOSEN":     ["upload_status"],
    "READY_TO_PUBLISH": ["upload_status"],
    "PUBLISHED":        [],
    "MONITORED":        [],
}


def get_conn():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found at {DB_PATH}. Run init_db.py first.")
    return sqlite3.connect(DB_PATH)


def stages_at_or_after(stage):
    idx = STAGE_ORDER.index(stage)
    return STAGE_ORDER[idx:]


def delete_batch(batch_no, dry_run, yes):
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM Carousel WHERE batch_no = ?", (batch_no,))
    count = c.fetchone()[0]

    if count == 0:
        print(f"\n  No rows found for batch {batch_no}.\n")
        conn.close()
        return

    print(f"\n{'─' * 60}")
    print(f"  DELETE — Batch {batch_no}")
    print(f"  Rows to delete: {count}")
    print(f"{'─' * 60}\n")

    if dry_run:
        print("  DRY RUN — no changes made.\n")
        conn.close()
        return

    if not yes:
        confirm = input(f"  Permanently DELETE all {count} rows for batch {batch_no}? Type YES to confirm: ").strip()
        if confirm != "YES":
            print("  Aborted.\n")
            conn.close()
            return

    try:
        # Also delete related performance rows
        c.execute(
            "DELETE FROM CarouselPerformance WHERE carousel_uuid IN "
            "(SELECT uuid FROM Carousel WHERE batch_no = ?)",
            (batch_no,)
        )
        perf_deleted = c.rowcount
        c.execute("DELETE FROM Carousel WHERE batch_no = ?", (batch_no,))
        carousel_deleted = c.rowcount
        conn.commit()
        print(f"  ✓ Deleted {carousel_deleted} carousel rows and {perf_deleted} performance rows for batch {batch_no}.\n")
    except Exception as e:
        conn.rollback()
        print(f"  ERROR: {e}")
        raise
    finally:
        conn.close()


def run(batch_no, to_stage, from_stage, single_uuid, dry_run, yes):
    if to_stage not in STAGE_ORDER:
        print(f"ERROR: '{to_stage}' is not a valid stage. Choose from:\n  " + "\n  ".join(STAGE_ORDER))
        return

    conn = get_conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Build the SELECT query to find rows to reset
    if single_uuid:
        c.execute(
            "SELECT uuid, title, current_stage FROM Carousel WHERE uuid = ? AND batch_no = ?",
            (single_uuid, batch_no)
        )
    elif from_stage:
        if from_stage not in STAGE_ORDER:
            print(f"ERROR: '{from_stage}' is not a valid stage.")
            conn.close()
            return
        c.execute(
            "SELECT uuid, title, current_stage FROM Carousel WHERE batch_no = ? AND current_stage = ?",
            (batch_no, from_stage)
        )
    else:
        # Reset all rows that are strictly AHEAD of to_stage
        stages_ahead = stages_at_or_after(to_stage)[1:]  # exclude to_stage itself
        if not stages_ahead:
            print(f"Nothing to reset — '{to_stage}' is already the furthest stage.")
            conn.close()
            return
        placeholders = ",".join("?" * len(stages_ahead))
        c.execute(
            f"SELECT uuid, title, current_stage FROM Carousel WHERE batch_no = ? AND current_stage IN ({placeholders})",
            [batch_no] + stages_ahead
        )

    rows = c.fetchall()

    if not rows:
        print(f"\n  No rows found matching the criteria for batch {batch_no}.\n")
        conn.close()
        return

    # Show preview
    cols_to_clear = STAGE_CLEARS.get(to_stage, [])
    print(f"\n{'─' * 60}")
    print(f"  Reset Preview — Batch {batch_no}")
    print(f"  Resetting TO: {to_stage}")
    print(f"  Rows affected: {len(rows)}")
    if cols_to_clear:
        print(f"  Columns to NULL: {', '.join(cols_to_clear)}")
    print(f"{'─' * 60}")
    for row in rows[:20]:
        print(f"  [{row['current_stage']:20}]  {row['title'][:55]}")
    if len(rows) > 20:
        print(f"  ... and {len(rows) - 20} more")
    print(f"{'─' * 60}\n")

    if dry_run:
        print("  DRY RUN — no changes made.\n")
        conn.close()
        return

    if not yes:
        confirm = input(f"  Reset {len(rows)} rows to '{to_stage}'? Type YES to confirm: ").strip()
        if confirm != "YES":
            print("  Aborted.\n")
            conn.close()
            return

    # Build the UPDATE
    uuids = [row["uuid"] for row in rows]
    set_clauses = ["current_stage = ?"]
    set_values = [to_stage]

    # Reset upload_status to PENDING unless resetting to/past PUBLISHED
    if to_stage not in ("PUBLISHED", "MONITORED"):
        set_clauses.append("upload_status = 'PENDING'")

    # NULL out columns that belong to stages after to_stage
    for col in cols_to_clear:
        if col != "upload_status":  # already handled above
            set_clauses.append(f"{col} = NULL")

    set_sql = ", ".join(set_clauses)
    placeholders = ",".join("?" * len(uuids))

    try:
        c.execute("BEGIN")
        c.execute(
            f"UPDATE Carousel SET {set_sql} WHERE uuid IN ({placeholders})",
            set_values + uuids
        )
        conn.commit()
        print(f"  ✓ Reset {c.rowcount} rows to '{to_stage}' in batch {batch_no}.\n")
    except Exception as e:
        conn.rollback()
        print(f"  ERROR: {e}")
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Reset carousel rows to a specific stage")
    parser.add_argument("--batch", type=int, required=True, help="Batch number")
    parser.add_argument("--to", dest="to_stage", default=None, help="Target stage to reset TO")
    parser.add_argument("--from", dest="from_stage", default=None, help="Only reset rows currently AT this stage")
    parser.add_argument("--uuid", dest="single_uuid", default=None, help="Reset a single carousel by UUID")
    parser.add_argument("--delete", action="store_true", help="Permanently delete all rows for this batch")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no changes")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    if args.delete:
        if args.to_stage:
            print("ERROR: Cannot use --delete and --to together.")
            return
        delete_batch(args.batch, args.dry_run, args.yes)
    else:
        if not args.to_stage:
            print("ERROR: --to is required unless using --delete.")
            return
        run(
            batch_no=args.batch,
            to_stage=args.to_stage,
            from_stage=args.from_stage,
            single_uuid=args.single_uuid,
            dry_run=args.dry_run,
            yes=args.yes,
        )


if __name__ == "__main__":
    main()
