#!/usr/bin/env python3
"""
step13_monitor.py — Daily Performance Monitor

Modes:
  list                            List all PUBLISHED carousels not yet monitored today
  log   --uuid <uuid> [metrics]   Manually log metrics for one carousel
  fetch [--batch N] [--force]     Auto-fetch metrics from Instagram Graph API for all PUBLISHED carousels

Metrics flags (for `log` mode):
  --views N --likes N --comments N --shares N --saves N
  --reach N --profile-visits N --follows N --notes "text"

Config file (Carousels/data/publish_config.env):
  IG_USER_ID=<id>
  IG_ACCESS_TOKEN=<token>
  IG_API_VERSION=v25.0    (optional)

Usage:
  python3 Carousels/scripts/step13_monitor.py list [--batch N]
  python3 Carousels/scripts/step13_monitor.py log --uuid <uuid> --views 1200 --likes 80 --saves 40
  python3 Carousels/scripts/step13_monitor.py fetch [--batch N] [--force]
"""

import argparse
import sqlite3
import sys
import uuid as uuid_lib
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────

WORKSPACE   = Path(__file__).resolve().parents[2]
DB_PATH     = WORKSPACE / "Carousels" / "data" / "db.sqlite"
CONFIG_PATH = WORKSPACE / "Carousels" / "data" / "publish_config.env"
GRAPH_HOST  = "https://graph.instagram.com"

# Metrics available from Instagram Media Insights API for carousel/feed posts
IG_METRICS = "impressions,reach,likes,comments,shares,saved,profile_visits,follows"

# ── Config ─────────────────────────────────────────────────────────────────────

def load_config():
    if not CONFIG_PATH.exists():
        sys.exit(
            f"\nConfig file not found: {CONFIG_PATH}\n"
            "Create it with: IG_USER_ID, IG_ACCESS_TOKEN, IG_API_VERSION=v25.0\n"
        )
    cfg = {}
    for line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    for required in ("IG_USER_ID", "IG_ACCESS_TOKEN"):
        if required not in cfg:
            sys.exit(f"Missing required config key: {required}")
    cfg.setdefault("IG_API_VERSION", "v25.0")
    return cfg


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_conn():
    if not DB_PATH.exists():
        sys.exit(f"Database not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def get_latest_batch(conn):
    row = conn.execute("SELECT MAX(batch_no) FROM Carousel").fetchone()
    return row[0] if row and row[0] is not None else None


MONITOR_WINDOW_DAYS = 15


def get_published_carousels(conn, batch_no):
    """
    Returns PUBLISHED carousels whose published_date is within the last
    MONITOR_WINDOW_DAYS days. Carousels with no published_date are included
    as a safety fallback.
    """
    rows = conn.execute(
        """
        SELECT uuid, running_no, title, category, instagram_post_id,
               published_date, last_performance_monitored
        FROM   Carousel
        WHERE  batch_no      = ?
          AND  upload_status = 'PUBLISHED'
          AND  (
                published_date IS NULL
             OR published_date >= datetime('now', ? || ' days')
          )
        ORDER BY running_no ASC
        """,
        (batch_no, f"-{MONITOR_WINDOW_DAYS}"),
    ).fetchall()
    cols = ["uuid", "running_no", "title", "category", "instagram_post_id",
            "published_date", "last_performance_monitored"]
    return [dict(zip(cols, r)) for r in rows]


def get_expired_carousels(conn, batch_no):
    """Returns PUBLISHED carousels whose monitoring window has passed."""
    rows = conn.execute(
        """
        SELECT uuid, running_no, title, category,
               published_date, last_performance_monitored
        FROM   Carousel
        WHERE  batch_no      = ?
          AND  upload_status = 'PUBLISHED'
          AND  published_date IS NOT NULL
          AND  published_date < datetime('now', ? || ' days')
        ORDER BY running_no ASC
        """,
        (batch_no, f"-{MONITOR_WINDOW_DAYS}"),
    ).fetchall()
    cols = ["uuid", "running_no", "title", "category",
            "published_date", "last_performance_monitored"]
    return [dict(zip(cols, r)) for r in rows]


def get_carousel_by_uuid(conn, carousel_uuid):
    row = conn.execute(
        "SELECT uuid, running_no, title, category, instagram_post_id FROM Carousel WHERE uuid = ?",
        (carousel_uuid,),
    ).fetchone()
    if not row:
        return None
    return dict(zip(["uuid", "running_no", "title", "category", "instagram_post_id"], row))


def already_monitored_today(conn, carousel_uuid):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        """
        SELECT 1 FROM CarouselPerformance
        WHERE carousel_uuid = ?
          AND performance_taken_time LIKE ?
        """,
        (carousel_uuid, f"{today}%"),
    ).fetchone()
    return row is not None


def insert_performance(conn, carousel_uuid, metrics: dict):
    perf_uuid = str(uuid_lib.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        """
        INSERT INTO CarouselPerformance
            (uuid, carousel_uuid, performance_taken_time,
             views, likes, comments, shares, saves,
             reach, profile_visits, follows_from_post, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            perf_uuid, carousel_uuid, now,
            metrics.get("views", 0),
            metrics.get("likes", 0),
            metrics.get("comments", 0),
            metrics.get("shares", 0),
            metrics.get("saves", 0),
            metrics.get("reach", 0),
            metrics.get("profile_visits", 0),
            metrics.get("follows", 0),
            metrics.get("notes"),
        ),
    )
    conn.execute(
        """
        UPDATE Carousel
        SET    last_performance_monitored = ?,
               current_stage             = 'MONITORED'
        WHERE  uuid = ?
        """,
        (now, carousel_uuid),
    )
    conn.commit()
    return perf_uuid, now


# ── Instagram Insights API ─────────────────────────────────────────────────────

def fetch_ig_insights(media_id: str, access_token: str, api_version: str) -> dict:
    """
    Fetch media insights from Instagram Graph API.
    Returns a dict with metric names mapped to values.
    Gracefully skips metrics not available for this media type.
    """
    url = f"{GRAPH_HOST}/{api_version}/{media_id}/insights"
    resp = requests.get(
        url,
        params={"metric": IG_METRICS, "access_token": access_token},
        timeout=15,
    )

    # Instagram returns 400 for unsupported metrics on some media types;
    # try a reduced set if that happens
    if resp.status_code == 400:
        reduced = "impressions,reach,likes,comments,shares,saved,profile_visits"
        resp = requests.get(
            url,
            params={"metric": reduced, "access_token": access_token},
            timeout=15,
        )

    if not resp.ok:
        error = resp.json().get("error", {})
        raise RuntimeError(
            f"Instagram API {resp.status_code}: {error.get('message', resp.text)}"
        )

    data = resp.json().get("data", [])
    # Map Instagram metric names → our DB column names
    name_map = {
        "impressions":    "views",
        "reach":          "reach",
        "likes":          "likes",
        "comments":       "comments",
        "shares":         "shares",
        "saved":          "saves",
        "profile_visits": "profile_visits",
        "follows":        "follows",
    }
    result = {}
    for item in data:
        our_name = name_map.get(item["name"])
        if our_name:
            result[our_name] = item.get("values", [{}])[0].get("value", item.get("value", 0))
    return result


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_list(args):
    conn = get_conn()
    batch_no = args.batch or get_latest_batch(conn)
    if batch_no is None:
        conn.close()
        sys.exit("No batches found in DB.")

    active  = get_published_carousels(conn, batch_no)
    expired = get_expired_carousels(conn, batch_no)
    conn.close()

    if not active and not expired:
        print(f"\nNo published carousels in batch {batch_no}.")
        return

    if active:
        print(f"\nBatch {batch_no} — Active monitoring window ({MONITOR_WINDOW_DAYS} days) — {len(active)} carousel(s):")
        print("=" * 80)
        for c in active:
            monitored = c["last_performance_monitored"] or "never"
            post_id   = c["instagram_post_id"] or "(no post ID)"
            pub_date  = c["published_date"] or "unknown"
            # Calculate days since publish
            days_info = ""
            if c["published_date"]:
                try:
                    from datetime import datetime, timezone
                    pub_str = c["published_date"].replace("Z", "").replace("T", " ").split(".")[0]
                    pub = datetime.fromisoformat(pub_str).replace(tzinfo=timezone.utc)
                    delta = (datetime.now(timezone.utc) - pub).days
                    remaining = MONITOR_WINDOW_DAYS - delta
                    days_info = f" (day {delta + 1}/{MONITOR_WINDOW_DAYS}, {remaining} day(s) left)"
                except Exception:
                    pass
            print(f"\n  #{c['running_no']:>3}  {c['title']}  [{c['category']}]")
            print(f"       UUID          : {c['uuid']}")
            print(f"       Post ID       : {post_id}")
            print(f"       Published     : {pub_date}{days_info}")
            print(f"       Last monitored: {monitored}")
        print("\n" + "=" * 80)

    if expired:
        print(f"\nExpired (outside {MONITOR_WINDOW_DAYS}-day window) — {len(expired)} carousel(s):")
        print("-" * 80)
        for c in expired:
            print(f"  #{c['running_no']:>3}  {c['title']}  — published {c['published_date']}  last monitored: {c['last_performance_monitored'] or 'never'}")
        print("-" * 80)

    if active:
        print("\nTo log metrics manually:")
        print("  python3 Carousels/scripts/step13_monitor.py log --uuid <uuid> --views N --likes N ...")
        print("\nTo auto-fetch from Instagram API:")
        print("  python3 Carousels/scripts/step13_monitor.py fetch\n")


def cmd_log(args):
    conn = get_conn()
    c = get_carousel_by_uuid(conn, args.uuid)
    if not c:
        conn.close()
        sys.exit(f"UUID {args.uuid} not found in DB.")

    if not args.force and already_monitored_today(conn, args.uuid):
        conn.close()
        print(f"  ⚠  #{c['running_no']} already has a performance entry for today. Use --force to override.")
        return

    metrics = {
        "views":          args.views,
        "likes":          args.likes,
        "comments":       args.comments,
        "shares":         args.shares,
        "saves":          args.saves,
        "reach":          args.reach,
        "profile_visits": args.profile_visits,
        "follows":        args.follows,
        "notes":          args.notes,
    }

    perf_uuid, ts = insert_performance(conn, args.uuid, metrics)
    conn.close()

    print(f"\n  ✓  #{c['running_no']} — {c['title']}")
    print(f"     Performance UUID : {perf_uuid}")
    print(f"     Recorded at      : {ts}")
    print(f"     Metrics          : views={metrics['views']} likes={metrics['likes']} "
          f"comments={metrics['comments']} shares={metrics['shares']} saves={metrics['saves']} "
          f"reach={metrics['reach']} profile_visits={metrics['profile_visits']} follows={metrics['follows']}")
    print(f"     DB updated       : current_stage=MONITORED, last_performance_monitored={ts}\n")


def cmd_fetch(args):
    cfg   = load_config()
    token = cfg["IG_ACCESS_TOKEN"]
    ver   = cfg["IG_API_VERSION"]

    conn     = get_conn()
    batch_no = args.batch or get_latest_batch(conn)
    if batch_no is None:
        conn.close()
        sys.exit("No batches found in DB.")

    carousels = get_published_carousels(conn, batch_no)   # already filtered to 15-day window
    expired   = get_expired_carousels(conn, batch_no)
    if expired:
        print(f"  ℹ  {len(expired)} carousel(s) outside the {MONITOR_WINDOW_DAYS}-day window — skipped automatically.")

    eligible  = [
        c for c in carousels
        if c["instagram_post_id"] and (args.force or not already_monitored_today(conn, c["uuid"]))
    ]

    skipped = len(carousels) - len(eligible)
    if skipped:
        print(f"  ℹ  Skipped {skipped} carousel(s) already monitored today (use --force to override).")

    if not eligible:
        conn.close()
        print("  Nothing to fetch.")
        return

    print(f"\nFetching insights for {len(eligible)} carousel(s) from Instagram (within {MONITOR_WINDOW_DAYS}-day window) …\n")
    success = 0
    failed  = 0

    for c in eligible:
        try:
            metrics = fetch_ig_insights(c["instagram_post_id"], token, ver)
            perf_uuid, ts = insert_performance(conn, c["uuid"], metrics)
            print(f"  ✓  #{c['running_no']:>3}  {c['title']}")
            print(f"       views={metrics.get('views',0)} reach={metrics.get('reach',0)} "
                  f"likes={metrics.get('likes',0)} saves={metrics.get('saves',0)} "
                  f"follows={metrics.get('follows',0)}")
            success += 1
        except Exception as e:
            print(f"  ✗  #{c['running_no']:>3}  {c['title']} — ERROR: {e}")
            failed += 1

    conn.close()
    print(f"\n{'='*60}")
    print(f"Done: {success} fetched, {failed} failed.")
    print(f"{'='*60}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Daily Performance Monitor for DadFit Carousels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=None,
                        help="Batch number (auto-detects latest if omitted)")
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all published carousels and their monitoring status")

    # log
    log_p = sub.add_parser("log", help="Manually log metrics for one carousel")
    log_p.add_argument("--uuid",           required=True)
    log_p.add_argument("--views",          type=int, default=0)
    log_p.add_argument("--likes",          type=int, default=0)
    log_p.add_argument("--comments",       type=int, default=0)
    log_p.add_argument("--shares",         type=int, default=0)
    log_p.add_argument("--saves",          type=int, default=0)
    log_p.add_argument("--reach",          type=int, default=0)
    log_p.add_argument("--profile-visits", type=int, default=0, dest="profile_visits")
    log_p.add_argument("--follows",        type=int, default=0)
    log_p.add_argument("--notes",          type=str, default=None)
    log_p.add_argument("--force",          action="store_true",
                       help="Allow logging even if already monitored today")

    # fetch
    fetch_p = sub.add_parser("fetch", help="Auto-fetch metrics from Instagram Graph API")
    fetch_p.add_argument("--force", action="store_true",
                         help="Re-fetch even for carousels already monitored today")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "fetch":
        cmd_fetch(args)


if __name__ == "__main__":
    main()
