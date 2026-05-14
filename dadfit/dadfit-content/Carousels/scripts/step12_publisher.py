#!/usr/bin/env python3
"""
step12_publisher.py — Instagram Carousel Publisher

Modes:
  dry-run                List next unpublished IMAGES_CREATED carousels (no API calls)
  publish --uuids ...    Publish specific carousels to Instagram as carousel posts

Config file (Carousels/data/publish_config.env):
  IG_USER_ID=<your-instagram-professional-account-id>
  IG_ACCESS_TOKEN=<your-long-lived-access-token>
  IG_API_VERSION=v25.0    (optional, default: v25.0)

Usage:
  python3 Carousels/scripts/step12_publisher.py dry-run [--batch N] [--count 3]
  python3 Carousels/scripts/step12_publisher.py publish --uuids uuid1,uuid2,uuid3 [--batch N]
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────

WORKSPACE   = Path(__file__).resolve().parents[2]   # dadfit-content/
DB_PATH     = WORKSPACE / "Carousels" / "data" / "db.sqlite"
CONFIG_PATH = WORKSPACE / "Carousels" / "data" / "publish_config.env"
GRAPH_HOST  = "https://graph.instagram.com"

PUBLISHABLE_STAGES = ("IMAGES_CREATED",)

# ── Config ─────────────────────────────────────────────────────────────────────

def load_config():
    if not CONFIG_PATH.exists():
        sys.exit(
            f"\nConfig file not found. Create it at:\n  {CONFIG_PATH}\n\n"
            "Contents:\n"
            "  IG_USER_ID=<your-instagram-professional-account-id>\n"
            "  IG_ACCESS_TOKEN=<your-long-lived-access-token>\n"
            "  IG_API_VERSION=v25.0\n"
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

def get_latest_batch(conn):
    row = conn.execute("SELECT MAX(batch_no) FROM Carousel").fetchone()
    return row[0] if row and row[0] is not None else None


def get_top_unpublished(conn, batch_no, count):
    rows = conn.execute(
        """
        SELECT uuid, running_no, title, category, caption, hook, cta, folder_name
        FROM   Carousel
        WHERE  batch_no = ?
          AND  current_stage IN ('IMAGES_CREATED')
          AND  upload_status  = 'PENDING'
        ORDER BY running_no ASC
        LIMIT ?
        """,
        (batch_no, count),
    ).fetchall()
    cols = ["uuid", "running_no", "title", "category", "caption", "hook", "cta", "folder_name"]
    return [dict(zip(cols, r)) for r in rows]


def get_carousel_by_uuid(conn, uuid):
    row = conn.execute(
        """
        SELECT uuid, running_no, title, category, caption, hook, cta, folder_name, batch_no
        FROM   Carousel
        WHERE  uuid = ?
        """,
        (uuid,),
    ).fetchone()
    if not row:
        return None
    cols = ["uuid", "running_no", "title", "category", "caption", "hook", "cta", "folder_name", "batch_no"]
    return dict(zip(cols, row))


def mark_published(conn, uuid):
    conn.execute(
        "UPDATE Carousel SET upload_status = 'PUBLISHED', current_stage = 'PUBLISHED' WHERE uuid = ?",
        (uuid,),
    )
    conn.commit()


# ── Slide helpers ──────────────────────────────────────────────────────────────

def get_slide_paths(batch_no, folder_name):
    slides_dir = WORKSPACE / "Carousels" / "data" / f"batch_{batch_no}_slides" / folder_name
    if not slides_dir.exists():
        return []
    return sorted(slides_dir.glob("slide-*.png"))


def convert_png_to_jpeg(png_path: Path) -> Path:
    """Convert PNG → JPEG (RGB). Returns path to .jpg file."""
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow not installed. Activate venv: source Carousels/.venv/bin/activate")

    jpg_path = png_path.with_suffix(".jpg")
    if not jpg_path.exists() or jpg_path.stat().st_mtime < png_path.stat().st_mtime:
        img = Image.open(png_path).convert("RGB")
        img.save(jpg_path, "JPEG", quality=95)
    return jpg_path


# ── Temp image hosting ─────────────────────────────────────────────────────────

def upload_to_catbox(jpeg_path: Path) -> str:
    """Upload a JPEG to catbox.moe (anonymous, free). Returns a permanent public URL."""
    with open(jpeg_path, "rb") as f:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": (jpeg_path.name, f, "image/jpeg")},
            timeout=60,
        )
    resp.raise_for_status()
    url = resp.text.strip()
    if not url.startswith("https://"):
        sys.exit(f"catbox.moe upload failed: {url}")
    return url


# ── Instagram Graph API calls ──────────────────────────────────────────────────

def _api_post(url, payload, timeout=30):
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        data = resp.json()
    except Exception:
        data = {}
    if not resp.ok:
        error = data.get("error", {})
        sys.exit(
            f"\nInstagram API error {resp.status_code}: "
            f"{error.get('message', resp.text)}\n"
            f"  Code: {error.get('code')}  Subcode: {error.get('error_subcode')}"
        )
    return data


def create_item_container(ig_user_id, access_token, api_version, image_url):
    """Upload a single carousel slide as a container. Returns container ID."""
    url = f"{GRAPH_HOST}/{api_version}/{ig_user_id}/media"
    data = _api_post(url, {
        "image_url":       image_url,
        "is_carousel_item": "true",
        "access_token":    access_token,
    })
    return data["id"]


def poll_container_status(access_token, api_version, container_id, max_polls=12, interval=5):
    """Poll /<container_id>?fields=status_code until FINISHED. Returns True/False."""
    url = f"{GRAPH_HOST}/{api_version}/{container_id}"
    for attempt in range(1, max_polls + 1):
        resp = requests.get(url, params={"fields": "status_code", "access_token": access_token}, timeout=15)
        if not resp.ok:
            return False
        status = resp.json().get("status_code", "")
        print(f"    [{attempt}/{max_polls}] status_code = {status}")
        if status == "FINISHED":
            return True
        if status == "ERROR":
            return False
        time.sleep(interval)
    return False


def create_carousel_container(ig_user_id, access_token, api_version, children_ids, caption):
    """Create the top-level carousel container. Returns container ID."""
    url = f"{GRAPH_HOST}/{api_version}/{ig_user_id}/media"
    data = _api_post(url, {
        "media_type":   "CAROUSEL",
        "children":     ",".join(children_ids),
        "caption":      caption,
        "access_token": access_token,
    })
    return data["id"]


def publish_container(ig_user_id, access_token, api_version, creation_id):
    """Publish a carousel container. Returns the published media ID."""
    url = f"{GRAPH_HOST}/{api_version}/{ig_user_id}/media_publish"
    data = _api_post(url, {
        "creation_id":  creation_id,
        "access_token": access_token,
    })
    return data["id"]


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_dry_run(args):
    if not DB_PATH.exists():
        sys.exit(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    batch_no = args.batch or get_latest_batch(conn)
    if batch_no is None:
        conn.close()
        sys.exit("No batches found in DB.")

    carousels = get_top_unpublished(conn, batch_no, args.count)
    conn.close()

    if not carousels:
        print(f"\nNo unpublished IMAGES_CREATED carousels in batch {batch_no}.")
        return

    print(f"\nBatch {batch_no} — Next {len(carousels)} unpublished carousel(s):")
    print("=" * 70)
    for c in carousels:
        slides = get_slide_paths(batch_no, c["folder_name"])
        caption = c["caption"] or ""
        preview = caption[:100] + ("…" if len(caption) > 100 else "")
        print(f"\n  #{c['running_no']:>3}  {c['title']}  [{c['category']}]")
        print(f"       UUID     : {c['uuid']}")
        print(f"       Folder   : {c['folder_name']}")
        print(f"       Slides   : {len(slides)} PNG(s) found")
        print(f"       Caption  : {preview}")
    print("\n" + "=" * 70)
    print("\nTo publish these, run:")
    uuids_str = ",".join(c["uuid"] for c in carousels)
    print(f"  python3 Carousels/scripts/step12_publisher.py publish --uuids {uuids_str}\n")


def cmd_publish(args):
    cfg   = load_config()
    ig_id = cfg["IG_USER_ID"]
    token = cfg["IG_ACCESS_TOKEN"]
    ver   = cfg["IG_API_VERSION"]

    uuids = [u.strip() for u in args.uuids.split(",") if u.strip()]
    if not uuids:
        sys.exit("No UUIDs provided via --uuids.")

    if not DB_PATH.exists():
        sys.exit(f"Database not found: {DB_PATH}")

    conn       = sqlite3.connect(DB_PATH)
    batch_no   = args.batch or get_latest_batch(conn)
    carousels  = []

    for uuid in uuids:
        c = get_carousel_by_uuid(conn, uuid)
        if not c:
            print(f"  WARNING: UUID {uuid} not found in DB — skipping.")
            continue
        slides = get_slide_paths(c["batch_no"], c["folder_name"])
        if not slides:
            print(f"  WARNING: No slides found for #{c['running_no']} ({c['folder_name']}) — skipping.")
            continue
        carousels.append((c, slides))

    if not carousels:
        conn.close()
        sys.exit("No publishable carousels found.")

    results = []

    try:
        for c, slide_paths in carousels:
            print(f"▶  Publishing #{c['running_no']} — {c['title']}")

            # Instagram carousel max = 10 slides
            slide_paths = slide_paths[:10]

            # Convert PNGs → JPEG
            print(f"   Converting {len(slide_paths)} slide(s) to JPEG …")
            jpeg_paths = [convert_png_to_jpeg(p) for p in slide_paths]

            # Upload each JPEG to catbox.moe to get public URLs
            print(f"   Uploading {len(jpeg_paths)} image(s) to catbox.moe …")
            slide_urls = []
            for i, jp in enumerate(jpeg_paths, 1):
                url = upload_to_catbox(jp)
                print(f"     Slide {i:02d}: {url}")
                slide_urls.append(url)

            # Create item containers
            print(f"   Creating {len(slide_urls)} item container(s) …")
            item_ids = []
            for i, img_url in enumerate(slide_urls, 1):
                cid = create_item_container(ig_id, token, ver, img_url)
                print(f"     Slide {i:02d}: container {cid}")
                item_ids.append(cid)

            # Poll each item container
            print("   Polling item containers …")
            for cid in item_ids:
                ok = poll_container_status(token, ver, cid)
                if not ok:
                    print(f"   ✗  Container {cid} did not reach FINISHED. Aborting this carousel.")
                    break
            else:
                # Create carousel container
                caption = c["caption"] or f"{c['hook']}\n\n{c['cta']}"
                carousel_cid = create_carousel_container(ig_id, token, ver, item_ids, caption)
                print(f"   Carousel container: {carousel_cid}")

                # Poll carousel container
                print("   Polling carousel container …")
                ok = poll_container_status(token, ver, carousel_cid, max_polls=12, interval=5)
                if not ok:
                    print(f"   ✗  Carousel container {carousel_cid} did not reach FINISHED.")
                    continue

                # Publish
                media_id = publish_container(ig_id, token, ver, carousel_cid)
                print(f"   ✓  Published! Instagram Media ID: {media_id}")

                # Update DB
                mark_published(conn, c["uuid"])
                print(f"   ✓  DB updated → upload_status=PUBLISHED, current_stage=PUBLISHED")
                results.append((c["running_no"], c["title"], media_id))

    finally:
        conn.close()

    # Summary
    print("\n" + "=" * 70)
    print(f"Published {len(results)} carousel(s):")
    for running_no, title, media_id in results:
        print(f"  #{running_no:>3}  {title}  → media_id {media_id}")
    print("=" * 70)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Instagram Carousel Publisher for DadFit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch number (auto-detects latest if omitted)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # dry-run
    dr = sub.add_parser("dry-run", help="Preview next unpublished carousels (no API calls)")
    dr.add_argument("--count", type=int, default=3, help="How many to preview (default: 3)")

    # publish
    pub = sub.add_parser("publish", help="Publish specific carousels to Instagram")
    pub.add_argument("--uuids", required=True, help="Comma-separated list of carousel UUIDs")

    args = parser.parse_args()

    if args.command == "dry-run":
        cmd_dry_run(args)
    elif args.command == "publish":
        cmd_publish(args)


if __name__ == "__main__":
    main()
