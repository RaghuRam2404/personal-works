#!/usr/bin/env python3
"""
carousel_viewer.py — DadFit Carousel Web Viewer & Approver

Usage:
  python3 Carousels/scripts/carousel_viewer.py [--batch 1] [--port 8765]

Open http://localhost:8765 in your browser.

Stage flow managed by this tool:
  HTML_CREATED  →  [Mark Doodles Done]  →  DOODLES_DONE
  DOODLES_DONE  →  [Approve]            →  HTML_APPROVED
"""

import argparse
import json
import os
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # dadfit-content/
CAROUSELS    = PROJECT_ROOT / "Carousels"
DB_PATH      = CAROUSELS / "data" / "db.sqlite"

PRODUCTION_STAGES = {"HTML_CREATED", "DOODLES_DONE", "HTML_APPROVED"}


# ── DB helpers ─────────────────────────────────────────────────────────────────

def get_conn():
    return sqlite3.connect(DB_PATH)


def fetch_carousels(batch_no):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        """
        SELECT running_no, uuid, folder_name, title, category, current_stage
        FROM Carousel
        WHERE batch_no = ?
        ORDER BY running_no
        """,
        (batch_no,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def fetch_stage_counts(batch_no):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT current_stage, COUNT(*) FROM Carousel WHERE batch_no=? GROUP BY current_stage",
        (batch_no,),
    )
    counts = dict(c.fetchall())
    conn.close()
    return counts


def set_stage(uuid, new_stage, expected_stage):
    """Atomically transition uuid from expected_stage to new_stage."""
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "UPDATE Carousel SET current_stage=? WHERE uuid=? AND current_stage=?",
        (new_stage, uuid, expected_stage),
    )
    affected = c.rowcount
    conn.commit()
    conn.close()
    return affected == 1


def doodle_files_for(batch_no, running_no):
    """Return list of doodle image filenames for this carousel (from shared doodles folder)."""
    doodles_dir = CAROUSELS / "data" / f"batch_{batch_no}" / "doodles"
    if not doodles_dir.exists():
        return []
    prefix = f"{running_no}-d-"
    return sorted(f.name for f in doodles_dir.iterdir() if f.name.startswith(prefix))


# ── HTML rendering ─────────────────────────────────────────────────────────────

STAGE_BADGE_COLOR = {
    "HTML_CREATED":  "#888",
    "DOODLES_DONE":  "#e6a817",
    "HTML_APPROVED": "#34C363",
}
STAGE_BADGE_DEFAULT = "#555"

CATEGORY_COLOR = {"TOFU": "#4a9eff", "MOFU": "#e67e22", "BOFU": "#34C363"}

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', system-ui, sans-serif; background: #111; color: #eee; }
header { background: #1a1a1a; border-bottom: 2px solid #34C363;
         padding: 16px 28px; display: flex; align-items: center; gap: 20px; }
header h1 { font-size: 20px; color: #34C363; flex: 1; }
.progress-bar { display: flex; gap: 16px; font-size: 13px; }
.pb-item { background: #222; border-radius: 6px; padding: 6px 14px;
           border: 1px solid #333; }
.pb-item b { font-size: 18px; }
.pb-created  b { color: #888; }
.pb-doodles  b { color: #e6a817; }
.pb-approved b { color: #34C363; }

.search-bar { background:#1a1a1a; padding:12px 28px; border-bottom:1px solid #222; }
.search-bar input { background:#222; border:1px solid #333; border-radius:6px;
                    color:#eee; padding:7px 14px; width:320px; font-size:14px; }

.carousel-list { display: flex; flex-direction: column; gap: 0; }

.carousel-row { border-bottom: 1px solid #222; }
.carousel-row-header {
  display: flex; align-items: center; gap: 12px; padding: 14px 28px;
  cursor: pointer; user-select: none;
  transition: background 0.15s;
}
.carousel-row-header:hover { background: #1c1c1c; }
.carousel-row.open .carousel-row-header { background: #1c1c1c; }

.run-no { font-size: 13px; color: #555; width: 36px; flex-shrink: 0; }
.cat-badge { font-size: 11px; font-weight: 700; padding: 3px 8px;
             border-radius: 4px; color: #000; flex-shrink: 0; }
.stage-badge { font-size: 11px; font-weight: 600; padding: 3px 10px;
               border-radius: 10px; color: #fff; flex-shrink: 0; }
.title { flex: 1; font-size: 14px; color: #ccc; white-space: nowrap;
         overflow: hidden; text-overflow: ellipsis; }
.chevron { font-size: 12px; color: #555; flex-shrink: 0; transition: transform 0.2s; }
.carousel-row.open .chevron { transform: rotate(90deg); }

.action-btn {
  font-size: 12px; font-weight: 700; padding: 5px 16px; border-radius: 6px;
  border: none; cursor: pointer; flex-shrink: 0; transition: opacity 0.15s;
}
.action-btn:disabled { opacity: 0.25; cursor: default; }
.btn-doodles { background: #e6a817; color: #000; }
.btn-approve { background: #34C363; color: #000; }

.carousel-detail {
  display: none; padding: 0 28px 28px 28px; background: #151515;
  border-top: 1px solid #222;
}
.carousel-row.open .carousel-detail { display: flex; gap: 24px; }

.preview-col { flex: 0 0 440px; }
.preview-col iframe { width: 440px; height: 580px; border: none;
                       border-radius: 8px; background: #1e1e1e; }
.preview-label { font-size: 11px; color: #555; margin-bottom: 6px; text-transform: uppercase; }

.doodles-col { flex: 1; overflow-y: auto; max-height: 580px; }
.doodles-grid { display: flex; flex-wrap: wrap; gap: 10px; }
.doodles-grid img { width: 120px; height: 120px; object-fit: cover;
                     border-radius: 6px; border: 1px solid #333; background: #000; }
.no-doodles { color: #555; font-size: 13px; padding: 20px 0;
               font-style: italic; }
"""

JS = """
function toggleRow(runNo) {
  const row = document.getElementById('row-' + runNo);
  const wasOpen = row.classList.contains('open');
  // close all
  document.querySelectorAll('.carousel-row.open').forEach(r => r.classList.remove('open'));
  if (!wasOpen) row.classList.add('open');
}

function doAction(uuid, runNo, action) {
  fetch('/action', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({uuid, action})
  })
  .then(r => r.json())
  .then(data => {
    if (data.ok) {
      location.reload();
    } else {
      alert('Action failed: ' + (data.error || 'unknown error'));
    }
  });
}

function filterCarousels() {
  const q = document.getElementById('search').value.toLowerCase();
  document.querySelectorAll('.carousel-row').forEach(row => {
    const text = row.querySelector('.title').textContent.toLowerCase();
    row.style.display = (!q || text.includes(q)) ? '' : 'none';
  });
}
"""


def render_page(batch_no):
    carousels = fetch_carousels(batch_no)
    counts = fetch_stage_counts(batch_no)

    html_created  = counts.get("HTML_CREATED",  0)
    doodles_done  = counts.get("DOODLES_DONE",  0)
    html_approved = counts.get("HTML_APPROVED", 0)

    rows_html = []
    for running_no, uuid, folder_name, title, category, stage in carousels:
        doodles = doodle_files_for(batch_no, running_no)

        cat_color   = CATEGORY_COLOR.get(category, "#888")
        stage_color = STAGE_BADGE_COLOR.get(stage, STAGE_BADGE_DEFAULT)

        # Action buttons
        can_doodles = stage == "HTML_CREATED"
        can_approve = stage == "DOODLES_DONE"

        btn_doodles = (
            f'<button class="action-btn btn-doodles" '
            f'onclick="event.stopPropagation(); doAction(\'{uuid}\', {running_no}, \'doodles_done\')" '
            f'{"" if can_doodles else "disabled"}>✓ Doodles Done</button>'
        )
        btn_approve = (
            f'<button class="action-btn btn-approve" '
            f'onclick="event.stopPropagation(); doAction(\'{uuid}\', {running_no}, \'approve\')" '
            f'{"" if can_approve else "disabled"}>✓ Approve</button>'
        )

        # Doodle thumbnails
        if doodles:
            doodle_src_base = f"/doodle/{batch_no}/{running_no}"
            imgs = "".join(
                f'<img src="{doodle_src_base}/{name}" alt="{name}" title="{name}">'
                for name in doodles
            )
            doodles_content = f'<div class="doodles-grid">{imgs}</div>'
        else:
            doodles_content = '<p class="no-doodles">No doodle images placed yet.<br>Add files to batch_1/doodles/ named {running_no}-d-NN.png</p>'

        # iFrame src — served via /carousel/{batch}/{running_no}
        iframe_src = f"/carousel/{batch_no}/{running_no}"

        row = f"""
<div class="carousel-row" id="row-{running_no}">
  <div class="carousel-row-header" onclick="toggleRow({running_no})">
    <span class="run-no">#{running_no}</span>
    <span class="cat-badge" style="background:{cat_color}">{category}</span>
    <span class="title">{title}</span>
    <span class="stage-badge" style="background:{stage_color}">{stage}</span>
    {btn_doodles}
    {btn_approve}
    <span class="chevron">▶</span>
  </div>
  <div class="carousel-detail">
    <div class="preview-col">
      <div class="preview-label">Carousel Preview</div>
      <iframe src="{iframe_src}" loading="lazy"></iframe>
    </div>
    <div class="doodles-col">
      <div class="preview-label">Doodle Images ({len(doodles)} found)</div>
      {doodles_content}
    </div>
  </div>
</div>"""
        rows_html.append(row)

    total = len(carousels)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DadFit Carousel Viewer — Batch {batch_no}</title>
<style>{CSS}</style>
</head>
<body>
<header>
  <h1>🎠 DadFit Carousel Viewer — Batch {batch_no}</h1>
  <div class="progress-bar">
    <div class="pb-item pb-created">
      <b>{html_created}</b><br>HTML Created
    </div>
    <div class="pb-item pb-doodles">
      <b>{doodles_done}</b><br>Doodles Done
    </div>
    <div class="pb-item pb-approved">
      <b>{html_approved}</b><br>Approved
    </div>
    <div class="pb-item" style="color:#555">
      <b>{total}</b><br>Total
    </div>
  </div>
</header>

<div class="search-bar">
  <input id="search" type="text" placeholder="Filter by title…" oninput="filterCarousels()">
</div>

<div class="carousel-list">
{"".join(rows_html)}
</div>

<script>{JS}</script>
</body>
</html>"""


# ── HTTP handler ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    batch_no = 1

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html, status=200):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, mime):
        try:
            data = Path(path).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", len(data))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"

        # ── main page ──────────────────────────────────────────────────────────
        if path == "/":
            self.send_html(render_page(self.batch_no))

        # ── serve carousel.html inside an iframe ───────────────────────────────
        # GET /carousel/{batch}/{running_no}
        elif path.startswith("/carousel/"):
            parts = path.split("/")   # ['', 'carousel', batch, running_no]
            if len(parts) == 4:
                try:
                    batch   = int(parts[2])
                    run_no  = int(parts[3])
                    # find folder_name for this running_no
                    conn = get_conn()
                    c = conn.cursor()
                    c.execute(
                        "SELECT folder_name FROM Carousel WHERE batch_no=? AND running_no=?",
                        (batch, run_no),
                    )
                    row = c.fetchone()
                    conn.close()
                    if row and row[0]:
                        carousel_path = (
                            CAROUSELS / "data" / f"batch_{batch}" / row[0] / "carousel.html"
                        )
                        self.send_file(carousel_path, "text/html; charset=utf-8")
                    else:
                        self.send_html("<h1>Not found</h1>", 404)
                except (ValueError, IndexError):
                    self.send_html("<h1>Bad request</h1>", 400)
            else:
                self.send_html("<h1>Not found</h1>", 404)

        # ── serve doodle images ────────────────────────────────────────────────
        # GET /doodle/{batch}/{running_no}/{filename}
        elif path.startswith("/doodle/"):
            parts = path.split("/")   # ['', 'doodle', batch, running_no, filename]
            if len(parts) == 5:
                try:
                    batch    = int(parts[2])
                    filename = parts[4]
                    img_path = CAROUSELS / "data" / f"batch_{batch}" / "doodles" / filename
                    ext      = Path(filename).suffix.lower()
                    mime     = {"png": "image/png", "jpg": "image/jpeg",
                                "jpeg": "image/jpeg", "webp": "image/webp"}.get(ext[1:], "image/png")
                    self.send_file(img_path, mime)
                except (ValueError, IndexError):
                    self.send_response(400); self.end_headers()
            else:
                self.send_response(404); self.end_headers()

        # ── static assets referenced by carousel.html (fonts, images, CSS) ────
        # Any path that escapes through the iframe that references project files
        elif path.startswith("/static/"):
            # /static/ → PROJECT_ROOT
            rel = path[len("/static/"):]
            self.send_file(PROJECT_ROOT / rel, self._guess_mime(rel))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path == "/action":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                data   = json.loads(body)
                uuid   = data["uuid"]
                action = data["action"]
            except (KeyError, json.JSONDecodeError):
                self.send_json({"ok": False, "error": "bad request"}, 400)
                return

            if action == "doodles_done":
                ok = set_stage(uuid, "DOODLES_DONE", "HTML_CREATED")
                self.send_json({"ok": ok, "error": None if ok else "stage mismatch"})
            elif action == "approve":
                ok = set_stage(uuid, "HTML_APPROVED", "DOODLES_DONE")
                self.send_json({"ok": ok, "error": None if ok else "stage mismatch"})
            else:
                self.send_json({"ok": False, "error": "unknown action"}, 400)

        else:
            self.send_response(404)
            self.end_headers()

    @staticmethod
    def _guess_mime(filename):
        ext = Path(filename).suffix.lower()
        return {
            ".html": "text/html", ".css": "text/css", ".js": "application/javascript",
            ".png": "image/png",  ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml", ".woff2": "font/woff2", ".woff": "font/woff",
            ".ttf": "font/ttf",
        }.get(ext, "application/octet-stream")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DadFit Carousel Viewer")
    parser.add_argument("--batch", type=int, default=1, help="Batch number (default: 1)")
    parser.add_argument("--port",  type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}", file=sys.stderr)
        sys.exit(1)

    Handler.batch_no = args.batch

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"\n  DadFit Carousel Viewer — Batch {args.batch}")
    print(f"  Open: {url}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
