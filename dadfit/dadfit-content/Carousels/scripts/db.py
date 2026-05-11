#!/usr/bin/env python3
"""
db.py -- Interactive SQLite CLI for the Carousel pipeline.
Behaves like psql: pretty tables, persistent history, meta-commands.

Usage:
  python3 scripts/db.py                  # interactive REPL
  python3 scripts/db.py "SELECT ..."     # run a single query and exit
  python3 scripts/db.py --batch 1 status # shortcut: show pipeline status
"""

import sqlite3
import os
import sys
import argparse
import readline
import atexit
import textwrap

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "db.sqlite")
HISTORY_FILE = os.path.expanduser("~/.carousel_db_history")

STAGE_ORDER = [
    "TOPIC_FETCHED", "CATEGORIZED", "ORDER_SET", "HOOK_WRITTEN",
    "SCRIPT_WRITTEN", "CTA_WRITTEN", "CAPTION_WRITTEN", "HTML_CREATED",
    "DOODLES_DONE", "IMAGES_CREATED", "MUSIC_CHOSEN", "READY_TO_PUBLISH",
    "PUBLISHED", "MONITORED",
]


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_table(cursor, rows):
    if not rows:
        print("(0 rows)")
        return
    col_names = [d[0] for d in cursor.description]
    col_widths = [len(c) for c in col_names]
    str_rows = []
    for row in rows:
        str_row = [str(v) if v is not None else "NULL" for v in row]
        for i, v in enumerate(str_row):
            col_widths[i] = max(col_widths[i], len(v))
        str_rows.append(str_row)

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header = "|" + "|".join(" %-*s " % (w, c) for w, c in zip(col_widths, col_names)) + "|"
    print(sep)
    print(header)
    print(sep)
    for row in str_rows:
        print("|" + "|".join(" %-*s " % (w, v) for w, v in zip(col_widths, row)) + "|")
    print(sep)
    print("(%d row%s)" % (len(rows), "" if len(rows) == 1 else "s"))


# ── Meta-commands ─────────────────────────────────────────────────────────────

def meta_tables(conn, _):
    c = conn.cursor()
    c.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name")
    rows = c.fetchall()
    print_table(c, rows)

def meta_describe(conn, args):
    table = args.strip()
    if not table:
        print("Usage: \\d tablename")
        return
    c = conn.cursor()
    c.execute("PRAGMA table_info(%s)" % table)
    rows = c.fetchall()
    if not rows:
        print("Table '%s' not found." % table)
        return
    print_table(c, rows)

def meta_indexes(conn, args):
    table = args.strip() or None
    c = conn.cursor()
    if table:
        c.execute("PRAGMA index_list(%s)" % table)
    else:
        c.execute("SELECT name, tbl_name FROM sqlite_master WHERE type='index' ORDER BY tbl_name")
    print_table(c, c.fetchall())

def meta_status(conn, args):
    batch_no = int(args.strip()) if args.strip().isdigit() else None
    c = conn.cursor()
    q = "SELECT current_stage, COUNT(*) as count FROM Carousel"
    params = []
    if batch_no:
        q += " WHERE batch_no = ?"
        params.append(batch_no)
    q += " GROUP BY current_stage"
    c.execute(q, params)
    rows = dict(c.fetchall())
    label = ("Batch %d" % batch_no) if batch_no else "All batches"
    total = sum(rows.values())
    print("\nPipeline Status — %s  (total: %d)" % (label, total))
    print("-" * 52)
    for stage in STAGE_ORDER:
        count = rows.get(stage, 0)
        bar = "#" * min(count, 30)
        print("  %-22s  %4d  %s" % (stage, count, bar))
    print("-" * 52 + "\n")

def meta_help(conn, _):
    print(textwrap.dedent("""
    Meta-commands:
      \\t  or  \\dt          List all tables
      \\d  <table>          Describe table columns (like \\d in psql)
      \\i  [table]          List indexes
      \\s  [batch_no]       Pipeline stage status (like orchestrator status)
      \\q                   Quit
      \\h  or  \\?           Show this help

    Tips:
      - End SQL statements with ; or just press Enter after a complete statement
      - Multi-line queries are supported — keep typing until you add ;
      - Use UP/DOWN arrows for history (saved to ~/.carousel_db_history)

    Example queries:
      SELECT title, category, current_stage FROM Carousel WHERE batch_no = 1 LIMIT 10;
      SELECT category, COUNT(*) FROM Carousel GROUP BY category;
      SELECT * FROM CarouselPerformance LIMIT 5;
    """))

META_COMMANDS = {
    "\\t": meta_tables,
    "\\dt": meta_tables,
    "\\d": meta_describe,
    "\\i": meta_indexes,
    "\\s": meta_status,
    "\\q": None,  # handled separately
    "\\h": meta_help,
    "\\?": meta_help,
}


# ── Query runner ──────────────────────────────────────────────────────────────

def run_query(conn, sql):
    sql = sql.strip()
    if not sql:
        return
    try:
        c = conn.cursor()
        c.execute(sql)
        if c.description:
            rows = c.fetchall()
            print_table(c, rows)
        else:
            conn.commit()
            print("OK  (%d rows affected)" % c.rowcount)
    except sqlite3.Error as e:
        print("ERROR: %s" % e)


# ── REPL ──────────────────────────────────────────────────────────────────────

def repl(conn):
    # History
    if os.path.exists(HISTORY_FILE):
        readline.read_history_file(HISTORY_FILE)
    readline.set_history_length(500)
    atexit.register(readline.write_history_file, HISTORY_FILE)

    print("carousel=# connected to %s" % os.path.abspath(DB_PATH))
    print("Type \\h for help, \\q to quit.\n")

    buffer = []
    while True:
        prompt = "carousel=# " if not buffer else "carousel-# "
        try:
            line = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        # Meta-commands (must be on their own line)
        cmd = line.split()[0] if line.split() else ""
        args = line[len(cmd):].strip()

        if cmd == "\\q":
            print("Bye.")
            break
        if cmd in META_COMMANDS:
            META_COMMANDS[cmd](conn, args)
            continue

        # SQL accumulator — execute when line ends with ;
        buffer.append(line)
        if line.rstrip().endswith(";"):
            sql = " ".join(buffer).rstrip(";")
            run_query(conn, sql)
            buffer = []


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DB_PATH):
        print("ERROR: DB not found at %s\nRun python3 scripts/init_db.py first." % DB_PATH)
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Single-query mode: python3 scripts/db.py "SELECT ..."
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        sql = " ".join(sys.argv[1:]).rstrip(";")
        run_query(conn, sql)
        conn.close()
        return

    # Shortcut: python3 scripts/db.py --batch 1 status
    if len(sys.argv) >= 3 and sys.argv[-1] == "status":
        batch = sys.argv[sys.argv.index("--batch") + 1] if "--batch" in sys.argv else ""
        meta_status(conn, batch)
        conn.close()
        return

    repl(conn)
    conn.close()


if __name__ == "__main__":
    main()
