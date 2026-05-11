"""
init_db.py
Creates the SQLite database and tables for the Carousel pipeline.
Run once: python scripts/init_db.py
Safe to re-run — uses CREATE TABLE IF NOT EXISTS.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "db.sqlite")


def init():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── Carousel ──────────────────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS Carousel (
            uuid                        TEXT PRIMARY KEY,
            batch_no                    INTEGER NOT NULL,
            running_no                  INTEGER,
            title                       TEXT NOT NULL,
            keyword                     TEXT,
            trait                       TEXT,
            category                    TEXT CHECK(category IN ('TOFU', 'MOFU', 'BOFU')),
            hook                        TEXT,
            caption                     TEXT,
            script_content              TEXT,
            cta                         TEXT,
            folder_name                 TEXT,
            upload_status               TEXT DEFAULT 'PENDING'
                                            CHECK(upload_status IN ('PENDING', 'PUBLISHED')),
            current_stage               TEXT DEFAULT 'TOPIC_FETCHED'
                                            CHECK(current_stage IN (
                                                'TOPIC_FETCHED',
                                                'CATEGORIZED',
                                                'ORDER_SET',
                                                'HOOK_WRITTEN',
                                                'SCRIPT_WRITTEN',
                                                'CTA_WRITTEN',
                                                'CAPTION_WRITTEN',
                                                'HTML_CREATED',
                                                'DOODLES_DONE',
                                                'IMAGES_CREATED',
                                                'MUSIC_CHOSEN',
                                                'READY_TO_PUBLISH',
                                                'PUBLISHED',
                                                'MONITORED'
                                            )),
            last_performance_monitored  TEXT
        )
    """)

    # ── CarouselPerformance ───────────────────────────────────────────────────
    c.execute("""
        CREATE TABLE IF NOT EXISTS CarouselPerformance (
            uuid                    TEXT PRIMARY KEY,
            carousel_uuid           TEXT NOT NULL REFERENCES Carousel(uuid),
            performance_taken_time  TEXT NOT NULL,
            views                   INTEGER DEFAULT 0,
            likes                   INTEGER DEFAULT 0,
            comments                INTEGER DEFAULT 0,
            shares                  INTEGER DEFAULT 0,
            saves                   INTEGER DEFAULT 0,
            reach                   INTEGER DEFAULT 0,
            profile_visits          INTEGER DEFAULT 0,
            follows_from_post       INTEGER DEFAULT 0,
            notes                   TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"✓ Database initialised at: {os.path.abspath(DB_PATH)}")


if __name__ == "__main__":
    init()
