"""
Step 8 — DB Insert
Reads html_checkpoint.json and sets folder_name + current_stage = HTML_CREATED
for all checkpointed carousels.

Usage (run from project root):
    python3 Carousels/scripts/step8_insert.py [--batch 1]
"""
import sqlite3, json, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=1)
args = parser.parse_args()
BATCH_NO = args.batch

checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/html_checkpoint.json'
with open(checkpoint_path, encoding='utf-8') as f:
    entries = json.load(f)

conn = sqlite3.connect('Carousels/data/db.sqlite')
updated = 0
for e in entries:
    conn.execute(
        'UPDATE Carousel SET folder_name = ?, current_stage = "HTML_CREATED" WHERE uuid = ?',
        (e['folder_name'], e['uuid'])
    )
    updated += 1

conn.commit()
conn.close()
print(f'Updated {updated} rows to HTML_CREATED for batch {BATCH_NO}')

# Verify
conn = sqlite3.connect('Carousels/data/db.sqlite')
count = conn.execute(
    'SELECT COUNT(*) FROM Carousel WHERE batch_no = ? AND current_stage = "HTML_CREATED"',
    (BATCH_NO,)
).fetchone()[0]
conn.close()
print(f'Verified: {count} rows with current_stage = HTML_CREATED')
