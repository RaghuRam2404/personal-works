"""
Step 8 — State Checker
Queries CAPTION_WRITTEN rows, diffs against html_checkpoint.json,
and outputs the next N carousels to /tmp/batch_{N}_html_round.json.

Usage (run from project root):
    python3 Carousels/scripts/step8_stepa.py [--batch 1] [--count 5]
"""
import sqlite3, json, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--count', type=int, default=1)
args = parser.parse_args()
BATCH_NO = args.batch
COUNT = args.count

checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/html_checkpoint.json'

conn = sqlite3.connect('Carousels/data/db.sqlite')
all_rows = conn.execute(
    'SELECT uuid, running_no, title, keyword, category, hook, script_content, cta '
    'FROM Carousel WHERE batch_no = ? AND current_stage = "CAPTION_WRITTEN" '
    'ORDER BY running_no',
    (BATCH_NO,)
).fetchall()
conn.close()

all_carousels = [{
    'uuid': r[0], 'running_no': r[1], 'title': r[2], 'keyword': r[3],
    'category': r[4], 'hook': r[5], 'script_content': r[6], 'cta': r[7],
    'folder_name': f'{r[1]}_{r[0]}'
} for r in all_rows]

done_uuids = set()
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, encoding='utf-8') as f:
        done_uuids = {e['uuid'] for e in json.load(f)}

remaining = [c for c in all_carousels if c['uuid'] not in done_uuids]
this_round = remaining[:COUNT]

total = len(all_carousels)
print(f'Done: {len(done_uuids)}/{total}  |  Remaining: {len(remaining)}  |  This round: {len(this_round)}')

if not this_round:
    print('STATUS: ALL DONE — proceed to step8_insert.py')
else:
    running_nos = ', '.join(str(c['running_no']) for c in this_round)
    print(f'STATUS: PROCESS running_no {running_nos}')
    for c in this_round:
        print(f'  # {c["running_no"]} [{c["category"]}] {c["title"]}')
    out_path = f'/tmp/batch_{BATCH_NO}_html_round.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(this_round, f, indent=2, ensure_ascii=False)
    print(f'\nWritten to {out_path}')
