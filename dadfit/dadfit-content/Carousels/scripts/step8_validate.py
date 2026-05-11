"""
Step 8 — Round Validator + Checkpoint Writer
Reads subagent results from /tmp/html_round_results.json,
validates each carousel.html, and if all pass:
  - appends to html_checkpoint.json
  - appends doodle_prompts to doodle_prompts.json

Usage (run from project root):
    python3 Carousels/scripts/step8_validate.py [--batch 1]

Before running, write subagent results to /tmp/html_round_results.json
as a JSON array. Each entry must have:
  {
    "uuid": "...",
    "running_no": N,
    "folder_name": "N_uuid",
    "slide_count": N,
    "html_path": "Carousels/data/batch_1/N_uuid/carousel.html",
    "doodle_prompts": [
      {"running_no": N, "image_name": "N-d-01.png", "prompt": "..."},
      ...
    ]
  }
"""
import os, json, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=1)
args = parser.parse_args()
BATCH_NO = args.batch

RESULTS_PATH = '/tmp/html_round_results.json'

if not os.path.exists(RESULTS_PATH):
    print(f'ERROR: {RESULTS_PATH} not found. Write subagent results there first.')
    exit(1)

with open(RESULTS_PATH, encoding='utf-8') as f:
    results = json.load(f)

errors = []
passed = []
all_doodle_prompts = []

for entry in results:
    uuid = entry['uuid']
    html_path = entry['html_path']
    label = f"#{entry.get('running_no', '?')} ({uuid[:8]})"
    fail = False

    # 1. HTML file exists and is substantial
    if not os.path.exists(html_path):
        errors.append(f"{label}: carousel.html NOT FOUND at {html_path}")
        fail = True
    else:
        size = os.path.getsize(html_path)
        if size < 10000:
            errors.append(f"{label}: carousel.html too small ({size} bytes) — likely truncated")
            fail = True
        else:
            with open(html_path, encoding='utf-8') as f:
                content = f.read()
            if '../../../../Resources/Images/logo.png' not in content:
                errors.append(f"{label}: logo path wrong — must be ../../../../Resources/Images/logo.png")
                fail = True
            if 'Brand assets' in content:
                errors.append(f"{label}: old template logo path '../../Brand assets/' still present")
                fail = True
            if 'buildSVG' not in content:
                errors.append(f"{label}: scribble generator <script> block missing")
                fail = True
            if '../doodles/' not in content:
                errors.append(f"{label}: doodle src must use ../doodles/ (shared batch folder)")
                fail = True

    # 2. doodle_prompts present
    if not entry.get('doodle_prompts'):
        errors.append(f"{label}: doodle_prompts missing from subagent response")
        fail = True

    size_info = f"{os.path.getsize(html_path)//1024}KB" if os.path.exists(html_path) else "MISSING"
    status = "FAIL" if fail else "OK"
    print(f"{label}: {entry.get('slide_count','?')} slides | {size_info} | {status}")

    if not fail:
        passed.append({'uuid': entry['uuid'], 'folder_name': entry['folder_name']})
        all_doodle_prompts.extend(entry['doodle_prompts'])

print()
if errors:
    print("FAILURES:")
    for e in errors:
        print(" ", e)
    print("\nCheckpoint NOT updated — fix failures first, then re-run.")
else:
    print("All HTML files PASSED validation.")

    # Update html_checkpoint.json
    checkpoint_path = f'Carousels/data/batch_{BATCH_NO}/html_checkpoint.json'
    with open(checkpoint_path, encoding='utf-8') as f:
        checkpoint = json.load(f)
    checkpoint.extend(passed)
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    print(f'html_checkpoint.json updated: {len(checkpoint)} entries total')

    # Append doodle prompts
    doodle_path = f'Carousels/data/batch_{BATCH_NO}/doodle_prompts.json'
    with open(doodle_path, encoding='utf-8') as f:
        existing = json.load(f)
    existing.extend(all_doodle_prompts)
    with open(doodle_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f'doodle_prompts.json updated: {len(existing)} doodle entries total')
