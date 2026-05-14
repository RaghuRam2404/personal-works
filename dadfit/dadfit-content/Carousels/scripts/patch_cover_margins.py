#!/usr/bin/env python3
"""
Patch pass 2: increase horizontal margins on A1/A4/A5 cover slides.
  - padding:60px  →  padding:60px 150px  (outer centered wrapper)
  - <div class="doodle-ph">  →  <div class="doodle-ph" style="right:150px;">
    (only inside A1/A4/A5 slide blocks)

Usage (from project root):
    python3 Carousels/scripts/patch_cover_margins.py --batch 1
"""

import re
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=1)
args = parser.parse_args()

BATCH_DIR = f'Carousels/data/batch_{args.batch}'

# Matches cover slide block: s-bg slide-type-A{1,4,5} up to the closing wrapper </div>
COVER_BLOCK_RE = re.compile(
    r'(<div class="s-bg slide-type-A[145]"></div>)(.*?)(\n\n        </div>)',
    re.DOTALL
)


def fix_block(m):
    block = m.group(1) + m.group(2)
    # 1. doodle-ph: add right:150px if not already present
    block = re.sub(
        r'<div class="doodle-ph">',
        '<div class="doodle-ph" style="right:150px;">',
        block
    )
    # Already has right:150px → leave untouched (idempotent)
    block = block.replace(
        '<div class="doodle-ph" style="right:150px;" style="right:150px;">',
        '<div class="doodle-ph" style="right:150px;">'
    )
    # 2. padding:60px → padding:60px 150px in the centered wrapper
    block = block.replace(
        'padding:60px;text-align:center;',
        'padding:60px 150px;text-align:center;'
    )
    return block + m.group(3)


def process_file(path):
    with open(path, encoding='utf-8') as fh:
        content = fh.read()

    new_content, n = COVER_BLOCK_RE.subn(fix_block, content)
    if n > 0:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(new_content)
        folder = os.path.basename(os.path.dirname(path))
        print(f'  Patched ({n} block): {folder}')
        return True
    return False


def main():
    pattern = os.path.join(BATCH_DIR, '*/carousel.html')
    files = sorted(glob.glob(pattern))
    print(f'Found {len(files)} carousel files in {BATCH_DIR}')

    patched = 0
    for path in files:
        if process_file(path):
            patched += 1

    print(f'\nDone — patched {patched} / {len(files)} files.')


if __name__ == '__main__':
    main()
