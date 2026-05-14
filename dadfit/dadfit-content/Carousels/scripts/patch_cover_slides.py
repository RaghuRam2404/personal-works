#!/usr/bin/env python3
"""
Patch existing batch carousel.html files:
  - A1: remove counter/logo/subtext/greenbar/swipe/arrow → centered headline + doodle
  - A4: remove counter/logo/greenbar/swipe/arrow → centered caveat+headline + doodle
  - A5: remove counter/logo/greenbar+subtext/arrow → centered marker+headline + doodle

Usage (from project root):
    python3 Carousels/scripts/patch_cover_slides.py --batch 1
"""

import re
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=1)
args = parser.parse_args()

BATCH_DIR = f'Carousels/data/batch_{args.batch}'

DOODLE_ICON_HTML = (
    '  <div class="doodle-ph-icon">\n'
    '    <svg width="44" height="44" viewBox="0 0 44 44" fill="none">'
    '<path d="M32 4L40 12L14 38H6V30L32 4Z" stroke="#34C363" stroke-width="2" stroke-linejoin="round"/>'
    '<path d="M26 10L34 18" stroke="#34C363" stroke-width="2"/>'
    '<path d="M6 30L14 38" stroke="#34C363" stroke-width="2"/>'
    '</svg>\n'
    '    <span>Doodle<br>placeholder</span>\n'
    '  </div>\n'
    '</div>'
)


def extract_doodle_img(block):
    """Return (src, alt) from the first <img> in a doodle-ph block, or None."""
    m = re.search(r'<img src="([^"]+)" alt="([^"]*)" />', block)
    return (m.group(1), m.group(2)) if m else None


def rebuild_doodle_ph(src, alt):
    return (
        '<div class="doodle-ph">\n'
        f'  <img src="{src}" alt="{alt}" />\n'
        f'{DOODLE_ICON_HTML}'
    )


def patch_a1(block):
    img = extract_doodle_img(block)
    if not img:
        return None
    src, alt = img

    # Headline is inside s-zone > first child div (font-size:108px)
    hl_m = re.search(
        r"<div style=\"font-family:'Inter',sans-serif;font-weight:800;font-size:108px;[^\"]*\">"
        r"\s*(.*?)\s*</div>",
        block, re.DOTALL
    )
    if not hl_m:
        return None
    headline = hl_m.group(1).strip()

    return (
        '<div class="s-bg slide-type-A1"></div>\n'
        + rebuild_doodle_ph(src, alt) + '\n'
        '<div style="position:absolute;inset:0;display:flex;align-items:center;'
        'justify-content:center;padding:60px;text-align:center;">\n'
        "  <div style=\"font-family:'Inter',sans-serif;font-weight:800;font-size:108px;"
        'line-height:1.0;color:#fff;text-transform:uppercase;letter-spacing:-2px;">\n'
        f'    {headline}\n'
        '  </div>\n'
        '</div>'
    )


def patch_a4(block):
    img = extract_doodle_img(block)
    if not img:
        return None
    src, alt = img

    caveat_m = re.search(
        r"<div style=\"font-family:'Caveat',cursive;font-weight:700;font-size:68px;[^\"]*\">"
        r"\s*(.*?)\s*</div>",
        block, re.DOTALL
    )
    hl_m = re.search(
        r"<div style=\"font-family:'Inter',sans-serif;font-weight:800;font-size:104px;[^\"]*\">"
        r"\s*(.*?)\s*</div>",
        block, re.DOTALL
    )
    if not hl_m:
        return None
    headline = hl_m.group(1).strip()
    caveat = caveat_m.group(1).strip() if caveat_m else ''

    caveat_html = ''
    if caveat:
        caveat_html = (
            "  <div style=\"font-family:'Caveat',cursive;font-weight:700;font-size:68px;"
            'color:#ADADAD;line-height:1.15;margin-bottom:32px;">'
            f'{caveat}</div>\n'
        )

    return (
        '<div class="s-bg slide-type-A4"></div>\n'
        + rebuild_doodle_ph(src, alt) + '\n'
        '<div style="position:absolute;inset:0;display:flex;flex-direction:column;'
        'align-items:center;justify-content:center;padding:60px;text-align:center;">\n'
        + caveat_html
        + "  <div style=\"font-family:'Inter',sans-serif;font-weight:800;font-size:104px;"
        'line-height:0.95;color:#fff;text-transform:uppercase;letter-spacing:-2px;">\n'
        f'    {headline}\n'
        '  </div>\n'
        '</div>'
    )


def patch_a5(block):
    img = extract_doodle_img(block)
    if not img:
        return None
    src, alt = img

    marker_m = re.search(
        r"<div style=\"font-family:'Permanent Marker',cursive;font-size:72px;[^\"]*\">"
        r"\s*(.*?)\s*</div>",
        block, re.DOTALL
    )
    hl_m = re.search(
        r"<div style=\"font-family:'Inter',sans-serif;font-weight:800;font-size:100px;[^\"]*\">"
        r"\s*(.*?)\s*</div>",
        block, re.DOTALL
    )
    if not hl_m:
        return None
    headline = hl_m.group(1).strip()
    marker_word = marker_m.group(1).strip() if marker_m else ''

    marker_html = ''
    if marker_word:
        marker_html = (
            "  <div style=\"font-family:'Permanent Marker',cursive;font-size:72px;"
            f'color:#34C363;line-height:1.1;margin-bottom:28px;">{marker_word}</div>\n'
        )

    return (
        '<div class="s-bg slide-type-A5"></div>\n'
        + rebuild_doodle_ph(src, alt) + '\n'
        '<div style="position:absolute;inset:0;display:flex;flex-direction:column;'
        'align-items:center;justify-content:center;padding:60px;text-align:center;">\n'
        + marker_html
        + "  <div style=\"font-family:'Inter',sans-serif;font-weight:800;font-size:100px;"
        'line-height:0.95;color:#fff;text-transform:uppercase;letter-spacing:-2px;">\n'
        f'    {headline}\n'
        '  </div>\n'
        '</div>'
    )


PATCHERS = {
    'A1': patch_a1,
    'A4': patch_a4,
    'A5': patch_a5,
}


def process_file(path):
    with open(path, encoding='utf-8') as fh:
        content = fh.read()

    modified = False

    for slide_type, patcher in PATCHERS.items():
        # Match from the s-bg div to the blank-line + 8-space </div> that closes the slide
        pattern = (
            rf'(<div class="s-bg slide-type-{slide_type}"><\/div>)(.*?)'
            r'(\n\n        <\/div>)'
        )

        def replacer(m, _patcher=patcher, _stype=slide_type):
            full_block = m.group(1) + m.group(2)
            new = _patcher(full_block)
            if new is None:
                print(f'  WARNING: could not patch {_stype} in {os.path.basename(path)}')
                return m.group(0)
            return new + m.group(3)

        new_content, n = re.subn(pattern, replacer, content, flags=re.DOTALL)
        if n > 0:
            content = new_content
            modified = True
            folder = os.path.basename(os.path.dirname(path))
            print(f'  Patched {slide_type}: {folder}')

    if modified:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)

    return modified


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
