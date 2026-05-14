#!/usr/bin/env python3
"""
Bake professional gradient backgrounds into:
  1. All snippet templates  (Carousels/templates/design 1/snippets/*.html)
  2. All generated carousel HTMLs  (Carousels/data/batch_{N}/*/carousel.html)

Skips photo-background slides (A3, B3, D2, F1) — they already have their own bg.
Idempotent: won't double-add if already patched.

Usage (from project root):
    python3 Carousels/scripts/patch_gradients.py --batch 1
"""

import re, os, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=1)
args = parser.parse_args()

SNIPPETS_DIR = "Carousels/templates/design 1/snippets"
BATCH_DIR    = f"Carousels/data/batch_{args.batch}"

# ── Gradient palette ──────────────────────────────────────────────────────────
# (linear_gradient, radial_glow)
# v1 intensity for A–G families; v2 (darkened) for H (CTA).
GRADIENTS = {
    # A — Cover / Hook: deep forest green
    'A1': ('linear-gradient(145deg,#0c1610 0%,#121e15 30%,#17211a 58%,#1c1e1c 100%)',
           'radial-gradient(ellipse at 75% 88%,rgba(52,195,99,0.10) 0%,rgba(52,195,99,0.02) 40%,transparent 65%)'),
    'A2': ('linear-gradient(145deg,#0c1610 0%,#121e15 30%,#17211a 58%,#1c1e1c 100%)',
           'radial-gradient(ellipse at 30% 70%,rgba(52,195,99,0.09) 0%,transparent 55%)'),
    'A4': ('linear-gradient(145deg,#0c1610 0%,#121e15 30%,#17211a 58%,#1c1e1c 100%)',
           'radial-gradient(ellipse at 75% 88%,rgba(52,195,99,0.10) 0%,rgba(52,195,99,0.02) 40%,transparent 65%)'),
    'A5': ('linear-gradient(145deg,#0c1610 0%,#121e15 30%,#17211a 58%,#1c1e1c 100%)',
           'radial-gradient(ellipse at 75% 88%,rgba(52,195,99,0.10) 0%,rgba(52,195,99,0.02) 40%,transparent 65%)'),

    # B — Content slides: dark indigo-navy
    'B1': ('linear-gradient(150deg,#131520 0%,#171b26 35%,#1a1e24 60%,#1c1e22 100%)',
           'radial-gradient(ellipse at 80% 20%,rgba(52,195,99,0.03) 0%,transparent 50%)'),
    'B2': ('linear-gradient(150deg,#131520 0%,#171b26 35%,#1a1e24 60%,#1c1e22 100%)',
           'radial-gradient(ellipse at 20% 80%,rgba(52,195,99,0.03) 0%,transparent 50%)'),
    'B4': ('linear-gradient(145deg,#111a13 0%,#151f16 35%,#192118 58%,#1c1e1c 100%)',
           'radial-gradient(ellipse at 60% 40%,rgba(52,195,99,0.04) 0%,transparent 52%)'),
    'B5': ('linear-gradient(150deg,#131520 0%,#171b26 35%,#1a1e24 60%,#1c1e22 100%)',
           'radial-gradient(ellipse at 70% 60%,rgba(52,195,99,0.03) 0%,transparent 50%)'),
    'B6': ('linear-gradient(155deg,#131516 0%,#171a1c 35%,#1a1d1f 60%,#1E1E1E 100%)',
           'radial-gradient(ellipse at 15% 50%,rgba(52,195,99,0.025) 0%,transparent 45%)'),

    # C — Pain / Problem: dark crimson
    'C1': ('linear-gradient(145deg,#1c1212 0%,#201616 35%,#1f1b1b 65%,#1E1E1E 100%)',
           'radial-gradient(ellipse at 20% 30%,rgba(255,107,107,0.03) 0%,transparent 55%)'),
    'C2': ('linear-gradient(145deg,#1c1212 0%,#201616 35%,#1f1b1b 65%,#1E1E1E 100%)',
           'radial-gradient(ellipse at 80% 70%,rgba(255,107,107,0.025) 0%,transparent 55%)'),
    'C3': ('linear-gradient(160deg,#1c1212 0%,#1f1616 30%,#1e1c1c 60%,#1E1E1E 100%)',
           'radial-gradient(ellipse at 50% 50%,rgba(255,107,107,0.02) 0%,transparent 60%)'),
    'C4': ('linear-gradient(145deg,#1c1212 0%,#201616 35%,#1f1b1b 65%,#1E1E1E 100%)',
           'radial-gradient(ellipse at 30% 60%,rgba(255,107,107,0.025) 0%,transparent 55%)'),

    # D — Stats / Data: deep teal-dark
    'D1': ('linear-gradient(145deg,#101a1c 0%,#141e20 30%,#181e1e 58%,#1c1e1e 100%)',
           'radial-gradient(ellipse at 30% 60%,rgba(52,195,99,0.035) 0%,transparent 55%)'),
    'D3': ('linear-gradient(145deg,#101a1c 0%,#141e20 30%,#181e1e 58%,#1c1e1e 100%)',
           'radial-gradient(ellipse at 60% 30%,rgba(52,195,99,0.035) 0%,transparent 55%)'),
    'D4': ('linear-gradient(145deg,#101a1c 0%,#141e20 30%,#181e1e 58%,#1c1e1e 100%)',
           'radial-gradient(ellipse at 40% 70%,rgba(52,195,99,0.035) 0%,transparent 55%)'),

    # E — Empathy / Story: warm amber-dark
    'E1': ('linear-gradient(145deg,#1a1510 0%,#1e1810 35%,#1c1a14 60%,#1e1e1c 100%)',
           'radial-gradient(ellipse at 30% 40%,rgba(255,180,80,0.05) 0%,transparent 50%)'),
    'E2': ('linear-gradient(145deg,#1a1510 0%,#1e1810 35%,#1c1a14 60%,#1e1e1c 100%)',
           'radial-gradient(ellipse at 70% 60%,rgba(255,180,80,0.05) 0%,transparent 50%)'),
    'E3': ('linear-gradient(145deg,#1a1510 0%,#1e1810 35%,#1c1a14 60%,#1e1e1c 100%)',
           'radial-gradient(ellipse at 50% 30%,rgba(255,180,80,0.05) 0%,transparent 50%)'),
    'E4': ('linear-gradient(145deg,#1a1510 0%,#1e1810 35%,#1c1a14 60%,#1e1e1c 100%)',
           'radial-gradient(ellipse at 40% 70%,rgba(255,180,80,0.05) 0%,transparent 50%)'),

    # F — Feature / Proof: slate green
    'F2': ('linear-gradient(150deg,#121614 0%,#161a18 35%,#181c1a 60%,#1e1e1e 100%)',
           'radial-gradient(ellipse at 60% 40%,rgba(52,195,99,0.03) 0%,transparent 50%)'),
    'F3': ('linear-gradient(150deg,#121614 0%,#161a18 35%,#181c1a 60%,#1e1e1e 100%)',
           'radial-gradient(ellipse at 40% 70%,rgba(52,195,99,0.03) 0%,transparent 50%)'),
    'F4': ('linear-gradient(150deg,#121614 0%,#161a18 35%,#181c1a 60%,#1e1e1e 100%)',
           'radial-gradient(ellipse at 70% 30%,rgba(52,195,99,0.03) 0%,transparent 50%)'),

    # G — Recap / Summary: rich forest green
    'G1': ('linear-gradient(145deg,#0f1710 0%,#141e14 35%,#182018 60%,#1c1e1c 100%)',
           'radial-gradient(ellipse at 50% 20%,rgba(52,195,99,0.04) 0%,transparent 50%)'),
    'G2': ('linear-gradient(145deg,#0f1710 0%,#141e14 35%,#182018 60%,#1c1e1c 100%)',
           'radial-gradient(ellipse at 30% 60%,rgba(52,195,99,0.04) 0%,transparent 50%)'),
    'G3': ('linear-gradient(145deg,#0f1710 0%,#141e14 35%,#182018 60%,#1c1e1c 100%)',
           'radial-gradient(ellipse at 70% 40%,rgba(52,195,99,0.04) 0%,transparent 50%)'),
    'G4': ('linear-gradient(145deg,#0f1710 0%,#141e14 35%,#182018 60%,#1c1e1c 100%)',
           'radial-gradient(ellipse at 50% 80%,rgba(52,195,99,0.04) 0%,transparent 50%)'),

    # H — CTA: strong green sweep (v2 — darkened ~40%)
    'H1': ('linear-gradient(145deg,#060c08 0%,#090f09 25%,#0c120d 50%,#0f1410 75%,#111512 100%)',
           'radial-gradient(ellipse at 50% 100%,rgba(52,195,99,0.08) 0%,rgba(52,195,99,0.02) 45%,transparent 65%)'),
    'H2': ('linear-gradient(145deg,#060c08 0%,#090f09 25%,#0c120d 50%,#0f1410 75%,#111512 100%)',
           'radial-gradient(ellipse at 50% 50%,rgba(52,195,99,0.06) 0%,transparent 55%)'),
    'H3': ('linear-gradient(145deg,#060c08 0%,#090f09 25%,#0c120d 50%,#0f1410 75%,#111512 100%)',
           'radial-gradient(ellipse at 30% 70%,rgba(52,195,99,0.06) 0%,transparent 55%)'),
    'H4': ('linear-gradient(145deg,#060c08 0%,#090f09 25%,#0c120d 50%,#0f1410 75%,#111512 100%)',
           'radial-gradient(ellipse at 70% 30%,rgba(52,195,99,0.06) 0%,transparent 55%)'),
}

GLOW_DIV = '<div class="gradient-glow" style="position:absolute;inset:0;background:{radial};pointer-events:none;"></div>'


def make_replacement(slide_type):
    lin, rad = GRADIENTS[slide_type]
    bg_div   = f'<div class="s-bg slide-type-{slide_type}" style="background:{lin};"></div>'
    glow_div = GLOW_DIV.format(radial=rad)
    return bg_div + '\n' + glow_div


def patch_content(content):
    """Replace all bare s-bg slide-type-XX divs with gradient versions. Idempotent."""
    modified = False
    for stype, (lin, rad) in GRADIENTS.items():
        bare   = f'<div class="s-bg slide-type-{stype}"></div>'
        # Also catch previously-patched versions (any style= already on the div)
        # We'll only replace the bare form (no style=) to avoid double-patching.
        # For already-patched: replace if the gradient has changed.
        old_styled_re = re.compile(
            rf'<div class="s-bg slide-type-{re.escape(stype)}" style="[^"]*"></div>\n'
            rf'<div(?:\s+class="gradient-glow")?\s+style="position:absolute;inset:0;background:[^"]*;pointer-events:none;"></div>'
        )
        new_block = make_replacement(stype)

        # Replace bare form
        if bare in content:
            content = content.replace(bare, new_block)
            modified = True
        # Replace previously-patched form (update gradient)
        elif old_styled_re.search(content):
            content = old_styled_re.sub(new_block, content)
            modified = True

    return content, modified


def patch_file(path, label=''):
    with open(path, encoding='utf-8') as fh:
        content = fh.read()
    new_content, modified = patch_content(content)
    if modified:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(new_content)
        print(f'  ✓  {label or os.path.basename(path)}')
    return modified


def main():
    # ── 1. Snippet templates ─────────────────────────────────────────────────
    print(f'\n── Snippets ({SNIPPETS_DIR}) ──')
    snippet_files = sorted(glob.glob(os.path.join(SNIPPETS_DIR, '*.html')))
    s_patched = sum(patch_file(p) for p in snippet_files)
    print(f'   {s_patched} / {len(snippet_files)} snippets updated.')

    # ── 2. Generated carousel HTMLs ──────────────────────────────────────────
    print(f'\n── Carousels ({BATCH_DIR}) ──')
    carousel_files = sorted(glob.glob(os.path.join(BATCH_DIR, '*/carousel.html')))
    c_patched = sum(
        patch_file(p, label=os.path.basename(os.path.dirname(p)))
        for p in carousel_files
    )
    print(f'\n   {c_patched} / {len(carousel_files)} carousels updated.')
    print('\nDone.')


if __name__ == '__main__':
    main()
