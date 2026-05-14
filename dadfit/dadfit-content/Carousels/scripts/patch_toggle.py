#!/usr/bin/env python3
"""
Patch existing carousel.html files to add:
  1. Gradient toggle CSS (before </style>)
  2. Gradient toggle button JS (inside the IIFE, before closing })(); )
  3. Add class="gradient-glow" to existing bare glow divs

Idempotent — won't double-inject if already present.

Usage:
    python3 Carousels/scripts/patch_toggle.py --batch 1
"""
import re, os, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, default=1)
args = parser.parse_args()

BATCH_DIR = f'Carousels/data/batch_{args.batch}'

TOGGLE_CSS = """\
    /* ── Gradient toggle ── */
    html.flat-bg .s-bg[style] { background: var(--primary-bg) !important; }
    html.flat-bg .gradient-glow { display: none !important; }
    #gradient-toggle-btn {
      position: fixed; bottom: 28px; right: 28px; z-index: 9999;
      padding: 10px 20px; border-radius: 8px; border: 1px solid #34C363;
      background: #1a1a1a; color: #34C363; font-family: 'Inter',sans-serif;
      font-size: 13px; font-weight: 600; cursor: pointer; letter-spacing: 0.5px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.5);
      transition: background 0.15s, color 0.15s;
    }
    #gradient-toggle-btn:hover { background: #34C363; color: #1a1a1a; }

  </style>"""

TOGGLE_JS = """\
  //document.addEventListener('DOMContentLoaded',function(){
  //document.querySelectorAll('.s-bg').forEach(function(bg){bg.insertAdjacentHTML('beforeend',buildSVG());});  });

  // ── Gradient toggle button ──
  document.addEventListener('DOMContentLoaded', function() {
    var btn = document.createElement('button');
    btn.id = 'gradient-toggle-btn';
    btn.textContent = 'Gradients: ON';
    btn.addEventListener('click', function() {
      var isFlat = document.documentElement.classList.toggle('flat-bg');
      btn.textContent = isFlat ? 'Gradients: OFF' : 'Gradients: ON';
    });
    document.body.appendChild(btn);
  });
})();"""

OLD_JS_TAIL = """\
  //document.addEventListener('DOMContentLoaded',function(){
  //document.querySelectorAll('.s-bg').forEach(function(bg){bg.insertAdjacentHTML('beforeend',buildSVG());});  });
})();"""

OLD_CSS_CLOSE = "  </style>"


def patch_file(path):
    with open(path, encoding='utf-8') as fh:
        content = fh.read()

    modified = False

    # 1. Add toggle CSS before </style> (idempotent check)
    if 'gradient-toggle-btn' not in content:
        content = content.replace(OLD_CSS_CLOSE, TOGGLE_CSS, 1)
        modified = True

    # 2. Replace JS tail to inject button (idempotent check)
    if 'gradient-toggle-btn' not in content or OLD_JS_TAIL in content:
        if OLD_JS_TAIL in content:
            content = content.replace(OLD_JS_TAIL, TOGGLE_JS, 1)
            modified = True

    # 3. Add class="gradient-glow" to bare glow divs
    bare_glow = '<div style="position:absolute;inset:0;background:radial-gradient('
    classed_glow = '<div class="gradient-glow" style="position:absolute;inset:0;background:radial-gradient('
    if bare_glow in content:
        content = content.replace(bare_glow, classed_glow)
        modified = True

    if modified:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(content)
        print(f'  ✓  {os.path.basename(os.path.dirname(path))}')

    return modified


def main():
    files = sorted(glob.glob(os.path.join(BATCH_DIR, '*/carousel.html')))
    print(f'Found {len(files)} carousel files in {BATCH_DIR}')
    patched = sum(patch_file(p) for p in files)
    print(f'\nDone — patched {patched} / {len(files)} files.')


if __name__ == '__main__':
    main()
