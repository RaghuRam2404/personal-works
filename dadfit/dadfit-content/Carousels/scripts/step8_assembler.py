"""
Step 8 — HTML Assembler
Reads a subagent-produced JSON containing only slide HTML, then wraps it
with the verbatim CSS and script blocks extracted from the master template.
This guarantees the output HTML always matches the template boilerplate exactly,
regardless of what the subagent copies or omits.

Usage (run from project root):
    python3 Carousels/scripts/step8_assembler.py --input /tmp/carousel_{uuid}.json [--batch 1]

Input JSON (written by subagent to /tmp/carousel_{uuid}.json):
    {
      "uuid": "...",
      "running_no": N,
      "folder_name": "N_uuid",
      "slide_count": N,
      "page_title": "Hook Headline Here — DadFit Carousel",
      "slides_html": "<div class=\\"section\\">...</div>\\n<div class=\\"section\\">...</div>...",
      "doodle_prompts": [
        {"running_no": N, "image_name": "N-d-01.png", "prompt": "..."},
        ...
      ]
    }

Output:
    - Writes carousel.html to Carousels/data/batch_{N}/{folder_name}/carousel.html
    - Prints a result JSON line to stdout (for the main agent to collect)
"""
import json, os, re, argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Path to subagent JSON file')
parser.add_argument('--batch', type=int, default=1)
args = parser.parse_args()

BATCH_NO = args.batch
TEMPLATE_PATH = 'Carousels/templates/design 1/DadFit Carousel Templates.html'
FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;700'
    '&family=Inter:wght@400;500;600;700;800&family=Permanent+Marker&display=swap" rel="stylesheet">'
)

# ── Load subagent input ──────────────────────────────────────────────────────
if not os.path.exists(args.input):
    print(f'ERROR: Input file not found: {args.input}', file=sys.stderr)
    sys.exit(1)

with open(args.input, encoding='utf-8') as f:
    data = json.load(f)

uuid         = data['uuid']
running_no   = data['running_no']
folder_name  = data['folder_name']
slide_count  = data['slide_count']
page_title   = data.get('page_title', f'Carousel {running_no} — DadFit')
slides_html  = data['slides_html']
doodle_prompts = data.get('doodle_prompts', [])

# ── Extract CSS + Script from template ──────────────────────────────────────
with open(TEMPLATE_PATH, encoding='utf-8') as f:
    template = f.read()

# CSS block — everything from <style> to </style> inclusive
style_start = template.index('<style>')
style_end   = template.index('</style>') + len('</style>')
style_block = template[style_start:style_end]

# Script block — last <script>...</script> in the file (the scribble generator)
script_start = template.rindex('<script>')
script_end   = template.rindex('</script>') + len('</script>')
script_block = template[script_start:script_end]

# ── Validate slides_html structure ─────────────────────────────────────────
# Every <div class="section"> must contain both a section-header and a slide-label.
# If any are missing the assembler rejects the file so the subagent must retry.

import re as _re

def _validate_slides(html: str) -> list[str]:
    """Return a list of error strings. Empty list = valid."""
    errors = []
    sections = _re.findall(r'<div class="section">(.*?)</div>\s*</div>\s*</div>\s*(?=<div class="section">|$)', html, _re.DOTALL)
    # Simpler: count occurrences
    n_sections     = html.count('<div class="section">')
    n_headers      = html.count('class="section-header">')
    n_labels       = html.count('class="slide-label"')
    if n_sections == 0:
        errors.append('No <div class="section"> blocks found in slides_html')
        return errors
    if n_headers < n_sections:
        errors.append(
            f'Missing section-header: found {n_headers} but expected {n_sections} '
            f'(one per slide). Every <div class="section"> must start with '
            f'<div class="section-header"><h2>Slide N — TYPE — Label</h2></div>.'
        )
    if n_labels < n_sections:
        errors.append(
            f'Missing slide-label: found {n_labels} but expected {n_sections} '
            f'(one per slide). Every slide-item must have '
            f'<div class="slide-label"><strong>TYPE</strong> — Label</div>.'
        )
    return errors

validation_errors = _validate_slides(slides_html)
if validation_errors:
    print('VALIDATION FAILED — slides_html rejected:', file=sys.stderr)
    for e in validation_errors:
        print(f'  ✗ {e}', file=sys.stderr)
    print('\nThe subagent must re-generate this carousel with the correct structure.', file=sys.stderr)
    print('See SKILL.md Step 2 — the section-header and slide-label divs are mandatory.', file=sys.stderr)
    result_err = {
        'uuid': data.get('uuid', '?'),
        'running_no': data.get('running_no', '?'),
        'status': 'VALIDATION_FAILED',
        'errors': validation_errors
    }
    print(json.dumps(result_err))
    sys.exit(2)

# ── Fix logo paths in slides_html ───────────────────────────────────────────
# Template uses ../../Brand assets/logo.png; output is 4 levels deep
slides_html = slides_html.replace(
    '../../Brand assets/logo.png',
    '../../../../Resources/Images/logo.png'
)
# Catch any other variant just in case
slides_html = re.sub(
    r'src="[^"]*Brand assets/logo\.png"',
    'src="../../../../Resources/Images/logo.png"',
    slides_html
)

# ── Build output folder ──────────────────────────────────────────────────────
out_dir  = f'Carousels/data/batch_{BATCH_NO}/{folder_name}'
out_path = f'{out_dir}/carousel.html'
os.makedirs(out_dir, exist_ok=True)

# ── Assemble full HTML ───────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{page_title}</title>
  {FONT_LINK}
  {style_block}
</head>
<body>

<h1 class="page-title">{page_title}</h1>
<p class="page-subtitle">{slide_count} slides · DadFit · Batch {BATCH_NO}</p>

{slides_html}

{script_block}
</body>
</html>"""

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

file_size = os.path.getsize(out_path)
print(f'Written: {out_path} ({file_size // 1024}KB, {slide_count} slides)', file=sys.stderr)

# ── Print result JSON for main agent to collect ──────────────────────────────
result = {
    'uuid': uuid,
    'running_no': running_no,
    'folder_name': folder_name,
    'slide_count': slide_count,
    'html_path': out_path,
    'doodle_prompts': doodle_prompts
}
print(json.dumps(result))
