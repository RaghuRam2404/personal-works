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
