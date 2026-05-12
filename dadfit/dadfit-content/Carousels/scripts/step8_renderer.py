"""
Step 8 — Snippet Renderer
Reads a subagent content JSON (type + vars per slide),
loads the matching snippet HTML file, substitutes {{VAR}} placeholders,
wraps each slide in the section/slide-item structure, and assembles
a full carousel.html identical in structure to the master template output.

Usage (run from project root):
    python3 Carousels/scripts/step8_renderer.py --input /tmp/carousel_{uuid}.json [--batch 1]

Input JSON (written by subagent to /tmp/carousel_{uuid}.json):
    {
      "uuid": "...",
      "running_no": N,
      "folder_name": "N_uuid",
      "page_title": "Hook Headline — DadFit Carousel",
      "slides": [
        {
          "type": "A1",
          "slide_no": 1,
          "label": "Cover",
          "vars": {
            "COUNTER": "01 / 10",
            "HEADLINE": "YOU DON'T NEED <span style=\\"color:#34C363;\\">MORE TIME</span>",
            "SUBTEXT": "You need a smarter 20-minute plan.",
            "DOODLE_SRC": "../doodles/1-d-01.png",
            "DOODLE_ALT": "A cracked hourglass"
          }
        }
      ],
      "doodle_prompts": [
        {"running_no": N, "image_name": "N-d-01.png", "prompt": "..."}
      ]
    }

Output:
    - Writes carousel.html to Carousels/data/batch_{N}/{folder_name}/carousel.html
    - Prints a result JSON line to stdout (for the main agent to collect)
"""
import json, os, re, argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('--input',  required=True, help='Path to subagent content JSON')
parser.add_argument('--batch',  type=int, default=1)
args = parser.parse_args()

BATCH_NO      = args.batch
SNIPPETS_DIR  = 'Carousels/templates/design 1/snippets'
TEMPLATE_PATH = 'Carousels/templates/design 1/DadFit Carousel Templates.html'
FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '  <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400;700'
    '&family=Inter:wght@400;500;600;700;800&family=Permanent+Marker&display=swap" rel="stylesheet">'
)

# Vars injected automatically if the subagent doesn't supply them
DEFAULT_VARS = {
    'LOGO_SRC': '../../../../Resources/Images/logo.png',
}

# ── Load subagent content JSON ───────────────────────────────────────────────
if not os.path.exists(args.input):
    print(f'ERROR: Input file not found: {args.input}', file=sys.stderr)
    sys.exit(1)

with open(args.input, encoding='utf-8') as fh:
    data = json.load(fh)

uuid           = data['uuid']
running_no     = data['running_no']
folder_name    = data['folder_name']
page_title     = data.get('page_title', f'Carousel {running_no} — DadFit')
slides         = data['slides']
doodle_prompts = data.get('doodle_prompts', [])
slide_count    = len(slides)

# ── Pull CSS + script out of master template ─────────────────────────────────
with open(TEMPLATE_PATH, encoding='utf-8') as fh:
    template_src = fh.read()

style_start  = template_src.index('<style>')
style_end    = template_src.index('</style>') + len('</style>')
style_block  = template_src[style_start:style_end]

script_start = template_src.rindex('<script>')
script_end   = template_src.rindex('</script>') + len('</script>')
script_block = template_src[script_start:script_end]

# ── Rendering helpers ────────────────────────────────────────────────────────

def _render_conditionals(html: str, vars_: dict) -> str:
    """Expand {{#if VAR}}...{{/if}} — removes block when var is absent/falsy."""
    def _replace(m):
        return m.group(2) if vars_.get(m.group(1)) else ''
    return re.sub(r'\{\{#if (\w+)\}\}(.*?)\{\{/if\}\}', _replace, html, flags=re.DOTALL)


def _render_habit_dots(html: str) -> str:
    """
    Expand G4 habit dot containers.
    Snippet uses: <div style="..." data-filled="5">...</div>
    where data-filled = number of filled (green) dots out of 7.
    """
    def _expand(m):
        try:
            filled = int(m.group(1))
        except ValueError:
            filled = 0
        filled = max(0, min(7, filled))
        dots = ''.join(
            '<div style="width:28px;height:28px;border-radius:4px;background:#34C363;"></div>'
            if i < filled else
            '<div style="width:28px;height:28px;border-radius:4px;background:#3a3a3a;border:1px solid #555;"></div>'
            for i in range(7)
        )
        return f'<div style="display:flex;gap:10px;">{dots}</div>'

    return re.sub(
        r'<div[^>]*\sdata-filled="(\d+)"[^>]*>.*?</div>',
        _expand,
        html,
        flags=re.DOTALL
    )


def render_slide(slide_type: str, vars_: dict) -> str:
    """Load snippet, run all substitutions, return inner slide HTML."""
    snippet_path = os.path.join(SNIPPETS_DIR, f'{slide_type}.html')
    if not os.path.exists(snippet_path):
        print(f'  WARNING: snippet not found: {slide_type}.html', file=sys.stderr)
        return f'<!-- ERROR: snippet {slide_type}.html not found -->'

    with open(snippet_path, encoding='utf-8') as fh:
        html = fh.read()

    # Merge defaults (logo, etc.) under slide vars — slide vars win
    all_vars = {**DEFAULT_VARS, **vars_}

    # 1. Resolve optional blocks
    html = _render_conditionals(html, all_vars)

    # 2. Substitute {{VAR}} tokens — log any missing keys
    def _sub(m):
        key = m.group(1)
        if key not in all_vars:
            print(f'  WARNING [{slide_type}]: missing var {{{{ {key} }}}}', file=sys.stderr)
            return f'<!-- MISSING: {key} -->'
        return all_vars[key]

    html = re.sub(r'\{\{(\w+)\}\}', _sub, html)

    # 3. Expand G4 habit dots (after var substitution so values are numbers)
    if slide_type == 'G4':
        html = _render_habit_dots(html)

    return html


# ── Build slides HTML ────────────────────────────────────────────────────────
section_blocks = []

for slide in slides:
    slide_type = slide['type']
    slide_no   = slide.get('slide_no', len(section_blocks) + 1)
    label      = slide.get('label', '')
    vars_      = slide.get('vars', {})

    inner_html = render_slide(slide_type, vars_)

    block = f'''\
<div class="section">
  <div class="section-header">
    <h2>Slide {slide_no} &mdash; {slide_type} &mdash; {label}</h2>
  </div>
  <div class="slides-grid">
    <div class="slide-item">
      <div class="slide-label"><strong>{slide_type}</strong> &mdash; {label}</div>
      <div class="slide-wrapper">
        <div class="slide">
{inner_html}
        </div>
      </div>
    </div>
  </div>
</div>'''
    section_blocks.append(block)

slides_html = '\n\n'.join(section_blocks)

# ── Output folder ────────────────────────────────────────────────────────────
out_dir  = f'Carousels/data/batch_{BATCH_NO}/{folder_name}'
out_path = f'{out_dir}/carousel.html'
os.makedirs(out_dir, exist_ok=True)

# ── Full HTML page ───────────────────────────────────────────────────────────
full_html = f"""\
<!DOCTYPE html>
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
<p class="page-subtitle">{slide_count} slides &middot; DadFit &middot; Batch {BATCH_NO}</p>

{slides_html}

{script_block}
</body>
</html>"""

with open(out_path, 'w', encoding='utf-8') as fh:
    fh.write(full_html)

file_size = os.path.getsize(out_path)
print(f'Written: {out_path} ({file_size // 1024}KB, {slide_count} slides)', file=sys.stderr)

# ── Result JSON for orchestrator ─────────────────────────────────────────────
result = {
    'uuid':           uuid,
    'running_no':     running_no,
    'folder_name':    folder_name,
    'slide_count':    slide_count,
    'html_path':      out_path,
    'doodle_prompts': doodle_prompts,
}
print(json.dumps(result))
