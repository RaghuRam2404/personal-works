---
name: step10-html-to-images
description: "Converts all HTML_APPROVED carousels in a batch to 1080×1080 PNG slide images using a headless browser. For each approved carousel: opens carousel.html, screenshots every slide, saves to slides/ folder, updates DB to IMAGES_CREATED."
argument-hint: "Batch number (e.g. '1' for batch 1)"
---

# Step 10 — HTML → PNG Slide Images

Renders each slide of every `HTML_APPROVED` carousel in the batch at full 1080×1080 px resolution, saves numbered PNGs, and advances the stage to `IMAGES_CREATED`.

## Inputs & Outputs

| | Path |
|---|---|
| **Input HTML** | `Carousels/data/batch_{N}/{folder_name}/carousel.html` |
| **Doodle images** | `Carousels/data/batch_{N}/doodles/{running_no}-d-NN.png` |
| **Logo** | `Resources/Images/logo.png` |
| **Output slides** | `Carousels/data/batch_{N}/{folder_name}/slides/slide-01.png … slide-NN.png` |
| **DB stage after** | `IMAGES_CREATED` |

> **Status filter:** Only carousels with `current_stage = HTML_APPROVED` are processed.

---

## Procedure

### Step 1 — Confirm the batch number

Determine `{N}` from the argument or conversation context.

### Step 2 — Check Puppeteer is available

Run from the workspace root (`dadfit-content/`):

```bash
node -e "require('puppeteer')" 2>/dev/null && echo "ok" || echo "missing"
```

If missing, install it (use `--ignore-scripts` to skip bundled Chromium download — the script uses system Chrome):

```bash
cd Carousels/skills/step10-html-to-images && npm install --ignore-scripts
```

### Step 3 — Run the export script

```bash
node "Carousels/skills/step10-html-to-images/scripts/export_batch.js" --batch {N} --workspace "{workspace-root}"
```

Replace:
- `{N}` with the batch number (e.g. `1`)
- `{workspace-root}` with the absolute path to the `dadfit-content` folder

**Example:**

```bash
node "Carousels/skills/step10-html-to-images/scripts/export_batch.js" --batch 1 --workspace "/Users/raghu-2264/Raghu/Personal Works/dadfit/dadfit-content"
```

The script will:
1. Open `Carousels/data/db.sqlite` and query all `HTML_APPROVED` rows for the batch
2. Skip any carousel whose `slides/` folder already has the expected number of PNGs
3. For each carousel:
   - Launch a headless Chromium browser
   - Load `carousel.html` via `file://` (all relative image paths — doodles, logo — resolve automatically)
   - Wait for fonts and images to load (`networkidle2`)
   - For each `.slide-wrapper`: remove the CSS `scale(0.45)` transform, expand to 1080×1080, screenshot
   - Save `slide-01.png`, `slide-02.png`, … to `{carousel_dir}/slides/` (always overwrites existing PNGs)
   - Update DB: `current_stage = IMAGES_CREATED`
4. Print a summary on completion

### Step 4 — Verify output

```bash
ls -1 "Carousels/data/batch_{N}/<folder_name>/slides/"
```

Spot-check 3–5 carousels to confirm correct slide count and image dimensions (should be 1080×1080).

### Step 5 — Report to user

List total carousels processed and any that were skipped or failed.

---

## Image Path Notes

The carousels use two image sources — both are already correct relative paths inside `carousel.html`:

| Image type | HTML `src` attribute | Resolves to |
|---|---|---|
| Doodles | `../doodles/{N}-d-NN.png` | `batch_{N}/doodles/` |
| Logo | `../../../../Resources/Images/logo.png` | `dadfit-content/Resources/Images/` |

No patching is required. The `file://` URL loader resolves both paths correctly.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot find module 'puppeteer'` | Run `npm install --ignore-scripts` inside `Carousels/skills/step10-html-to-images/` |
| `Cannot find module 'sql.js'` | Run `npm install` inside `Carousels/skills/step10-html-to-images/` |
| Doodle images missing / blank | Confirm PNGs exist in `Carousels/data/batch_{N}/doodles/` before running |
| Fonts look wrong | Ensure network is available (Puppeteer fetches Google Fonts via CDN) |
| Blank / black slides | Increase `networkidle2` timeout in script (`timeout: 60000`) |
| Wrong slide count | Check HTML: `grep -c 'slide-wrapper' carousel.html` |
| Chrome not found | Update `executablePath` in the script to your Chrome installation path |
