# SKILL: Step 9 — Doodle Processor

## Purpose

Process all raw AI-generated doodle PNG images for a batch:
- Remove background (most common pixel color → transparent)
- Recolor all ink to DadFit brand green `#34C363`
- Preserve anti-aliased edges via alpha mapping

Run this **after** doodle images are placed in the doodles folder and **before** approving carousels in the web viewer.

---

## When to Run

- Stage trigger: doodle images have been generated and dropped into `Carousels/data/batch_{batch_no}/doodles/`
- Run once per batch
- Safe to re-run (idempotent if images are already processed, but re-processing a processed image will re-detect the new dominant color, so keep originals if unsure)

---

## Command

```bash
cd Carousels
python3 scripts/process_doodles.py --batch 1
```

For dry-run (preview without saving):
```bash
python3 scripts/process_doodles.py --batch 1 --dry-run
```

---

## How It Works

### Algorithm

For each `*.png` in `Carousels/data/batch_{batch_no}/doodles/`:

1. **Detect background** — find the most common pixel color using `numpy.unique` (handles any bg color: black, white, grey, etc.)
2. **Compute alpha** — for each pixel, Euclidean distance from background color → normalize to 0–255
   - Pure background pixel → alpha 0 (fully transparent)
   - Pure ink pixel → alpha 255 (fully opaque)
   - Anti-aliased edge → proportional alpha (smooth)
3. **Recolor** — set all pixels to `#34C363` with the computed alpha
4. **Save** — overwrite original as RGBA PNG

### Formula

$$\text{alpha} = \text{clamp}\left(\frac{\sqrt{(r-r_{bg})^2 + (g-g_{bg})^2 + (b-b_{bg})^2}}{\sqrt{3} \times 255} \times 255,\ 0,\ 255\right)$$

---

## Output

- Files overwritten in-place: `Carousels/data/batch_{batch_no}/doodles/{running_no}-d-{NN}.png`
- Format: RGBA PNG (transparent background, `#34C363` ink)
- Console shows per-file: detected bg color + ink coverage %

---

## Naming Convention (reminder)

Doodle files must be named exactly:

```
{running_no}-d-{slide_no:02d}.png
```

Examples: `1-d-01.png`, `7-d-03.png`, `100-d-11.png`

These names map to the `image_name` field in `doodle_prompts.json` and are referenced directly in `carousel.html` via `src="../doodles/{name}"`.

---

## Dependencies

- Python 3.x
- `Pillow` and `numpy` (installed in the project venv — `source Carousels/.venv/bin/activate`)

---

## Script Location

`Carousels/scripts/process_doodles.py`
