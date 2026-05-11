#!/usr/bin/env python3
"""
process_doodles.py — DadFit Doodle Post-Processor

For every PNG in Carousels/data/batch_{batch_no}/doodles/:
  1. Detect background by sampling 10×10 pixels at all four corners
  2. Make every pixel within `threshold` color-distance of the bg → fully transparent
  3. Recolor remaining (ink) pixels to brand green #34C363

Usage:
  python3 process_doodles.py                          # batch 1
  python3 process_doodles.py --batch 2
  python3 process_doodles.py --threshold 40           # looser bg removal
  python3 process_doodles.py --dry-run                # preview without saving
"""

import argparse
import math
import os
import sys
import numpy as np
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────

BRAND_GREEN  = (0x34, 0xC3, 0x63)   # #34C363
SAMPLE_SIZE  = 10                    # corner sample area (pixels)
DEFAULT_THRESHOLD = 30               # Euclidean RGB distance cutoff

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Carousels/

# ── Background detection (corner sampling) ────────────────────────────────────

def detect_bg_color(arr: np.ndarray) -> tuple:
    """
    Average the SAMPLE_SIZE×SAMPLE_SIZE pixel corners of the image.
    Returns (r, g, b) as plain ints.
    """
    h, w, _ = arr.shape
    sz = min(SAMPLE_SIZE, h, w)
    samples = np.concatenate([
        arr[:sz,   :sz  ],           # top-left
        arr[:sz,   w-sz:],           # top-right
        arr[h-sz:, :sz  ],           # bottom-left
        arr[h-sz:, w-sz:],           # bottom-right
    ]).reshape(-1, 3)
    mean = samples.mean(axis=0)
    return (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))


# ── Core processing ───────────────────────────────────────────────────────────

def process_image(path: str, threshold: int, dry_run: bool = False) -> dict:
    """
    Load image, remove background, recolor ink to brand green.
    Overwrites the original file as RGBA PNG.
    """
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32)   # (h, w, 3)
    h, w, _ = arr.shape

    # 1. Detect background from corners
    bg = detect_bg_color(arr.astype(np.uint8))
    bg_arr = np.array(bg, dtype=np.float32)  # shape (3,)

    # 2. Euclidean distance of each pixel from background color
    diff = arr - bg_arr                      # (h, w, 3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (h, w)

    # 3. Build RGBA output
    out = np.zeros((h, w, 4), dtype=np.uint8)

    is_bg  = dist <= threshold               # True where pixel ≈ background
    is_ink = ~is_bg

    # Background → fully transparent
    out[is_bg, 3] = 0

    # Ink → brand green, fully opaque
    out[is_ink, 0] = BRAND_GREEN[0]
    out[is_ink, 1] = BRAND_GREEN[1]
    out[is_ink, 2] = BRAND_GREEN[2]
    out[is_ink, 3] = 255

    result = Image.fromarray(out, "RGBA")

    ink_pixels   = int(is_ink.sum())
    total_pixels = h * w
    bg_hex       = "#{:02X}{:02X}{:02X}".format(*bg)

    if not dry_run:
        result.save(path)

    return {
        "bg_color":  bg_hex,
        "ink_pct":   round(ink_pixels / total_pixels * 100, 1),
        "changed":   ink_pixels,
        "total":     total_pixels,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process DadFit doodle images")
    parser.add_argument("--batch",     type=int, default=1,
                        help="Batch number (default: 1)")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"Color distance cutoff for bg removal (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Preview only — do not save files")
    args = parser.parse_args()

    doodles_dir = os.path.join(BASE_DIR, "data", f"batch_{args.batch}", "doodles")

    if not os.path.isdir(doodles_dir):
        print(f"ERROR: Doodles folder not found: {doodles_dir}")
        sys.exit(1)

    pngs = sorted(f for f in os.listdir(doodles_dir) if f.lower().endswith(".png"))

    if not pngs:
        print(f"No PNG files found in: {doodles_dir}")
        sys.exit(0)

    print(f"\n  DadFit Doodle Processor")
    print(f"  Batch     : {args.batch}")
    print(f"  Folder    : {doodles_dir}")
    print(f"  Files     : {len(pngs)}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Ink color : #34C363 (brand green)")
    print(f"  Dry run   : {'YES (no files saved)' if args.dry_run else 'NO (files will be overwritten)'}")
    print()

    ok = 0
    for fname in pngs:
        fpath = os.path.join(doodles_dir, fname)
        try:
            stats  = process_image(fpath, args.threshold, dry_run=args.dry_run)
            status = "DRY " if args.dry_run else "OK  "
            print(f"  {status} {fname:20s}  bg={stats['bg_color']}  ink={stats['ink_pct']}%")
            ok += 1
        except Exception as e:
            print(f"  FAIL {fname}: {e}")

    print(f"\n  Done. {ok}/{len(pngs)} images {'previewed' if args.dry_run else 'processed'}.\n")


if __name__ == "__main__":
    main()
