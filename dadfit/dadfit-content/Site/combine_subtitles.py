#!/usr/bin/env python3
"""Combine subtitle-like files into paragraphs.

Usage:
  python3 combine_subtitles.py input1.txt input2.txt -o combined.txt

Behavior:
- Skips timestamp lines like "00:00:03:00 - 00:00:06:12" and blank lines.
- Joins remaining text lines into a single paragraph per input file (single-line paragraph).
- Appends paragraphs to the output file separated by a blank line.
"""

import argparse
import re
import sys
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^\s*\d{2}:\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}:\d{2}\s*$")


def extract_paragraph_from_text(text: str) -> str:
    """Return a single paragraph by removing timestamp lines and empty lines, joining remaining lines."""
    lines = text.splitlines()
    pieces = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if TIMESTAMP_RE.match(s):
            continue
        pieces.append(s)
    # Join with single spaces and normalize whitespace
    paragraph = " ".join(pieces)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    return paragraph


def process_files(input_paths, output_path):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    appended = 0

    with out_path.open("a", encoding="utf-8") as out_f:
        for p in input_paths:
            pth = Path(p)
            if not pth.exists():
                print(f"Warning: input file not found: {pth}", file=sys.stderr)
                continue
            text = pth.read_text(encoding="utf-8")
            paragraph = extract_paragraph_from_text(text)
            if paragraph:
                out_f.write(paragraph)
                out_f.write("\n\n")  # separate paragraphs
                appended += 1
            else:
                print(f"Note: no text extracted from {pth}", file=sys.stderr)

    print(f"Appended {appended} paragraph(s) to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Combine subtitle-like files into paragraphs and append to output file.")
    parser.add_argument("inputs", nargs="+", help="Input text files to combine")
    parser.add_argument("-o", "--output", default="combined.txt", help="Output file to append combined paragraphs to")
    args = parser.parse_args()

    process_files(args.inputs, args.output)


if __name__ == "__main__":
    # Optional usage: provide input file paths as a Python list inside this file.
    # If you want to supply the files directly in the script, set INPUT_FILES to a list of paths
    # and optionally set OUTPUT_FILE. If INPUT_FILES is empty, the script falls back to CLI args.
    # Example:
    # INPUT_FILES = ["intro.txt", "part2.txt"]
    # OUTPUT_FILE = "combined.txt"
    INPUT_FILES = ["Intro.txt", "diet.txt", "Exercise.txt", "Lifestyle.txt"]  # <-- put your list of input filenames here if desired
    OUTPUT_FILE = "combined.txt"  # <-- change output filename if using INPUT_FILES

    if INPUT_FILES:
        process_files(INPUT_FILES, OUTPUT_FILE)
    else:
        main()
