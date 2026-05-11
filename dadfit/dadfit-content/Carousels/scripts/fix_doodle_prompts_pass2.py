#!/usr/bin/env python3
"""
Second-pass cleanup: remove white ink / style fragments that leaked into
the subject portion of already-rebuilt prompts in doodle_prompts.json.
"""
import json
import re
from pathlib import Path

PROMPTS_FILE = Path(__file__).parent.parent / "data" / "batch_1" / "doodle_prompts.json"

# This marks the start of the new style block appended by fix_doodle_prompts.py
NEW_STYLE_RE = re.compile(r"\. Flat hand-drawn ink-line illustration, #34C363 brand green ink")

# Patterns to strip from the SUBJECT portion only (trailing style junk)
JUNK_PATTERNS = [
    # Any sentence/clause with "white ink" — catch all variants
    # Handles: ", white ink, ...", ". White ink ...", ", drawn in white ink ...", etc.
    re.compile(r"[,\.]?\s+(?:drawn in\s+)?[Ww]hite ink[^.]*\.?", re.IGNORECASE),
    # "#34C363 green and white ink" → strip "and white ink"
    re.compile(r"\s+and\s+white ink", re.IGNORECASE),
    # Trailing ", flat hand-drawn ink-line, white ink, ..." fragments
    re.compile(r",\s+flat hand-drawn ink-line,\s+white ink[^.]*", re.IGNORECASE),
]


def clean_subject(subject: str) -> str:
    for pat in JUNK_PATTERNS:
        subject = pat.sub("", subject)
    return subject.strip().rstrip(".,")


def fix_prompt(prompt: str) -> str:
    m = NEW_STYLE_RE.search(prompt)
    if not m:
        return prompt
    subject = prompt[: m.start()]
    rest = prompt[m.start() :]
    cleaned = clean_subject(subject)
    return cleaned + rest


def main():
    data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} entries")

    changed = 0
    remaining = []

    for entry in data:
        original = entry["prompt"]
        fixed = fix_prompt(original)
        if fixed != original:
            changed += 1
        entry["prompt"] = fixed
        if "white ink" in fixed.lower():
            remaining.append((entry["image_name"], fixed[:200]))

    print(f"Fixed: {changed} prompts")
    print(f"Still contains white ink: {len(remaining)}")
    if remaining:
        for name, snippet in remaining[:5]:
            print(f"  {name}: {snippet}")

    PROMPTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written to {PROMPTS_FILE}")


if __name__ == "__main__":
    main()
