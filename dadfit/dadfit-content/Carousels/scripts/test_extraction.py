#!/usr/bin/env python3
"""Test subject extraction logic on sample entries."""
import json
import re

OLD_TAGS = [
    "Solid dark background #1E1E1E. Flat line art. No shading. No text. No typography. DadFit doodle style.",
    " White ink line art on dark background #1E1E1E. Flat line art. No shading. No text. No typography. DadFit doodle style.",
    "3 green ink line art on dark background #1E1E1E. Flat line art. No shading. No text. No typography. DadFit doodle style.",
]

STYLE_SPLIT = re.compile(
    r"[,\.]?\s+(?="
    r"[Ww]hite ink on (?:solid )?dark background|"
    r"#34C363 green ink[,\s~]|"
    r"Flat hand-drawn ink-line[,\s](?!doodle\s)|"
    r"Flat hand-drawn ink-line style|"
    r"[Ss]ubject fills|"
    r"[Ss]tyle:\s+flat"
    r")"
)
OLD_TAIL = re.compile(
    r"\.\s+(?:Solid dark background|on dark background|White ink line art on dark|3 green ink line art on dark)"
)


def strip_old_tag(text):
    for tag in OLD_TAGS:
        if tag in text:
            return text[: text.index(tag)].strip()
    m = OLD_TAIL.search(text)
    return text[: m.start()].strip() if m else text.strip()


def extract_subject(prompt):
    clean = strip_old_tag(prompt)
    m = STYLE_SPLIT.search(clean)
    return clean[: m.start()].strip().rstrip(".") if m else clean.rstrip(".")


def is_atmospheric(text):
    tl = text.lower()
    return any(
        p in tl
        for p in [
            "wide flat hand-drawn atmospheric",
            "wide atmospheric scene",
            "atmospheric scene:",
            "panoramic",
        ]
    )


data = json.load(open("Carousels/data/batch_1/doodle_prompts.json"))

print("=== SAMPLES ===")
for i in [0, 1, 2, 50, 100, 200, 500, 800]:
    d = data[i]
    subj = extract_subject(d["prompt"])
    atmo = is_atmospheric(d["prompt"])
    print(f'{d["image_name"]} atmo={atmo}:')
    print(f"  {subj[:120]}")
print()
print("=== EDGE CASES ===")
specials = [d for d in data if "#FF6B6B" in d["prompt"] or d["prompt"].startswith("Corner")]
for d in specials[:5]:
    subj = extract_subject(d["prompt"])
    print(f'{d["image_name"]}: {subj[:120]}')
