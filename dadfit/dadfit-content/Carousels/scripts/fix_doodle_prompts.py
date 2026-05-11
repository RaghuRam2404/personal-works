#!/usr/bin/env python3
"""
Rewrite all doodle_prompts.json entries to the new standard:
- Pure black background #000000
- #34C363 green ink ONLY
- New mandatory closing tag
- Clean subject extraction from old boilerplate
"""
import json
import re
import shutil
from pathlib import Path

PROMPTS_FILE = Path(__file__).parent.parent / "data" / "batch_1" / "doodle_prompts.json"

OLD_TAGS = [
    "Solid dark background #1E1E1E. Flat line art. No shading. No text. No typography. DadFit doodle style.",
    " White ink line art on dark background #1E1E1E. Flat line art. No shading. No text. No typography. DadFit doodle style.",
    "3 green ink line art on dark background #1E1E1E. Flat line art. No shading. No text. No typography. DadFit doodle style.",
]

NEW_TAG = "Pure black background #000000. Flat green line art, #34C363 ink. No fill. No shading. No text. No typography. DadFit doodle style."

# Pattern: detect where style block starts (after the subject description)
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

# Pattern: detect old closing tag tail (fallback)
OLD_TAIL = re.compile(
    r"\.\s+(?:Solid dark background|on dark background|White ink line art on dark|3 green ink line art on dark)"
)


def strip_old_tag(text: str) -> str:
    for tag in OLD_TAGS:
        if tag in text:
            return text[: text.index(tag)].strip()
    # fallback
    m = OLD_TAIL.search(text)
    if m:
        return text[: m.start()].strip()
    return text.strip()


def extract_subject(prompt: str) -> str:
    clean = strip_old_tag(prompt)
    m = STYLE_SPLIT.search(clean)
    if m:
        return clean[: m.start()].strip().rstrip(".")
    return clean.rstrip(".")


def is_atmospheric(text: str) -> bool:
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


def rebuild_prompt(entry: dict) -> str:
    prompt = entry["prompt"]
    subject = extract_subject(prompt)

    if is_atmospheric(prompt) or is_atmospheric(subject):
        return (
            f"{subject}. "
            f"Flat hand-drawn ink-line illustration, #34C363 brand green ink, ~3\u20135px stroke, no fill. "
            f"Loose spacious line-work. "
            f"{NEW_TAG}"
        )
    else:
        return (
            f"{subject}. "
            f"Flat hand-drawn ink-line illustration, #34C363 brand green ink, ~3\u20135px stroke, no fill. "
            f"Subject fills ~70\u201380% of canvas, centered. "
            f"{NEW_TAG}"
        )


def main():
    data = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} entries")

    # Backup
    backup = PROMPTS_FILE.with_suffix(".json.bak")
    shutil.copy(PROMPTS_FILE, backup)
    print(f"Backed up to {backup}")

    # Dry-run check on first 10
    print("\n--- DRY RUN (first 5) ---")
    for entry in data[:5]:
        new_prompt = rebuild_prompt(entry)
        atmo = is_atmospheric(entry["prompt"])
        print(f"{entry['image_name']} [atmo={atmo}]:")
        print(f"  OLD: {entry['prompt'][:100]}...")
        print(f"  NEW: {new_prompt[:120]}...")
        print()

    confirm = input("Proceed with full rewrite? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Rewrite
    problems = []
    for entry in data:
        old_prompt = entry["prompt"]
        new_prompt = rebuild_prompt(entry)
        entry["prompt"] = new_prompt

        # Sanity checks
        if "#1E1E1E" in new_prompt:
            problems.append((entry["image_name"], "still has #1E1E1E"))
        if "White ink" in new_prompt or "white ink" in new_prompt:
            problems.append((entry["image_name"], "still has White ink"))
        if NEW_TAG not in new_prompt:
            problems.append((entry["image_name"], "missing new tag"))

    if problems:
        print(f"\nWARNING: {len(problems)} issues found:")
        for img, issue in problems[:10]:
            print(f"  {img}: {issue}")

    PROMPTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(data)} updated prompts to {PROMPTS_FILE}")

    # Stats
    atmo_count = sum(1 for d in data if is_atmospheric(d["prompt"]) or "Loose spacious" in d["prompt"])
    print(f"Atmospheric prompts: {atmo_count}")
    print(f"Standard prompts: {len(data) - atmo_count}")
    print(f"Problems: {len(problems)}")


if __name__ == "__main__":
    main()
