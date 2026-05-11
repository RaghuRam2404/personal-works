#!/usr/bin/env python3
import json

data = json.load(open("Carousels/data/batch_1/doodle_prompts.json"))
new_tag = "Pure black background #000000. Flat green line art, #34C363 ink. No fill. No shading. No text. No typography. DadFit doodle style."

white_ink = sum(1 for d in data if "white ink" in d["prompt"].lower())
old_bg = sum(1 for d in data if "#1E1E1E" in d["prompt"])
has_new_tag = sum(1 for d in data if new_tag in d["prompt"])
has_green = sum(1 for d in data if "#34C363" in d["prompt"])
atmo = sum(1 for d in data if "Loose spacious line-work" in d["prompt"])
standard = sum(1 for d in data if "70\u201380% of canvas" in d["prompt"])

print(f"Total entries:   {len(data)}")
print(f"White ink refs:  {white_ink}  (should be 0)")
print(f"Old #1E1E1E:     {old_bg}  (should be 0)")
print(f"Has new tag:     {has_new_tag}  (should be 918)")
print(f"Has #34C363:     {has_green}  (should be 918)")
print(f"Atmospheric:     {atmo}")
print(f"Standard:        {standard}")
print()
print("=== SAMPLE NEW PROMPTS ===")
for i in [0, 1, 50, 200, 500, 917]:
    d = data[i]
    print(f'{d["image_name"]}: {d["prompt"][:140]}...')
    print()
