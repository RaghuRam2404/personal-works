# Week 34 Overview — Unsloth: The Speed Unlock

This file is your map for Week 34. Read it first; everything else fits inside it.

## The story this week

Your Week 33 training ran on A100 and took 25–45 minutes for 5K examples. For 15K examples (the Week 38 sprint), that's ~75–135 minutes. Several sources of inefficiency in vanilla HuggingFace QLoRA:

## What you need to do

- [ ] Colab Pro with A100 runtime
- [ ] Install Unsloth: `pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"` (or use their pinned pip release)
- [ ] Vanilla QLoRA script from Week 33 saved and ready
- [ ] Same 5K training examples and 100-example test set from previous weeks
- [ ] W&B project `week-34-unsloth` created

Concretely, by the end of the week you should be able to:

- Explain the core optimizations Unsloth applies to make LoRA/QLoRA training 2–5x faster
- Re-run your Week 33 QLoRA training with Unsloth and empirically verify the speedup
- Compare training time, VRAM usage, and final model quality between vanilla HuggingFace QLoRA and Unsloth
- Configure Unsloth for Qwen2.5-Coder-7B fine-tuning
- Understand which optimizations Unsloth provides and what their limitations are

## Suggested order through the files

1. **1 - Curriculum.md** — read first. Concepts, connections, common pitfalls.
2. **2 - Resources.md** — papers, videos, blog posts, repos, docs to consult while studying.
3. **3 - Assignment.md** — the hands-on task you must complete this week.
4. **4 - AssignmentSolutions.md** — only after you have attempted the assignment yourself.
5. **5 - Quiz.md** — self-test that you have absorbed the week's material.
6. **6 - Answers.md** — quiz answers and explanations; check after attempting.
7. **7 - Glossary.md** — terminology used this week, indexed for fast lookup.
8. **8 - TakeAway.md** — the one-page summary you keep for the rest of the course.

## Time budget

| Activity | Time |
|---|---|
| Read Unsloth README and relevant blog posts | 1h |
| Convert Week 33 training script to Unsloth | 1h |
| Run vanilla QLoRA (from Week 33 script) and record baseline metrics | 1.5h |
| Run same training with Unsloth, record metrics | 1.5h |
| Write comparison report | 1h |
| Commit to GitHub | 30m |

## Why this week matters

**One-liner:** Unsloth = fused LoRA kernels + custom GC + RoPE optimization → 2–5x faster, 40–60% less VRAM. Same model quality.
