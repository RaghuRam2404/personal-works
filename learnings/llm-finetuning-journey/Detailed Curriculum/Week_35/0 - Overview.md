# Week 35 Overview — Hyperparameter Tuning for SFT/LoRA

This file is your map for Week 35. Read it first; everything else fits inside it.

## The story this week

For LoRA/QLoRA SFT, roughly ranked by impact:

## What you need to do

- [ ] Colab Pro (A100 or T4)
- [ ] 1K training examples and 200 eval examples (subset of your domain dataset)
- [ ] W&B project `week-35-hp-sweep` created
- [ ] Raschka's article read: https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms
- [ ] Unsloth installed (from Week 34)

Concretely, by the end of the week you should be able to:

- Articulate what each major SFT/LoRA hyperparameter controls and how it affects loss curves
- Run a W&B sweep over LR, rank, and alpha on a 1K example subset
- Interpret sweep results to choose the best hyperparameter configuration for your 15K dataset (Week 38)
- Understand the interaction between learning rate, batch size, and gradient accumulation
- Apply practical heuristics for hyperparameter selection without exhaustive search

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
| Read Sebastian Raschka's LoRA insights article (fully) | 1.5h |
| Design sweep: choose which hyperparameters to sweep, define config | 30m |
| Write W&B sweep script | 1h |
| Run sweep (12 runs × 5–10 min each on 1K subset) | 2h |
| Analyze results, write recommendation | 1.5h |
| Commit to GitHub | 30m |

## Why this week matters

**One-liner:** LR is king. Fix alpha=2r, target all linear layers, sweep LR first, use early stopping always.
