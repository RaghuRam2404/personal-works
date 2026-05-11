# Week 5 Overview — Optimization, LR Schedules, and Reading Loss Curves

This file is your map for Week 5. Read it first; everything else fits inside it.

## The story this week

**Vanilla SGD:** `θ ← θ - lr * g` where `g = ∇L(θ)`. Simple, but sensitive to the LR and slow in ravines (directions with high curvature).

## What you need to do

- [ ] Your Week 3 CIFAR-10 CNN code is available and working in Colab.
- [ ] W&B project `week-05-optimization` created.
- [ ] Colab with T4 GPU (needed for AMP benchmarking).

Concretely, by the end of the week you should be able to:

- Explain SGD, Momentum, Adam, and AdamW mathematically and state when to prefer each.
- Implement a linear warmup + cosine decay LR schedule from scratch.
- Explain why weight decay in Adam is not equivalent to L2 regularization, and what AdamW fixes.
- Apply gradient clipping correctly and explain why it is needed independently of the optimizer.
- Enable PyTorch AMP (automatic mixed precision) training and verify it speeds up GPU-side compute.
- Read a W&B loss curve and diagnose: LR too high, LR too low, overfitting, gradient explosion.

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
| Read Ruder's optimizer overview (all sections) | 1 h |
| Read AdamW paper intro + algorithm box | 30 min |
| Watch Yannic Kilcher's AdamW explanation (~25m) | 30 min |
| Implement warmup + cosine schedule from scratch | 1 h |
| Add AdamW + clipping + AMP to Week 3 CNN | 1.5 h |
| Generate W&B comparison report | 1 h |
| Journal + commit | 30 min |

## Why this week matters

**This week in 15 words:** AdamW + warmup/cosine + grad clip + AMP is the canonical modern training recipe.
