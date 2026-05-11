# Week 51 Overview — Iteration Week 2: Pick Your Best Model

This file is your map for Week 51. Read it first; everything else fits inside it.

## The story this week

Iteration without a stopping criterion leads to overfitting to the eval set. The stopping rules:

## What you need to do

- [ ] Week 50 iteration results (v3-iter1 model, iteration log)
- [ ] Second failure mode hypothesis from Week 50 diagnosis.md
- [ ] All previous model checkpoints accessible (v1, v2, v3, v3-iter1)
- [ ] 200-query held-out eval set (the same one used throughout Phase 5)
- [ ] Remaining RunPod budget: ~$15–25

Concretely, by the end of the week you should be able to:

- Continue the targeted iteration process from Week 50 on the second-priority failure mode
- Run a final comparative evaluation across all models (v1, v2, v3, v3-iter1, and any new checkpoints)
- Select the best model checkpoint based on a clear, multi-metric decision framework
- Freeze the model that will be presented at the Phase 5 Gate (Week 52)
- Produce a final Phase 5 eval report that tells the complete story of v1→v2→v3

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

- 30 min: Review Week 50 results. Decide what Week 51 experiment addresses.
- 30 min: Write the Week 51 hypothesis and experiment plan.
- 3–4 hours async: Run the second targeted GRPO experiment.
- 1 hour: Run final evaluation on all models (v1, v2, v3, v3-iter1, v3-iter2).
- 1 hour: Write the final Phase 5 eval report.
- 30 min: Select the best model, push it to HF Hub with a clear version tag.

## Why this week matters

**One-liner:** Stop when diminishing returns dominate; pick the model with best semantic accuracy among those meeting the Gate criterion.
