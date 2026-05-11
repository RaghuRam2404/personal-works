# Week 50 Overview — Iteration Week 1: Fix Bugs, Expand Dataset, Retry

This file is your map for Week 50. Read it first; everything else fits inside it.

## The story this week

Week 50 continues the curriculum's thread; the Curriculum file explains the conceptual setup in detail.

## What you need to do

- [ ] Week 48 eval report available (v1/v2/v3 comparison)
- [ ] v3 model checkpoint accessible (local or HF Hub)
- [ ] Reward function from Week 47 available
- [ ] RunPod account ready (or Colab Pro for smaller experiments)
- [ ] Week 48 W&B run logs accessible

Concretely, by the end of the week you should be able to:

- Diagnose why your v3 model did not meet eval targets (if applicable) using a structured debugging framework
- Apply specific targeted fixes to the reward function, training data, or hyperparameters based on diagnosis
- Expand your SQL training dataset to better cover complex query types
- Run one or more targeted GRPO experiments to improve on the identified weaknesses
- Produce a documented iteration log with hypothesis, experiment, and result for each change

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

- 1 hour: Diagnose v3 failures using the eval report from Week 48.
- 30 min: Write your iteration hypothesis for the highest-impact fix.
- 1 hour: Implement the fix (reward function change or dataset expansion).
- 3–4 hours async: Run the targeted GRPO experiment on RunPod.
- 1 hour: Evaluate the result. Write the iteration log.
- 30 min: Plan Week 51 based on this week's result.

## Why this week matters

**One-liner:** Every iteration experiment must have a hypothesis; change one thing at a time; start from v3, not v1.
