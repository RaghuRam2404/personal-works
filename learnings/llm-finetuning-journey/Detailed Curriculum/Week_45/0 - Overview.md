# Week 45 Overview — DPO on Your Domain Model

This file is your map for Week 45. Read it first; everything else fits inside it.

## The story this week

DPO trains the model to increase the probability of chosen SQL completions relative to the reference model (v1) and decrease the probability of rejected ones. In the SQL domain, the key effects you should observe:

## What you need to do

- [ ] Colab Pro with A100 (40GB) GPU
- [ ] Packages: `unsloth`, `trl>=0.9.0`, `transformers`, `datasets`, `peft`, `wandb`
- [ ] Your model: `postgres-sqlcoder-7b-v1` (or the HF Hub path from Phase 4)
- [ ] Your dataset: `<your-handle>/postgres-sql-preferences-v1` (from Week 44)
- [ ] Held-out test set: 200 prompts with reference SQL (NOT used in preference labeling)
- [ ] Postgres DB accessible from Colab (use `ngrok` or a cloud Postgres instance)

Concretely, by the end of the week you should be able to:

- Apply DPO using Unsloth's DPO trainer to your SFT model with your own preference dataset
- Diagnose and fix common DPO training issues specific to the SQL domain
- Produce a quantitative eval report comparing v1 (SFT only) vs. v2 (SFT + DPO)
- Interpret the reward_margin metric and relate it to downstream SQL execution correctness
- Understand why DPO might not improve on hard queries and what that implies for GRPO

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

- 30 min: Review Week 43 training config and adapt for 7B model + your preference dataset
- 4–5 hours: Run DPO training (this is compute-heavy; Colab Pro A100 needed)
- 1–1.5 hours: Build eval pipeline and run it on v1 and v2
- 30 min: Write eval report comparing v1 vs. v2 across query difficulty tiers

## Why this week matters

**One-liner:** DPO reduces syntax errors and improves easy queries; complex queries need GRPO's online reward.
