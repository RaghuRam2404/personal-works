# Week 59 Overview — DPO on a Refreshed Preference Dataset

This file is your map for Week 59. Read it first; everything else fits inside it.

## The story this week

**Strategy:** Sample prompts from your v3 dataset. For each, generate:
- **Chosen:** Your best available answer — either the v3 training example (teacher-generated + filtered) or, if your SFT-v3 model gets it right, the SFT model's output
- **Rejected:** Generate 4–8 SQL candidates using your SFT-v3 model with temperature 0.8 (stochastic). Select the one that is most wrong in a meaningful way (not just a syntax error)

## What you need to do

- [ ] SFT-v3 checkpoint accessible: `<your-handle>/qwen2.5-coder-7b-postgres-sft-v3`
- [ ] PostgreSQL with test schemas running
- [ ] Colab Pro with A100 (or RunPod A100 if needed — ~$4 for 2 hours)
- [ ] `trl` library with DPOTrainer: `pip install trl>=0.8`
- [ ] W&B project `week-59-dpo` created

Concretely, by the end of the week you should be able to:

- Build a refreshed, high-quality 5K SQL preference dataset using your v3 SFT model as the policy
- Apply DPO training starting from your SFT-v3 checkpoint
- Tune DPO's beta hyperparameter for the SQL execution domain
- Detect and handle the DPO loss-going-negative problem
- Evaluate whether DPO improved execution accuracy vs. the SFT baseline

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

- 1.5h: Build 5K preference pairs (generate candidates with SFT-v3, execute, label)
- 0.5h: Audit 50 random pairs — verify "hard" criterion is met
- 1h: Configure and test DPO training script (100-step smoke test)
- 2.5h: Run DPO on Colab Pro (5K pairs at LoRA DPO fits on A100)
- 0.5h: Evaluate DPO checkpoint vs SFT-v3
- 0.5h: Push checkpoint; commit; log W&B

## Why this week matters

**One-liner:** Hard on-policy pairs with execution-based labels; beta=0.2; stop before loss goes deeply negative.
