# Week 58 Overview — Full SFT on the 50K v3 Dataset

This file is your map for Week 58. Read it first; everything else fits inside it.

## The story this week

At 25K+ examples with a 7B model and LoRA, the key decisions:

## What you need to do

- [ ] v3 dataset on HuggingFace (`<your-handle>/postgres-sql-v3`) accessible
- [ ] CPT checkpoint (`<your-handle>/qwen2.5-coder-7b-postgres-cpt`) accessible
- [ ] RunPod H100 access; budget ~$15–20 for this run
- [ ] Your 200-example custom eval set (PostgreSQL/TimescaleDB benchmark) ready
- [ ] W&B project `week-58-sft` created

Concretely, by the end of the week you should be able to:

- Configure a full SFT run on your v3 dataset starting from the CPT checkpoint
- Choose the right LoRA rank, learning rate, and sequence length for a 25K+ example dataset
- Monitor SFT for overfitting, convergence, and domain coverage throughout training
- Evaluate the SFT checkpoint against your Phase 5 baseline to verify improvement
- Save and push the SFT checkpoint in the correct format for downstream DPO training

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

- 1h: Final data validation — run the full pipeline end-to-end on 10 examples before training
- 0.5h: Configure training script (hyperparameters, logging, checkpoint schedule)
- 0.5h: Smoke test locally (100 steps on 1K examples, verify loss decreases)
- 0.5h: Spin up RunPod H100, upload checkpoint and dataset
- 4h: Run full SFT + monitor (3–4 hours active training)
- 0.5h: Evaluate SFT checkpoint; compare to Phase 5 baseline
- 0.5h: Push to HuggingFace; terminate RunPod; commit code

## Why this week matters

**One-liner:** Start from CPT checkpoint, LoRA rank 32, 2 epochs, completion-only loss, monitor execution accuracy not just val loss.
