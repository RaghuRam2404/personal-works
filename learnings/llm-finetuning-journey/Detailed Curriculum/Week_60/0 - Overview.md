# Week 60 Overview — GRPO with Executable Rewards (Final)

This file is your map for Week 60. Read it first; everything else fits inside it.

## The story this week

Your reward function must balance multiple objectives. A multi-signal reward works better than binary correctness alone:

## What you need to do

- [ ] DPO-v3 checkpoint: `<your-handle>/qwen2.5-coder-7b-postgres-dpo-v3`
- [ ] PostgreSQL with all test schemas running
- [ ] RunPod H100 access; budget ~$12 for this run
- [ ] `trl>=0.8` with GRPOTrainer; Unsloth
- [ ] 200-example custom eval set (plus 100-example held-out for honest eval)
- [ ] W&B project `week-60-grpo` created

Concretely, by the end of the week you should be able to:

- Configure and run the final GRPO training step starting from your DPO-v3 checkpoint
- Design a multi-signal reward function that combines execution correctness, result accuracy, and SQL quality signals
- Tune GRPO group size, reward scale, and KL coefficient for the SQL domain
- Verify that GRPO improves over the DPO baseline without catastrophic regression
- Produce the capstone model: `postgres-sqlcoder-7b-final`

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

- 1h: Finalize reward function; run on 100 examples to verify it works correctly
- 0.5h: Select 1,500 diverse GRPO training prompts from your eval distribution
- 0.5h: Configure GRPO script; smoke test on 50 prompts locally
- 0.5h: Spin up RunPod H100; upload
- 3.5h: Run GRPO on RunPod (~3.5 hours); monitor W&B
- 0.5h: Evaluate final model; compare to DPO baseline
- 0.5h: Merge adapters; push `postgres-sqlcoder-7b-final` to HuggingFace; terminate RunPod

## Why this week matters

**One-liner:** GRPO needs reward variance within each group — select prompts where the model sometimes fails, always use partial credit rewards.
