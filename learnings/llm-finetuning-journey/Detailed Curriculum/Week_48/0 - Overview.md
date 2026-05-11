# Week 48 Overview — GRPO Sprint Week 2: Run It

This file is your map for Week 48. Read it first; everything else fits inside it.

## The story this week

The 7B model with GRPO (K=8) requires approximately 35–45GB VRAM. Unsloth's memory optimization reduces this to ~20–30GB with 4-bit quantization. An A100 80GB gives you comfortable headroom. An A100 40GB works with aggressive memory optimization.

## What you need to do

- [ ] RunPod account with payment method (A100 80GB recommended; 40GB works with 4-bit)
- [ ] RunPod pod started: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image
- [ ] Week 47 training script (`grpo_train.py`) uploaded to RunPod via SSH/scp
- [ ] Week 47 reward function (`reward_fn.py`) uploaded to RunPod
- [ ] W&B API key set: `export WANDB_API_KEY=...`
- [ ] HF token set: `huggingface-cli login`
- [ ] Your training prompt dataset with reference SQL loaded on RunPod

Concretely, by the end of the week you should be able to:

- Run a complete GRPO training job on RunPod A100 using Unsloth
- Monitor GRPO-specific metrics (mean reward, reward std, group advantages) in W&B
- Diagnose and fix GRPO training issues in real time
- Evaluate the resulting v3 model against v1 and v2 on your SQL benchmark
- Produce a trained `postgres-sqlcoder-7b-v3-grpo` that outperforms v2-dpo on complex queries

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

- 1 hour: Set up RunPod instance, install dependencies, upload scripts
- 30 min: Verify one training step completes without error
- 5–8 hours async: Let training run while you work on other tasks. Check W&B every 30–60 minutes.
- 1 hour: After training completes, push model to HF Hub, run eval pipeline
- 30 min: Write eval report comparing v1, v2, v3

## Why this week matters

**One-liner:** Monitor mean_reward and reward_std together; flat reward_std means zero gradient, not convergence.
