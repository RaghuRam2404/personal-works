# Week 75 Overview — Iteration Polish: Trying Different Base Models

This file is your map for Week 75. Read it first; everything else fits inside it.

## The story this week

Your postgres-sqlcoder-7b was built on Qwen2.5-Coder-7B-Instruct. That choice was motivated by: strong SQL benchmarks, permissive Apache 2.0 license, good Unsloth support, and a code-optimized pretraining corpus. But the field moves fast. In the months since you started this course, Llama 3.1 8B, Gemma 2 9B, and DeepSeek-Coder-V2-Lite have all been released or updated. This week you empirically answer: was your base model choice optimal?

## What you need to do

- [ ] RunPod A100-40GB (or 2× A100) provisioned — budget ~$15 for 10 GPU-hours
- [ ] All three candidate models downloaded or accessible via Hub
- [ ] Your v3 SFT dataset at `data/sqlcoder_v3_train` (from Week 58)
- [ ] Custom-200 benchmark at `data/custom_200.json`
- [ ] W&B project `week75-base-model-comparison` created

Concretely, by the end of the week you should be able to:

- Design a controlled experiment that isolates base model quality from fine-tuning quality
- Run your full SFT pipeline on a new base model with minimal code changes
- Compare two fine-tuned models on the same benchmark with the same inference settings
- Explain why the best zero-shot base model is not always the best fine-tuned model
- Produce a model comparison table suitable for inclusion in your technical report

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

- 0.5h: Verify chat templates for each candidate model
- 1.0h: Prepare training data for each model (3 data prep runs)
- 3.0h: Run SFT for each candidate (sequential on one GPU; parallel if you have multiple)
- 1.5h: Evaluate all four models (existing + 3 candidates) on Custom-200
- 1.0h: Write comparison table, analysis, and decision in `results/base_model_comparison.md`

## Why this week matters

**This week in 15 words:** A controlled four-model comparison tells you which base model best fits your pipeline and data.
