# Week 22 Overview — Evaluate the 50M LM: Perplexity, Samples, and Writeup

This file is your map for Week 22. Read it first; everything else fits inside it.

## The story this week

Perplexity is the exponential of the average negative log-likelihood per token:

## What you need to do

- [ ] Best checkpoint from Week 21 available locally or downloaded from HuggingFace Hub
- [ ] `val.bin` available (from Week 20 data pipeline)
- [ ] Model can be loaded with `torch.load` and runs a forward pass
- [ ] W&B run from Week 21 is accessible for reference

Concretely, by the end of the week you should be able to:

- Compute validation perplexity correctly using log-likelihood accumulation
- Generate text samples with temperature, top-k, and top-p sampling
- Critically analyze generated samples to identify failure modes
- Write a structured model evaluation report
- Recognize what poor perplexity means and diagnose causes

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

- 1h: Implement and validate perplexity computation
- 1h: Implement text generation with temperature + top-k sampling
- 1.5h: Generate 15–20 samples across various prompts and temperatures
- 2h: Write the evaluation report (`week-22-evaluation.md`)
- 0.5h: Upload checkpoint to HuggingFace (if not done in Week 21)
- 0.5h: Commit and write journal entry

## Why this week matters

**One-liner:** Perplexity = exp(mean CE loss); diagnose with samples; your 50M model is not your fine-tuning base — Qwen2.5-Coder-7B is.
