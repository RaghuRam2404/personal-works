# Week 8 Overview — Phase 1 Gate: Capstone Mini-Project

This file is your map for Week 8. Read it first; everything else fits inside it.

## The story this week

This week is integration, not new material. You are proving that Weeks 1–7 are internalized.

## What you need to do

- [ ] All prior week commits exist in your `llm-finetuning-journey` GitHub repo (Weeks 1–7).
- [ ] HuggingFace account has at least 1 uploaded artifact (Spider tokenized dataset from Week 7).
- [ ] Colab Pro or Free T4 available for training (if using Option A with nanoGPT-size model).
- [ ] W&B project `week-08-capstone` created.
- [ ] Choose your option: Option A (char-level transformer) or Option B (distilgpt2 fine-tuning).

Concretely, by the end of the week you should be able to:

- Synthesize all Phase 1 skills into a single end-to-end project with clean, documented code.
- Train either a char-level transformer (nanoGPT-style) or fine-tune distilgpt2 on Spider SQL data using the full modern training recipe.
- Write a project README that clearly explains dataset, model, training setup, results, and lessons learned.
- Self-assess honestly against the Phase Gate criteria and identify your weakest areas.
- Articulate what PyTorch fluency means and demonstrate it by writing a training loop from memory.

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
| Choose Option A or B, plan your approach | 30 min |
| Set up project structure and data pipeline | 1 h |
| Implement model (copy nanoGPT arch for A; configure Trainer for B) | 1 h |
| Write training loop with full modern recipe | 1.5 h |
| Train model, monitor W&B, generate samples | 1.5 h |
| Write README + journal self-assessment | 1 h |
| Self-test: write training loop from memory (timed) | 30 min |

## Why this week matters

**This week in 15 words:** Phase 1 gate: write the training loop from memory, diagnose loss curves, advance honestly.
