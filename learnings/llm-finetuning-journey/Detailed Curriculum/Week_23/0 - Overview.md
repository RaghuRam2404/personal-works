# Week 23 Overview — Evaluation 101: Perplexity, lm-evaluation-harness, MMLU, HellaSwag, ARC

This file is your map for Week 23. Read it first; everything else fits inside it.

## The story this week

**Perplexity-based evaluation** (Weeks 21–22):
- Measures how well the model predicts held-out text
- Language-model native: works on any text, no task structure needed
- Lower is better
- Problem: does not tell you if the model can do useful tasks

## What you need to do

- [ ] `pip install lm-eval` (version >= 0.4)
- [ ] Your 50M model checkpoint accessible (locally or on HuggingFace Hub)
- [ ] Colab Pro or RunPod A100 available for running eval (takes 1–2 hours total)
- [ ] `gpt2` model available (HuggingFace will auto-download)

Concretely, by the end of the week you should be able to:

- Explain the difference between perplexity-based evaluation and task-based downstream evaluation
- Install and run `lm-evaluation-harness` on your 50M model and on `gpt2`
- Interpret results from MMLU, HellaSwag, and ARC benchmarks
- Explain why a model can have low perplexity but still fail downstream tasks
- Design a domain-specific evaluation appropriate for your PostgreSQL/TimescaleDB target

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

- 0.5h: Install lm-eval, verify it works on a quick GPT-2 run (5 min sanity check)
- 1.5h: Run full evaluation on GPT-2 (HellaSwag + ARC-E + ARC-C + MMLU-5shot subset)
- 1.5h: Run full evaluation on your 50M model
- 1h: Interpret and compare results; write evaluation table
- 1.5h: Write the comparison report (`week-23-eval-report.md`)
- 0.5h: Commit and journal entry

## Why this week matters

**One-liner:** lm-eval uses log-likelihood to score multiple-choice; MMLU tests facts; HellaSwag tests commonsense; your SQL model needs execution accuracy, not MMLU.
