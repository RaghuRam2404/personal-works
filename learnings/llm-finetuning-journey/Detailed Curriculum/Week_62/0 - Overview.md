# Week 62 Overview — Comprehensive Eval Part 2: Head-to-Head vs Frontier Models

This file is your map for Week 62. Read it first; everything else fits inside it.

## The story this week

**Your model:** `postgres-sqlcoder-7b-final` (7B parameters, locally deployable)

## What you need to do

- [ ] OpenAI API key (budget ~$15 for this week)
- [ ] Anthropic API key (budget ~$8 for this week)
- [ ] `openai`, `anthropic` Python packages installed
- [ ] SQLCoder-7B downloaded or accessible via HuggingFace
- [ ] DeepSeek-Coder-V2-Lite accessible (via Together AI or local)
- [ ] Week 61 eval harness (`eval_harness.py`) tested and working

Concretely, by the end of the week you should be able to:

- Run a fair, reproducible head-to-head comparison between your model and GPT-4o, Claude 3.5, SQLCoder, and DeepSeek-Coder-V2
- Apply identical evaluation protocols across models to ensure comparability
- Analyze where your model wins and loses relative to each competitor
- Write a head-to-head evaluation section suitable for a technical report
- Compute per-model costs (inference cost per query × accuracy) to make the efficiency argument

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

- 1h: Set up API clients for GPT-4o and Claude; test on 5 examples
- 1h: Run base Qwen2.5-Coder-7B evaluation (same harness, different model path)
- 1h: Run SQLCoder-7B evaluation
- 1h: Run GPT-4o evaluation (200 custom + 100 BIRD — budget: ~$15)
- 1h: Run Claude 3.5 evaluation (200 custom + 100 BIRD — budget: ~$8)
- 0.5h: Run DeepSeek-Coder-V2-Lite (local or via API)
- 1.5h: Compile comprehensive comparison table; per-model error analysis; 5 wins/losses examples
- 0.5h: Write `head_to_head_comparison.md`; commit

## Why this week matters

**One-liner:** Same prompts, same databases, same result comparison for all models; cache all API calls; report CI.
