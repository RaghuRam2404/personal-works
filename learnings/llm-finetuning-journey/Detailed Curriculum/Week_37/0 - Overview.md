# Week 37 Overview — Domain-Tuning Sprint Week 1: Build Your 15K SQL Dataset

This file is your map for Week 37. Read it first; everything else fits inside it.

## The story this week

Your Week 33 run used 5K examples and achieved ~35–50% exact match on a simple held-out test. For a production-quality PostgreSQL SQL expert, you need:

## What you need to do

- [ ] `pip install datasets sqlparse anthropic openai` (or skip API packages if using free sources)
- [ ] HuggingFace `datasets` library installed
- [ ] Mac or Colab (no GPU needed for data processing)
- [ ] Optional: Claude/GPT-4o API key for synthetic generation ($10–20 budget recommended)

Concretely, by the end of the week you should be able to:

- Curate and combine SQL datasets from multiple public sources into a unified PostgreSQL-focused dataset
- Generate synthetic training examples using an LLM API (Claude or GPT-4) with controlled prompts
- Design schema-diverse examples that cover PostgreSQL-specific SQL features
- Validate and deduplicate your dataset to ensure quality
- Deliver a clean 15K-example dataset ready for QLoRA fine-tuning in Week 38

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
| Survey and download public datasets | 1h |
| Write data processing and filtering pipeline | 2h |
| Generate synthetic examples (API or from existing synthetic datasets) | 1.5h |
| Run quality checks and deduplication | 1h |
| Hand-craft 20–50 TimescaleDB-specific examples | 1h |
| Format final dataset, verify statistics | 30m |

## Why this week matters

**One-liner:** 15K = 8–10K public + 3–5K synthetic + 50–100 domain-specific. Filter aggressively; diversity > volume.
