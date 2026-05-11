# Week 54 Overview — Synthetic Data Generation: Magpie and Genie Approaches

This file is your map for Week 54. Read it first; everything else fits inside it.

## The story this week

Before the modern era of strong teacher models, synthetic data was primarily generated via templates — manually crafted rules that substituted values into fixed SQL patterns. This produced large but shallow datasets. The revolution came when models like GPT-4 became strong enough to serve as teachers: you describe a task in natural language, and the teacher generates diverse, novel examples.

## What you need to do

- [ ] OpenAI API key (or Anthropic API key) set in environment variable `OPENAI_API_KEY`
- [ ] PostgreSQL running locally with test schemas loaded (from Week 53's schema list)
- [ ] `openai`, `asyncio`, `httpx`, `psycopg2`, `sqlglot`, `datasketch` installed
- [ ] Week 53 gap analysis (`gap_analysis.md`) open — this drives your skill targets
- [ ] W&B project `week-54-generation` created

Concretely, by the end of the week you should be able to:

- Explain the Magpie self-instruct approach and how it differs from classic Self-Instruct
- Explain the Genie approach to generating grounded, schema-aware synthetic data
- Design teacher model prompts that produce high-quality, diverse PostgreSQL/TimescaleDB SQL pairs
- Build an async generation pipeline that calls the teacher API, validates execution, and saves results
- Target your generation budget toward under-represented skills identified in Week 53's gap analysis

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

- 1h: Read Magpie paper (arXiv 2406.08464) — focus on Section 3 (methodology)
- 1h: Read Genie paper (arXiv 2401.14367) — focus on Section 3 (grounded generation)
- 2h: Write and test teacher prompts (start with 10 examples, verify execution)
- 2.5h: Build the async generation pipeline; run first 1,000 examples
- 0.5h: Log metrics to W&B; commit everything

## Why this week matters

**One-liner:** Ground every teacher prompt in the target schema DDL; validate execution before saving; resume from checkpoints.
