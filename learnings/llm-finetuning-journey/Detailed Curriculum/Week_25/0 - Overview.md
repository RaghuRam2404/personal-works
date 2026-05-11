# Week 25 Overview — Domain Dataset Construction: Formats, Self-Instruct, and Planning

This file is your map for Week 25. Read it first; everything else fits inside it.

## The story this week

The same data can be stored in multiple formats. Most modern fine-tuning frameworks expect the data in a specific structure. If your data is in the wrong format, your training will silently produce garbage (no error, but the model learns the wrong thing).

## What you need to do

- [ ] `pip install datasets sqlglot datasketch openai` (or `anthropic` if using Claude)
- [ ] Spider dataset available: `load_dataset("spider")`
- [ ] BIRD dataset available: check [bird-bench.github.io](https://bird-bench.github.io/) for download instructions
- [ ] GitHub repo with `postgres-sql-v1/` directory created

Concretely, by the end of the week you should be able to:

- Explain the Alpaca, ShareGPT, and ChatML dataset formats and convert between them
- Implement Self-Instruct to generate synthetic instruction-response pairs
- Plan the full PostgreSQL/TimescaleDB dataset v1 (5K examples)
- Write data curation scripts that enforce schema consistency and quality filters
- Identify the three sources you will use for your dataset and their trade-offs

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

- 1h: Read Self-Instruct paper (Sections 1–3); read Alpaca and ShareGPT format docs
- 1h: Download Spider and BIRD datasets; examine the format; write conversion scripts
- 2h: Write data pipeline: format conversion, quality filters, deduplication
- 1.5h: Write 20 hand-crafted PostgreSQL/TimescaleDB examples
- 1h: Set up Self-Instruct generation script (do not run the full generation yet — that is Week 26)
- 0.5h: Commit and document your dataset plan in `dataset_plan.md`

## Why this week matters

**One-liner:** ChatML format, loss on assistant turn only, 3-tier dataset (Spider+BIRD → hand-written → self-instruct), SQL validity via sqlglot.
