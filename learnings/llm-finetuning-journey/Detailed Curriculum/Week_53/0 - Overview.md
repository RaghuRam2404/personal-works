# Week 53 Overview — Dataset v3 Strategy: Quality over Quantity

This file is your map for Week 53. Read it first; everything else fits inside it.

## The story this week

The LIMA paper (Zhou et al., 2023) trained a 65B LLaMA model on exactly 1,000 hand-selected examples and found it matched or exceeded models trained on orders-of-magnitude more data on many benchmarks. The mechanism: most of what a model needs to learn — style, format, safety, helpfulness — is already latent in the pretrained weights. Alignment fine-tuning just surfaces it.

## What you need to do

- [ ] Your v2 dataset is accessible (HuggingFace or local JSONL)
- [ ] Python environment with `datasets`, `pandas`, `numpy` installed
- [ ] A PostgreSQL instance running locally (for schema introspection)
- [ ] Access to Spider and BIRD-SQL test sets (for contamination checking)
- [ ] GitHub repo `llm-finetuning-journey` with Week 53 branch ready

Concretely, by the end of the week you should be able to:

- Articulate why 1,000 carefully curated examples can outperform 100,000 mediocre ones, backed by empirical evidence from LIMA
- Explain Tulu 3's data selection pipeline, including skill coverage, deduplication, and contamination removal
- Design a principled quality strategy for your PostgreSQL/TimescaleDB dataset v3
- Define clear acceptance criteria for what counts as a "high-quality" text-to-SQL example in your domain
- Distinguish between data quantity scaling and data quality scaling, and know when each applies

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

- 2.5h: Read LIMA paper (arXiv 2305.11206) fully — take structured notes per section
- 2h: Read Tulu 3 paper (arXiv 2411.15124) — focus on Sections 2–4 (data pipeline)
- 1h: Audit your existing v2 dataset: count examples by type, schema, difficulty
- 1h: Write your data card for v3 (one Markdown file in your repo)
- 0.5h: Commit the data card + your week notes to GitHub

## Why this week matters

**One-liner:** Quality, diversity, and contamination-free curation beat raw data volume every time.
