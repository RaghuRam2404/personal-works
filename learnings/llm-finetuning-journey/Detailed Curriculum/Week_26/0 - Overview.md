# Week 26 Overview — Domain Dataset Construction Part 2: Build the PostgreSQL/TimescaleDB v1 Dataset

This file is your map for Week 26. Read it first; everything else fits inside it.

## The story this week

This week you execute the plan from Week 25. The order of operations matters:

## What you need to do

- [ ] Docker installed and running: `docker run -d -e POSTGRES_PASSWORD=test -p 5432:5432 postgres:16`
- [ ] `pip install psycopg2-binary sqlglot datasketch datasets huggingface_hub`
- [ ] All Week 25 scripts available: `converters.py`, `quality_filter.py`, `self_instruct.py`
- [ ] HuggingFace account with write access (`huggingface-cli login`)
- [ ] Ollama installed (optional, for free local generation): `ollama pull qwen2.5-coder:7b`

Concretely, by the end of the week you should be able to:

- Execute the full 3-tier dataset construction pipeline to produce 5,000 training examples
- Validate every example with SQL parsing and schema consistency checks
- Apply MinHash deduplication to remove near-duplicate examples
- Document and publish a dataset to HuggingFace Hub with a proper dataset card
- Analyze your v1 dataset's coverage gaps and plan v2 improvements

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

- 0.5h: Set up Docker PostgreSQL for local SQL execution verification
- 2.5h: Complete 80 hand-written examples, verify each against PostgreSQL
- 1h: Run full Tier 1 processing (Spider + BIRD conversion + filter)
- 1.5h: Run Self-Instruct generation (can be background if using local Ollama)
- 0.5h: Merge, deduplicate, split, and produce final JSONL files
- 0.5h: Publish to HuggingFace Hub with dataset card
- 0.5h: Commit and journal entry

## Why this week matters

**One-liner:** 3 tiers (Spider+BIRD + hand-written + self-instruct), sqlglot validation, execute on real Postgres, MinHash dedup, push to Hub.
