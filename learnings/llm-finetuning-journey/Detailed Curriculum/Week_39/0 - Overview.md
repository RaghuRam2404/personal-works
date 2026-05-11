# Week 39 Overview — Domain-Tuning Sprint Week 3: Execution-Based Evaluation

This file is your map for Week 39. Read it first; everything else fits inside it.

## The story this week

Exact match is a proxy metric. Two SQL queries can return identical rows while being textually different:

## What you need to do

- [ ] Colab Pro notebook open with GPU runtime (T4 or A100)
- [ ] Week 38 model adapter saved locally or on HuggingFace Hub (`<your-handle>/postgres-sqlcoder-7b-v1`)
- [ ] `held_out_test.json` from Week 32 (100 examples, never used in training)
- [ ] Python packages: `psycopg2-binary`, `sqlparse`, `datasets`, `transformers`, `peft`
- [ ] PostgreSQL installed (Option A) or Docker available (Option B)
- [ ] GitHub repo open for committing the harness

Concretely, by the end of the week you should be able to:

- Spin up a PostgreSQL database inside a Colab notebook using Docker
- Execute model-generated SQL against live Postgres and compare result sets
- Implement an execution-based evaluation harness from scratch (or adapting sql-eval)
- Compare your model against the base model and against GPT-4o/Claude on execution correctness
- Interpret execution correctness as a more reliable metric than exact match

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
| Study Defog AI's sql-eval repo (read README + key files) | 1h |
| Set up PostgreSQL in Colab (Option A or B) | 1h |
| Implement the evaluation harness | 2h |
| Run eval on 100 examples: base model, Week 33 model, Week 38 model | 1.5h |
| Write `week39_eval_report.md` with all results | 1h |
| Clean up and commit harness to GitHub | 30m |

## Why this week matters

Execute generated SQL against real Postgres; compare result sets, not token sequences.
