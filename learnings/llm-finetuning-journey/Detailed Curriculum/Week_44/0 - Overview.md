# Week 44 Overview — Building a Preference Dataset for SQL

This file is your map for Week 44. Read it first; everything else fits inside it.

## The story this week

RLAIF (Bai et al. 2022, Lee et al. 2023) replaces human labelers with a strong language model ("AI annotator"). The core pipeline:
1. For each prompt, generate multiple candidate responses
2. Ask a strong model (GPT-4o, Claude) to evaluate them against a rubric
3. Use the resulting preference labels to train your model

## What you need to do

- [ ] PostgreSQL running locally (or a cloud Postgres instance). Your TimescaleDB setup from earlier phases works.
- [ ] Python packages: `psycopg2-binary`, `transformers`, `datasets`, `huggingface_hub`, `torch`
- [ ] HuggingFace account and `huggingface-cli login` completed
- [ ] Your SFT model checkpoint from Phase 4 (`postgres-sqlcoder-7b-v1`) accessible
- [ ] W&B project for logging stats (optional but recommended)
- [ ] At least 3000 SQL prompts (adapted from Spider/WikiSQL or synthetically generated)

Concretely, by the end of the week you should be able to:

- Define RLAIF (AI feedback) and explain how it differs from RLHF (human feedback)
- Apply Constitutional AI principles to generate AI-labeled preference data for SQL
- Build an execution-based preference labeling pipeline: generate two SQL candidates, execute both on Postgres, label the one that executes correctly as "chosen"
- Produce and push a preference dataset of ≥ 2000 pairs to HuggingFace Hub
- Understand the quality tradeoffs between human-labeled, AI-labeled, and execution-labeled preference data

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

- 1 hour: Read Constitutional AI paper abstract + Section 2. Read RLAIF paper abstract + Section 2.
- 30 min: Design your SQL constitution (5–7 principles). Write them down.
- 1 hour: Set up Postgres connection and execute function in Python.
- 3–4 hours: Build the pipeline end-to-end; generate and label 2000+ pairs.
- 30 min: Push dataset to HuggingFace Hub; validate the schema.
- 30 min: Analyze the dataset — what is the discard rate? Which query types are hardest?

## Why this week matters

**One-liner:** Execution-based labeling uses Postgres as the judge — no humans, no subjectivity, ground truth for free.
