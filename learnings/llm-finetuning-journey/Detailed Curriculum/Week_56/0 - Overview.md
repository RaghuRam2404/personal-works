# Week 56 Overview — Conversational Multi-Turn SQL (CoSQL/SParC Style)

This file is your map for Week 56. Read it first; everything else fits inside it.

## The story this week

CoSQL (Yu et al., 2019) is a dataset of SQL queries grounded in multi-turn dialogues between a user and a system. Each dialogue has 3–7 turns. The user asks questions; the "system" (the SQL engine) generates SQL, runs it, shows results; the user responds to those results with a follow-up.

## What you need to do

- [ ] CoSQL downloaded: `git clone https://github.com/taoyds/cosql`
- [ ] SParC downloaded: `git clone https://github.com/taoyds/sparc`
- [ ] PostgreSQL with Spider databases loaded (convert SQLite to Postgres using pgloader or manual)
- [ ] Teacher API access for synthetic multi-turn generation
- [ ] `v3_filtered.jsonl` from Week 55 accessible
- [ ] W&B project `week-56-multiturn` created

Concretely, by the end of the week you should be able to:

- Explain the difference between single-turn and multi-turn text-to-SQL and why it matters for real applications
- Understand the CoSQL and SParC dataset formats and what makes them challenging
- Convert single-turn SQL examples into coherent multi-turn conversations
- Design a chat template that correctly represents multi-turn SQL history for training
- Add 5,000 multi-turn examples to your v3 dataset in the correct format

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

- 1h: Download CoSQL and SParC; run conversion scripts; count valid PostgreSQL examples
- 1h: Generate 2K synthetic multi-turn TimescaleDB examples using teacher model
- 1.5h: Build and run the multi-turn training format conversion pipeline
- 1h: Validate all multi-turn SQL executes correctly; apply quality filter
- 1h: Merge with single-turn filtered dataset; push final v3 to HuggingFace
- 0.5h: Commit, tag, and document

## Why this week matters

**One-liner:** Multi-turn training requires correct loss masking — loss only on assistant turns, all of them.
