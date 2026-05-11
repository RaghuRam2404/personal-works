# Week 55 Overview — Aggressive Filtering: LLM-as-Judge

This file is your map for Week 55. Read it first; everything else fits inside it.

## The story this week

You already have this from Week 54. Every example that fails SQL execution is removed. This is the cheapest filter and should always run first. Expected attrition: 30–45% of generated examples.

## What you need to do

- [ ] Raw dataset from Week 54 (`v3_raw.jsonl`) accessible locally
- [ ] PostgreSQL with all test schemas loaded
- [ ] `openai` client configured; API key set
- [ ] 50 hand-annotated examples ready (your gold calibration set)
- [ ] W&B project `week-55-filtering` created

Concretely, by the end of the week you should be able to:

- Design a multi-signal quality filter that combines execution validation, LLM-as-judge scoring, and semantic checks
- Implement a judge prompt that reliably scores SQL training examples on correctness, efficiency, and idiomaticity
- Calibrate an LLM judge's decisions against a human-annotated gold set
- Apply filtering at the right threshold to balance dataset size against quality
- Build a filtering pipeline that is reproducible, auditable, and fast enough to process 30K examples in a weekend

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

- 1h: Read Alpagasus paper (arXiv 2307.08701) — understand the judge prompt design
- 0.5h: Hand-annotate 50 examples from your raw data as calibration set
- 1.5h: Design and refine your judge prompt; achieve > 80% agreement with calibration set
- 2h: Build and run the full filtering pipeline on your 30K raw examples
- 1h: Analyze filter statistics, identify skill gaps after filtering
- 0.5h: Push filtered dataset to HuggingFace; commit code; log to W&B

## Why this week matters

**One-liner:** Filter at temperature=0.0, calibrate against human labels first, apply skill-adaptive thresholds.
