# Week 61 Overview — Comprehensive Eval Harness Part 1: BIRD-SQL, Spider 2.0, Defog

This file is your map for Week 61. Read it first; everything else fits inside it.

## The story this week

Spider (Yu et al., 2018) is the foundational text-to-SQL benchmark. It has:
- 10,206 question/SQL pairs across 206 databases
- Cross-database setting (training and test use different databases — tests generalization)
- Standard split: train (7,000), dev (1,034), test (2,147 — labels not public)
- Evaluation metric: execution accuracy (does the predicted SQL produce the same rows as the gold SQL?) and exact-match accuracy

## What you need to do

- [ ] BIRD-SQL dev set downloaded: [bird-bench.github.io](https://bird-bench.github.io/)
- [ ] Spider 1.0 downloaded: [yale-lily/spider on GitHub](https://github.com/taoyds/spider)
- [ ] PostgreSQL instance with sufficient connections for batch eval
- [ ] Defog sql-eval installed: `pip install git+https://github.com/defog-ai/sql-eval`
- [ ] Your final model accessible: `<your-handle>/postgres-sqlcoder-7b-final`
- [ ] Colab Pro or RunPod A100 for batch inference (local Mac too slow for 1,534 questions)

Concretely, by the end of the week you should be able to:

- Run execution-based evaluation on BIRD-SQL and Spider 2.0 benchmarks with your model
- Integrate the Defog sql-eval framework as a standardized evaluation harness
- Understand the methodological differences between Spider, BIRD, and Spider 2.0
- Build a reproducible eval script that can be re-run on any model checkpoint
- Interpret evaluation results correctly — what each benchmark measures and what it doesn't

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

- 1h: Download BIRD-SQL dev set, Spider 1.0, and Spider 2.0 (PostgreSQL subset)
- 0.5h: Set up databases for each benchmark in PostgreSQL
- 2h: Build the generic eval harness script with batch inference
- 1.5h: Run evaluation on BIRD-SQL dev (1,534 questions)
- 1h: Run evaluation on Spider 1.0 dev (1,034 questions)
- 0.5h: Run Defog sql-eval on your model
- 0.5h: Compile results into `eval_results_part1.md`; commit

## Why this week matters

**One-liner:** Always use dev set for selection, execution accuracy over result comparison, and report confidence intervals.
