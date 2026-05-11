# Week 47 Overview — GRPO with Executable Rewards on SQL (Week 1 of 2)

This file is your map for Week 47. Read it first; everything else fits inside it.

## The story this week

A binary {0, 1} reward for SQL execution is too sparse — if only 10% of your model's completions execute correctly, 90% of steps get zero gradient. Design a hierarchy:

## What you need to do

- [ ] Postgres DB accessible (local or cloud)
- [ ] `execute_sql()` from Week 44 available
- [ ] At least 200 prompts with reference SQL for diagnostic testing
- [ ] Python environment with `psycopg2-binary`, `re`, `torch`
- [ ] Your v2-dpo model loaded (this is the GRPO starting checkpoint)

Concretely, by the end of the week you should be able to:

- Design a multi-level SQL reward function that resists reward hacking
- Implement and test the reward function against a real Postgres instance
- Set up the GRPO training pipeline with Unsloth and TRL
- Validate that the reward function produces correct reward distributions on a diagnostic sample
- Identify reward hacking patterns specific to SQL and build defenses against them

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

- 1 hour: Design your reward hierarchy on paper. List the levels, thresholds, and anti-hack rules.
- 1 hour: Implement `extract_sql()` and the execution harness.
- 2 hours: Implement the full `sql_reward_fn()` with all 4 levels.
- 1 hour: Diagnostic testing on 50 prompts. Debug any extraction or execution issues.
- 1–2 hours: Set up the GRPO training script (Unsloth + TRL). Do NOT start training yet — that is Week 48.
- 30 min: Write your reward hacking audit in `week-47-grpo/reward_hacking_audit.md`.

## Why this week matters

**One-liner:** Design rewards with multiple levels and explicit anti-hack guards; test on 100 completions before connecting to GRPO.
