# Week 76 Overview — Curriculum — Multi-Turn Agentic SQL with Tool Use

This file is your map for Week 76. Read it first; everything else fits inside it.

## The story this week

Single-shot NL→SQL is the task you have been training for: given a schema and a question, produce SQL in one pass. Agentic SQL extends this into a loop. The model generates a candidate query, that query is executed against a real or simulated database, and the result — whether a table of rows, an error message, or empty results — is fed back to the model. The model then decides: accept the result, ask a clarifying question, or generate a revised query.

## What you need to do

- [ ] PostgreSQL instance running locally (or on Colab via `apt-get install postgresql` + `service postgresql start`)
- [ ] Tables loaded: at minimum `orders`, `customers`, `products`, `timeseries_metrics` (use your existing schema from Week 53 onward)
- [ ] `psycopg2` installed: `pip install psycopg2-binary`
- [ ] Your fine-tuned model (`postgres-sqlcoder-7b-final`) loaded locally or accessible via the HuggingFace Hub
- [ ] W&B project `week-76-agentic-sql` created
- [ ] Python environment with `transformers>=4.40`, `trl>=0.8`, `peft>=0.10`

Concretely, by the end of the week you should be able to:

- Construct a multi-turn training dataset where the model generates partial SQL, receives an execution error, and refines its output.
- Implement a tool-calling loop in Python that connects your fine-tuned model to a live PostgreSQL database for agentic self-correction.
- Format tool-call messages using a structured schema (JSON function signatures) that the model is trained to emit and parse.
- Fine-tune your model on multi-turn agentic trajectories using SFT with correct loss masking across all assistant turns.
- Evaluate agentic SQL performance with metrics that capture correction ability, not just single-shot accuracy.

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

- 1.5 hours: Read tool-calling format docs for your base model; implement `apply_chat_template` with tools and inspect output.
- 2 hours: Build trajectory generation pipeline; create 500 agentic training examples.
- 2 hours: Implement multi-turn SFT with correct loss masking; run 500 steps; verify loss curve.
- 1 hour: Implement the agentic inference loop; test against 20 Custom-200 examples; compare first-attempt vs final EM.
- 0.5 hours: Log all metrics to W&B project `week-76-agentic-sql`; write decision memo.

## Why this week matters

**This week in 15 words:** Train on error-correction trajectories; run a database-connected loop that self-corrects SQL at inference time.
