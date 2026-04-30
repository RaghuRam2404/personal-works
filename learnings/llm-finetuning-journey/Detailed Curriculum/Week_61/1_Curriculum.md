# Week 61 — Comprehensive Eval Harness Part 1: BIRD-SQL, Spider 2.0, Defog

## Learning Objectives

By the end of this week, you will be able to:

- Run execution-based evaluation on BIRD-SQL and Spider 2.0 benchmarks with your model
- Integrate the Defog sql-eval framework as a standardized evaluation harness
- Understand the methodological differences between Spider, BIRD, and Spider 2.0
- Build a reproducible eval script that can be re-run on any model checkpoint
- Interpret evaluation results correctly — what each benchmark measures and what it doesn't

## Why Rigorous Evaluation Matters Now

Your model is trained. Now you must measure it honestly. This is where many projects go wrong: they report cherry-picked numbers on easy benchmarks, or they design their eval set to match their training distribution, or they compare against weaker baselines. Your technical report will be credible only if your evaluation is rigorous, reproducible, and honest.

This week builds the infrastructure. Week 62 runs the head-to-head comparisons. Together they produce the evaluation section of your technical report.

## Concepts

### Spider: The Baseline Benchmark

Spider (Yu et al., 2018) is the foundational text-to-SQL benchmark. It has:
- 10,206 question/SQL pairs across 206 databases
- Cross-database setting (training and test use different databases — tests generalization)
- Standard split: train (7,000), dev (1,034), test (2,147 — labels not public)
- Evaluation metric: execution accuracy (does the predicted SQL produce the same rows as the gold SQL?) and exact-match accuracy

**Key limitation:** Spider uses SQLite and relatively simple schemas (mostly 2–5 table schemas). It does not test PostgreSQL-specific features, time-series SQL, or complex CTEs. Your model may score lower on Spider than on your domain benchmark — this is expected and should be explained in your report.

**Converting Spider to PostgreSQL for your eval:**
Spider was designed for SQLite. To evaluate your PostgreSQL-trained model on Spider, you have two options:
1. Use the SQLite databases directly with a SQLite connection for eval
2. Convert Spider schemas to PostgreSQL using your sqlglot conversion pipeline

Option 2 is more rigorous for your claim of "PostgreSQL specialist."

### BIRD-SQL: The Harder Benchmark

BIRD-SQL (Li et al., 2023, arXiv 2305.03111) is the current state-of-the-art benchmark for text-to-SQL. Key differences from Spider:
- 12,751 question/SQL pairs across 95 databases
- Much more complex queries (nested CTEs, window functions, multi-step reasoning)
- Includes database content (actual data values, not just schema) in the evaluation context
- Evaluation: execution accuracy with a "soft match" for equivalent queries

BIRD is widely used for leaderboard comparisons and is the benchmark you will use to compare against GPT-4o, SQLCoder, and DeepSeek-Coder in Week 62.

**The BIRD "external knowledge" challenge:** Many BIRD questions require external knowledge to answer — e.g., "Which employees are at risk of burnout?" requires knowing what "burnout risk" means. Your model receives this knowledge in the question context, but handling it correctly is non-trivial.

**Subset selection:** For your Week 61 eval, use the BIRD dev set (1,534 questions) — the test set labels are not public. Run all 1,534 dev questions through your model and compare execution accuracy against the provided reference SQL.

### Spider 2.0: The Enterprise SQL Benchmark

Spider 2.0 (spider2-sql.github.io) is a 2024 evolution addressing key limitations of Spider:
- Enterprise-scale databases (BigQuery, Snowflake, DuckDB, PostgreSQL)
- Multi-step reasoning with agent loops
- Complex SQL: window functions, CTEs, JSON operations, semi-structured data
- Much harder than Spider 1.0 — current SOTA models score < 10% on its hardest examples

For your eval, use Spider 2.0's PostgreSQL subset. This is the most relevant subset for your domain and tests exactly the skills you trained on.

### Defog SQL-Eval Framework

The Defog sql-eval framework (github.com/defog-ai/sql-eval) provides:
- A standardized runner that manages database connections, prompt formatting, and result comparison
- Support for multiple model backends: OpenAI API, local models via HuggingFace, vLLM
- A dataset of 200 "real-world" SQL questions from Defog's customer deployments
- Execution-based scoring with tolerance for equivalent SQL expressions

The Defog dataset is particularly valuable because it tests SQL that real users ask in real applications — not academic benchmark questions. Running your model on Defog gives you a market-relevant score.

**Integration:** Defog sql-eval accepts a HuggingFace model path and runs evaluation automatically. You pass `--model <your-handle>/postgres-sqlcoder-7b-final` and it handles the rest.

### Building Your Eval Harness

Your eval harness should be a single script `eval_harness.py` that:
1. Takes a model path and benchmark name as arguments
2. Loads the model with efficient inference configuration (4-bit or fp16)
3. Runs the benchmark's test cases through the model
4. Computes and reports: execution accuracy, exact match, latency per query

Key implementation decisions:
- **Batch inference:** For large benchmarks (BIRD: 1,534 questions), use batch generation with `model.generate(input_ids_batch)`. This is 5–10× faster than sequential.
- **Result comparison:** Two SQL queries are equivalent if they produce identical sorted result sets (not identical queries — there are many ways to write the same SQL). Always sort result sets before comparison.
- **Timeout:** Set a 30-second timeout per query execution. Runaway queries (missing indexes, cross joins) should be counted as fails, not allowed to hang.
- **Schema context:** Provide the same schema DDL format in your eval prompts as you used during training. Mismatched formats cause artificial performance drops.

### Common Misconceptions and Pitfalls

**"My eval score on BIRD represents my model's general SQL ability."** BIRD measures a specific type of complex SQL reasoning on specific domains (mostly business databases). Your model's true domain advantage (TimescaleDB) is not measured by BIRD at all.

**"Exact match accuracy is the right metric."** Exact match is brittle — there are many equivalent SQL queries. Always use execution accuracy as your primary metric. Exact match is a secondary, stricter metric.

**"I should evaluate on the test set."** Never evaluate on the test set while still developing your model. The test set labels are often not public (BIRD, Spider) precisely to prevent test set overfitting. Use dev set for development; reserve test set (if labels are available) for final one-shot reporting.

## Connections

This week's evaluation infrastructure builds directly on Week 39 (execution-based eval), where you first built a PostgreSQL runner that compares predicted SQL against reference SQL by executing both. That runner is the prototype; this week you generalize it into a benchmark-agnostic harness.

Week 60 produced `postgres-sqlcoder-7b-final`, the model under evaluation this week. Run evaluation only on the final merged checkpoint, not on intermediate GRPO checkpoints — checkpoint selection bias inflates reported numbers.

Week 62 uses the BIRD-SQL and Spider 2.0 scores you establish this week as the baseline for the head-to-head comparison against GPT-4o, Claude 3.5 Sonnet, and SQLCoder-7B. The eval harness you build this week must be identical in Week 62 — do not change prompt format, schema context, or comparison logic between weeks, or the numbers will not be comparable.

The `eval_results_part1.md` you commit at the end of this week feeds directly into the evaluation section of your technical report (Weeks 68–70).

## Time Allocation (6–8 hrs)

- 1h: Download BIRD-SQL dev set, Spider 1.0, and Spider 2.0 (PostgreSQL subset)
- 0.5h: Set up databases for each benchmark in PostgreSQL
- 2h: Build the generic eval harness script with batch inference
- 1.5h: Run evaluation on BIRD-SQL dev (1,534 questions)
- 1h: Run evaluation on Spider 1.0 dev (1,034 questions)
- 0.5h: Run Defog sql-eval on your model
- 0.5h: Compile results into `eval_results_part1.md`; commit
