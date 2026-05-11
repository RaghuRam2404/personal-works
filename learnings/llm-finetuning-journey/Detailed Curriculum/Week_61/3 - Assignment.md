# Week 61 Assignment — Build the Evaluation Harness

## Setup Checklist

- [ ] BIRD-SQL dev set downloaded: [bird-bench.github.io](https://bird-bench.github.io/)
- [ ] Spider 1.0 downloaded: [yale-lily/spider on GitHub](https://github.com/taoyds/spider)
- [ ] PostgreSQL instance with sufficient connections for batch eval
- [ ] Defog sql-eval installed: `pip install git+https://github.com/defog-ai/sql-eval`
- [ ] Your final model accessible: `<your-handle>/postgres-sqlcoder-7b-final`
- [ ] Colab Pro or RunPod A100 for batch inference (local Mac too slow for 1,534 questions)

---

## Task 1 — Build the Generic Eval Harness

**Goal:** One script that can evaluate any model on any SQL benchmark.

**Requirements:**
Write `eval_harness.py` with CLI interface:
```
python eval_harness.py \
  --model <model-path-or-hf-repo> \
  --benchmark bird|spider|custom \
  --split dev \
  --max-examples 200 \
  --timeout 30 \
  --output eval_results.json
```

The harness must:
- Load examples from the benchmark's JSON format
- For each example: format the prompt using your standard system + schema + question template
- Run batch inference (batch size 8) using `model.generate()` with greedy decoding (temperature=0)
- Execute predicted SQL against the benchmark's database
- Compare result sets (sorted) against reference SQL result sets
- Compute: execution_accuracy, exact_match_accuracy, error_type_distribution, avg_latency_per_query
- Handle timeouts (wrap execution in a Thread with 30-second timeout)
- Save full results to `eval_results.json`: one record per question with predicted SQL, reference SQL, execution status, accuracy

**Deliverable:** `eval_harness.py` committed; runs successfully on 10 examples from BIRD-SQL dev.

---

## Task 2 — BIRD-SQL Dev Evaluation

**Goal:** Get honest execution accuracy on BIRD-SQL.

**Requirements:**
- Run eval harness on all 1,534 BIRD-SQL dev examples
- Log results per question to `bird_eval_results.json`
- Analyze error breakdown:
  - Type 1: SQL executes, correct result
  - Type 2: SQL executes, wrong result
  - Type 3: SQL syntax error
  - Type 4: SQL references wrong tables/columns
  - Type 5: Timeout
- Report in `bird_eval_summary.md`:
  - Overall execution accuracy
  - Execution accuracy by difficulty (BIRD has "Simple", "Moderate", "Challenging" labels)
  - Error type distribution
  - 5 example failures with predicted SQL and why it's wrong

**Expected range:** 50–70% execution accuracy depending on training data and prompt format.

**Deliverable:** `bird_eval_results.json` + `bird_eval_summary.md` committed.

---

## Task 3 — Spider 1.0 Dev Evaluation

**Requirements:**
Same as Task 2, run on Spider 1.0 dev set (1,034 examples). Convert Spider databases to PostgreSQL first:
- Load Spider `database/` folder
- Convert each SQLite schema to PostgreSQL DDL using sqlglot
- Load into a temporary PostgreSQL schema for each database

Report:
- Execution accuracy (all databases)
- Execution accuracy by SQL complexity (select-only, single join, multiple join, nested)
- Comparison vs. BIRD-SQL — your model should score higher on Spider (simpler queries)

**Deliverable:** `spider_eval_results.json` + `spider_eval_summary.md` committed.

---

## Task 4 — Defog sql-eval

**Requirements:**
- Run Defog's sql-eval framework on your final model
- Command: `python -m defog_eval --model <your-handle>/postgres-sqlcoder-7b-final --db postgres`
- Report execution accuracy from Defog's output
- Compare to Defog's published leaderboard numbers for SQLCoder and GPT-4

**Deliverable:** Defog eval output saved as `defog_eval_results.txt`.

---

## Task 5 — Custom Domain Evaluation

**Requirements:**
Run your 200-example PostgreSQL/TimescaleDB custom benchmark (built across phases 3–5) through the same harness. This is the most important evaluation — it directly measures performance on your target domain.

Break down by:
- TimescaleDB-specific queries (time_bucket, hyperfunctions, continuous aggregates)
- Standard PostgreSQL multi-table queries
- Multi-turn conversations
- Difficulty tier (Easy/Medium/Hard/Expert)

**Deliverable:** `custom_eval_results.json` + `custom_eval_summary.md` committed.

---

## Stretch Goals

- Implement "pass@K" evaluation: for each question, generate K=5 candidates with temperature=0.8; report what fraction of questions have at least 1 correct answer
- Compute "bootstrapped confidence intervals" on your eval scores (resample eval set 1,000 times; report 95% CI). This is essential for scientific credibility.
- Profile time-to-first-token and tokens-per-second for your model vs GPT-4o API (for the deployment cost argument in your technical report)
