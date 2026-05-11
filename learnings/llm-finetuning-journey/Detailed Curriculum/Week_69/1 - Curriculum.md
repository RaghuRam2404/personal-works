# Week 69 — Technical Report Week 3: Evaluation and Ablations

## Learning Objectives

By the end of this week, you will be able to:

- Write an evaluation section that clearly distinguishes your benchmarks, your baselines, and your metrics
- Present results tables in the format expected by ML publications
- Design and document ablation studies that isolate the contribution of each training stage
- Identify and communicate failure modes as a separate analysis subsection
- Integrate evaluation and ablations into the report draft

## Concepts

### Structure of the Evaluation Section

The evaluation section answers three questions: what did you measure, how did you measure it, and what did you find? It is the most-read section of an LLM paper after the abstract. Structure:

```
5. Evaluation

5.1 Evaluation Setup
  5.1.1 Benchmarks
  5.1.2 Metrics
  5.1.3 Baselines
  5.1.4 Inference configuration

5.2 Main Results

5.3 Analysis
  5.3.1 Performance by SQL complexity
  5.3.2 Performance by schema type
  5.3.3 Failure mode analysis
```

### Writing the Evaluation Setup

The setup subsection must answer: exactly what test sets did you use? How big are they? What metric? What inference settings (temperature, max tokens, prompt format)?

Benchmarks to document:
- Your custom 200-example TimescaleDB benchmark (release it)
- BIRD-SQL development set (1,534 examples, publicly available)
- Spider 1.0 development set (1,034 examples, publicly available)
- Defog sql-eval (subset you ran)

For each: name, size, domain/schema type, what accuracy metric you used. Do not assume readers know BIRD or Spider.

Metrics to define precisely:
- Exact-match SQL (EM): the generated SQL matches the reference SQL string exactly (after normalization). Easy to compute, but too strict — semantically equivalent SQL may differ in whitespace or alias names.
- Execution accuracy (EX): the generated SQL produces the same result set as the reference SQL when executed against the database. More meaningful but requires a live database. Document which metric you used for each benchmark.

Baselines: list every model you compare against, with its checkpoint or API version:
- Qwen2.5-Coder-7B-Instruct (base model, before any fine-tuning)
- SQLCoder-7B (Defog, v2)
- DeepSeek-Coder-V2-Lite-Instruct
- GPT-4o (gpt-4o-2024-11-20, evaluated December 2024)
- Claude 3.5 Sonnet (claude-3-5-sonnet-20241022, evaluated December 2024)

### Writing the Main Results Table

This is the centerpiece of your paper. Every row is a model, every column is a benchmark:

```markdown
| Model | Custom-200 | BIRD-SQL dev | Spider 1.0 dev | Defog sql-eval |
|-------|-----------|--------------|----------------|----------------|
| Qwen2.5-Coder-7B (base) | 61.0 | 54.2 | 71.3 | 68.4 |
| SQLCoder-7B | 63.5 | 57.8 | 80.2 | 74.1 |
| DeepSeek-Coder-V2-Lite | 67.4 | 60.1 | 78.9 | 72.8 |
| GPT-4o (gpt-4o-2024-11-20) | 79.4 | 72.3† | — | — |
| Claude 3.5 Sonnet | 81.2 | 73.1† | — | — |
| postgres-sqlcoder-7b (ours) | **83.1** | 68.4 | 82.7 | 79.3 |
```

Notes on presentation:
- Bold the best result in each column
- Use † for results taken from published sources rather than measured by you
- Use — for benchmarks you did not run
- Put the most important column first (your domain benchmark)

### Writing the Ablation Study

The ablation study is Section 6, but it references the main results table as the full-pipeline baseline. Each ablation row removes one training stage:

```markdown
| Configuration | Custom-200 | BIRD-SQL dev |
|---|---|---|
| Base model only | 61.0 | 54.2 |
| + CPT | 67.3 | 58.6 |
| + CPT + SFT | 80.3 | 65.7 |
| + CPT + SFT + DPO | 83.1 | 68.4 |
| + CPT + SFT + DPO + GRPO | **83.1** | **68.4** |
```

Wait — if DPO and GRPO give the same Custom-200 result, something is off. Use your actual numbers. The ablation table must be constructed from actual checkpoints you evaluated — not estimated. If you did not save intermediate checkpoints, acknowledge this: "We did not save the CPT-only and SFT-only checkpoints; ablations for stages 1 and 2 are reported from the nearest available checkpoint."

Additional ablation dimensions:
- SFT dataset ablation: curated-only vs synthetic-only vs combined
- DPO β ablation: β ∈ {0.05, 0.1, 0.5}
- GRPO K ablation: K ∈ {4, 8, 16}

### Writing the Failure Mode Analysis

The most honest part of the paper. Analyze your model's errors on the 200-example benchmark:

Categorize failures:
- Correct SQL structure, wrong aggregation function (e.g., SUM vs COUNT)
- Incorrect JOIN type (LEFT vs INNER)
- Missing TimescaleDB-specific syntax (`time_bucket`, `first()`, `last()`)
- Schema hallucination (model invents a column that does not exist)
- Multi-table queries with wrong alias references

Quantify each category. Example: "Of the 34 failed queries, 12 (35%) involved missing `time_bucket` usage, 9 (26%) had incorrect JOIN types, 7 (21%) hallucinated column names, and 6 (18%) used wrong aggregations."

This analysis directly motivates your Limitations section in Week 70.

## Connections

This week's evaluation section assembles results from Weeks 61–62 (eval harness). The ablation table references checkpoints from Weeks 57–60. The failure mode analysis feeds directly into the Limitations section (Week 70). The main results table is what gets cited in future work that builds on yours.

## Common Misconceptions / Pitfalls

Do not conflate exact-match and execution accuracy — they are different metrics and produce different numbers on the same model and benchmark. If your evaluation used exact-match (simpler to compute), be honest about this and note that execution accuracy would be more meaningful.

The ablation table should show additive contributions — each row adds exactly one component. Jumping from base → CPT+SFT (skipping CPT-only) obscures whether CPT contributed independently of SFT.

Do not cherry-pick example outputs in the paper. If you show qualitative examples, randomly sample them or show the best, worst, and median performance examples.

## Time Allocation (6–8 hours)

- 1.0h: Review and clean up results from Weeks 61–62 into a single consolidated table
- 1.0h: Write evaluation setup subsection (benchmarks, metrics, baselines, inference config)
- 1.0h: Write main results table + 2 paragraphs of analysis
- 1.5h: Write ablation study table + analysis (may require re-running some evaluations)
- 1.0h: Write failure mode analysis (manual inspection of 34 failed examples)
- 0.5h: Integrate into `report_draft_v2.md` through Section 6
