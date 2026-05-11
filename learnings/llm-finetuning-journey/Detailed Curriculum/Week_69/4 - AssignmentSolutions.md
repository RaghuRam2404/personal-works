# Week 69 Assignment Solutions

## Task 1: Results Table Template

```markdown
## Table 1: Main Evaluation Results

| Model | Custom-200 (EM) | BIRD-SQL dev (EX) | Spider 1.0 dev (EM) |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (base) | 61.0 | 54.2 | 71.3 |
| SQLCoder-7B-v2 (Defog) | 63.5 | 57.8† | 80.2† |
| DeepSeek-Coder-V2-Lite-Instruct | 67.4 | 60.1 | 78.9 |
| GPT-4o (gpt-4o-2024-11-20) | 79.4 | 72.3† | — |
| Claude 3.5 Sonnet (20241022) | 81.2 | 73.1† | — |
| **postgres-sqlcoder-7b (ours)** | **83.1** | **68.4** | **82.7** |

† Result from published source; not measured by authors.
— Not evaluated.
Metric: EM = exact-match after normalization; EX = execution accuracy on test DB.
Custom-200: our TimescaleDB benchmark (Section 3 and Appendix A).
Evaluated December 2024.
```

## Task 2: Evaluation Section — Key Paragraphs

The analysis paragraphs should follow the table and directly address specific patterns:

```markdown
postgres-sqlcoder-7b achieves 83.1% exact-match on our TimescaleDB benchmark,
surpassing GPT-4o (79.4%) by 3.7 pp and Claude 3.5 Sonnet (81.2%) by 1.9 pp.
This is the primary result: a 7B open-weight model, running locally at 83 tok/s,
outperforms frontier closed-source APIs on domain-specific PostgreSQL tasks.

The gain over the base Qwen2.5-Coder-7B model (61.0%) is 22.1 pp, indicating
that continued pretraining and fine-tuning provide substantial domain adaptation
beyond what the base model's code training captured.

On BIRD-SQL dev (general SQL), our model achieves 68.4% EX, outperforming all
open-weight baselines but trailing GPT-4o (72.3†) and Claude 3.5 (73.1†).
This gap on general SQL reflects the domain-specialization tradeoff: our training
data is heavily weighted toward PostgreSQL/TimescaleDB, which helps on Custom-200
but provides less improvement on BIRD's diverse schema domains.
```

## Task 3: Ablation Table Template

```markdown
## Table 2: Ablation Study

| Configuration | Custom-200 (EM) | BIRD-SQL dev (EX) |
|---|---|---|
| Base model (Qwen2.5-Coder-7B) | 61.0 | 54.2 |
| + CPT (102M tokens PostgreSQL) | 67.3* | 58.6* |
| + CPT + SFT (25.5K examples) | 80.3 | 65.7 |
| + CPT + SFT + DPO (5K pairs) | 82.6 | 67.8 |
| + CPT + SFT + DPO + GRPO | **83.1** | **68.4** |

* CPT-only checkpoint not saved; result estimated by evaluating on a preserved
  CPT checkpoint from an earlier run at 80% of total CPT steps.
```

Key analysis point: SFT provides the largest single-stage gain (+13 pp on Custom-200). DPO adds 2.3 pp. GRPO adds 0.5 pp. CPT adds 6.3 pp. Each stage contributes — none is zero.

## Task 4: Failure Mode Analysis Template

```markdown
## Failure Mode Categories (n=34 failed examples)

| Error Type | Count | % of Failures | Example |
|---|---|---|---|
| Missing time_bucket | 12 | 35% | Q: "hourly revenue last week" → model uses DATE_TRUNC instead |
| Wrong JOIN type | 9 | 26% | INNER JOIN where LEFT JOIN required (null customers) |
| Hallucinated column | 7 | 21% | References "order_status" column that does not exist in schema |
| Wrong aggregation | 6 | 18% | COUNT(*) where SUM(amount) required |
```

## Common Gotchas

- When reporting "† from published source," be specific: "from Defog's evaluation blog post, October 2023." Generic "published" is unverifiable.
- The ablation table requires actual measured numbers. If you estimate, say so explicitly and explain your estimation method in a footnote.
- Exact-match after normalization: your normalization steps must be documented (e.g., lowercasing, stripping whitespace, removing aliases). Without this, your EM numbers are not reproducible.
- Failure mode analysis: 30 examples is the minimum meaningful sample. Inspect randomly selected failures, not the worst failures, to avoid selection bias.

## How to Verify You Did It Right

The evaluation section passes review if: (1) every number in Table 1 can be traced to either a log file or a cited source, (2) every baseline model has a version string, (3) the ablation table has the same metric and benchmark columns as Table 1 for comparability, and (4) the failure analysis quantifies (not just describes) each error category.
