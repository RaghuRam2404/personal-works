# Week 69 TakeAway — Technical Report: Evaluation and Ablations

The main results table is the most-read element of your paper. Every number must trace to a log or a cited source.

## Results Table Template

```
| Model              | Custom-200 (EM) | BIRD-SQL dev (EX) | Spider (EM) |
|--------------------|----------------|-------------------|-------------|
| Base model          | 61.0           | 54.2              | 71.3        |
| Best open-weight    | 67.4           | 60.1              | 80.2†       |
| GPT-4o              | 79.4           | 72.3†             | —           |
| postgres-sqlcoder-7b| **83.1**       | **68.4**          | **82.7**    |
```

## Ablation Table Template

```
| Configuration           | Custom-200 | BIRD-SQL |
|-------------------------|-----------|---------|
| Base model              | 61.0      | 54.2    |
| + CPT                   | 67.3*     | 58.6*   |
| + CPT + SFT             | 80.3      | 65.7    |
| + CPT + SFT + DPO       | 82.6      | 67.8    |
| + full pipeline         | 83.1      | 68.4    |
* = estimated; checkpoint not saved
```

## Decision Rules

- If you did not measure it: mark as — or †, never interpolate
- If improvement < CI bound: report the CI, do not remove the data point
- If failure mode is domain-specific (time_bucket): fix = add training data, not change training algorithm
- EM vs EX: report both if possible; EX is preferred for evaluating semantic correctness
- Baseline version strings are mandatory: "GPT-4o-2024-11-20, evaluated December 2024"

## Numbers to Remember

- 200-example test: 95% CI ≈ ±5.3 pp — improvements < 3 pp are not statistically conclusive
- Normalization: lowercase, strip whitespace, remove trailing semicolons (minimum)
- Failure mode analysis: inspect at least 30 examples; quantify each category as % of all failures
- Ablation principle: each row = base + exactly one additional component

## Red Flags

- Results table mixes EM and EX numbers in same column: apples vs oranges — use consistent metric
- Ablation jumps from base → full pipeline: no intermediate rows — readers cannot tell which stage helped
- Failure analysis describes errors qualitatively only: add counts and percentages
- Missing baseline version strings: reviewers will reject
- 0.5 pp improvement claimed as significant on N=200: must acknowledge CI
