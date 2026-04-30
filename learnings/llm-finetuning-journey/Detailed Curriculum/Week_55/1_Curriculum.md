# Week 55 — Aggressive Filtering: LLM-as-Judge

## Learning Objectives

By the end of this week, you will be able to:

- Design a multi-signal quality filter that combines execution validation, LLM-as-judge scoring, and semantic checks
- Implement a judge prompt that reliably scores SQL training examples on correctness, efficiency, and idiomaticity
- Calibrate an LLM judge's decisions against a human-annotated gold set
- Apply filtering at the right threshold to balance dataset size against quality
- Build a filtering pipeline that is reproducible, auditable, and fast enough to process 30K examples in a weekend

## Why Filtering Matters More Than Generation

The key insight from the alignment literature — from Alpagasus to Tulu 3 — is that filtering is more impactful than generation. Generating more examples is linear: 2× cost → 2× data. Filtering smarter is superlinear: removing the bottom 30% of your data can yield a model that is substantially better, not just 30% different. This week is arguably more important than Week 54.

The intuition: a model trained on 70% good + 30% bad data does not achieve 70% of peak performance. The bad examples actively degrade performance by teaching the model wrong patterns, contradicting good examples, or reinforcing biases in phrasing. Quality filtering creates a cleaner gradient signal during training.

## Concepts

### Signal 1 — Execution Correctness

You already have this from Week 54. Every example that fails SQL execution is removed. This is the cheapest filter and should always run first. Expected attrition: 30–45% of generated examples.

However, execution correctness alone passes examples that are:
- Syntactically valid but semantically wrong
- Correct for the test schema but wrong for the intended question
- Unnecessarily complex (full table scans when indexes exist)

### Signal 2 — LLM-as-Judge

The Alpagasus paper (Chen et al., 2023) showed that GPT-4 can reliably score instruction-following data on a 1–5 scale, and that filtering to only score-4+ examples dramatically improves fine-tuned model quality. For SQL, the judge needs to evaluate:

1. **Question clarity:** Is the natural language question unambiguous and realistic?
2. **SQL correctness:** Does the SQL correctly answer the question against the given schema?
3. **SQL efficiency:** Does the SQL use reasonable indexing and avoid obvious inefficiencies?
4. **SQL idiomaticity:** Does the SQL use the most appropriate construct (CTE vs. subquery, `time_bucket` vs. manual date arithmetic)?

**Designing the judge prompt.** The judge prompt must include the schema DDL, the question, the SQL, and a rubric. Without the rubric, the judge produces inconsistent scores. The rubric must define what each score level means precisely:

```
Score 5: SQL exactly answers the question, uses the schema correctly, is production-appropriate in efficiency and style.
Score 4: SQL correctly answers the question with minor style issues or mild inefficiency.
Score 3: SQL answers the question but has notable problems (missing edge case, suboptimal join order, wrong aggregate window).
Score 2: SQL is syntactically valid but semantically wrong for the question, or uses non-existent schema elements.
Score 1: SQL has syntax errors, references non-existent tables/columns, or completely fails to address the question.
```

Only keep score ≥ 4 examples. This typically retains 55–65% of execution-passing examples.

### Signal 3 — Semantic Consistency Check

For examples where you have a reference answer (Spider, BIRD, your hand-curated set), compare the generated answer's output rows against the reference answer's output rows. An example passes only if the result sets are identical (or equivalent under column reordering).

For synthetic examples without a reference, you can apply a weaker check: run the query and verify the result set is non-empty and of plausible size (not 0 rows, not 1M rows for a query about recent data).

### Signal 4 — Complexity vs. Question Alignment

A question labeled "Easy" should have a query of AST depth < 20. A question labeled "Expert" should use at least 2 advanced constructs. Mismatches indicate the teacher misunderstood the difficulty level. Filter out examples where `difficulty_label` and `sql_complexity` disagree by more than one tier.

### Building the Filtering Pipeline

The filter pipeline should run in this order (cheapest to most expensive):

1. Parse validation (JSON parseable) — run during generation (Week 54)
2. Execution correctness — run during generation (Week 54)
3. Deduplication — run during generation (Week 54)
4. Semantic size check (non-empty, non-exploding result set) — fast, no API call
5. Complexity-label alignment check — fast, uses sqlglot AST
6. LLM-as-judge scoring — expensive, batch and parallelize

Total cost for judging 20K execution-passing examples (after steps 1–5 reduce to ~15K):
- At ~100 tokens input + 50 tokens output per example
- 15K × 150 tokens = 2.25M tokens ≈ $11 at GPT-4o pricing
- Use GPT-4o-mini for initial scoring (~$0.30), escalate borderline (score = 3) to GPT-4o (~$3)

### Calibration: Validate Your Judge

Before running the judge on 15K examples, calibrate it on 50–100 hand-annotated examples from your gold set. Compute:
- Agreement rate (judge score ≥ 4 ↔ human label "keep"): target > 80%
- Cohen's kappa: target > 0.6 (substantial agreement)

If agreement is below 80%, your rubric is ambiguous. Refine the rubric with concrete examples at each score level (few-shot calibration examples in the judge prompt).

### Audit Trail

Every filtered-out example should be saved with its filter reason. This enables:
- Diagnosing which signals are most aggressive
- Reviewing edge cases where the judge and execution disagree
- Reporting in your technical report: "We removed X% of examples at each stage"

### Common Misconceptions and Pitfalls

**"LLM judges are biased toward verbose answers."** Yes — a known finding. Longer SQL is rated higher by judges even when a shorter query is more appropriate. Counter this by explicitly including in your rubric: "Prefer concise, idiomatic SQL. Penalize unnecessary CTEs, redundant subqueries, or overly verbose aliases."

**"Filtering is all-or-nothing."** You can keep borderline examples (score = 3) if they are your only examples for a rare skill. Better one imperfect TimescaleDB hyperfunction example than no example at all. Apply softer filtering (score ≥ 3) for rare skills, strict filtering (score ≥ 4) for common skills.

**"More filtering = better model."** Past a certain threshold, aggressive filtering shrinks the dataset so much that the model has insufficient examples for rare skills. Monitor your skill distribution throughout filtering — don't let any skill drop below 200 examples.

## Connections

This week consumes Week 54's raw data. The filtered dataset (`postgres-sql-v3-filtered`) is the direct input to Week 58's SFT run. The filtering methodology and statistics (how many examples removed at each stage) become a key subsection of your technical report (Week 67).

## Time Allocation (6–8 hrs)

- 1h: Read Alpagasus paper (arXiv 2307.08701) — understand the judge prompt design
- 0.5h: Hand-annotate 50 examples from your raw data as calibration set
- 1.5h: Design and refine your judge prompt; achieve > 80% agreement with calibration set
- 2h: Build and run the full filtering pipeline on your 30K raw examples
- 1h: Analyze filter statistics, identify skill gaps after filtering
- 0.5h: Push filtered dataset to HuggingFace; commit code; log to W&B
