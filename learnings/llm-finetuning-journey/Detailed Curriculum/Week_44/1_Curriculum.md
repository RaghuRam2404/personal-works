# Week 44 — Building a Preference Dataset for SQL

## Learning Objectives

By the end of this week, you will be able to:

- Define RLAIF (AI feedback) and explain how it differs from RLHF (human feedback)
- Apply Constitutional AI principles to generate AI-labeled preference data for SQL
- Build an execution-based preference labeling pipeline: generate two SQL candidates, execute both on Postgres, label the one that executes correctly as "chosen"
- Produce and push a preference dataset of ≥ 2000 pairs to HuggingFace Hub
- Understand the quality tradeoffs between human-labeled, AI-labeled, and execution-labeled preference data

## The Key Idea: Execution as Ground Truth

Human preference labeling for SQL quality is expensive and subjective. "Is this SQL better?" depends on whether you know the expected output, the schema, and the business context. But execution against a real database is objective: the SQL either runs and returns the correct rows, or it does not.

This week you build preference data using **execution-based labeling** — no humans, no expensive API calls (beyond model inference), and no subjectivity. This is the cleanest possible preference signal for a code/SQL domain, and it directly informs what your GRPO reward function will do in Weeks 47–48.

## Concepts

### RLAIF: Reinforcement Learning from AI Feedback

RLAIF (Bai et al. 2022, Lee et al. 2023) replaces human labelers with a strong language model ("AI annotator"). The core pipeline:
1. For each prompt, generate multiple candidate responses
2. Ask a strong model (GPT-4o, Claude) to evaluate them against a rubric
3. Use the resulting preference labels to train your model

RLAIF scales far more easily than human feedback — you can generate millions of labeled pairs at API cost rather than human labor cost. The quality depends heavily on the quality of the AI annotator and the rubric.

For SQL, a pure RLAIF approach would ask GPT-4o "which of these two SQL queries is better?" This is reasonable but introduces:
- Cost: ~$0.002 per API call × 2000 pairs = ~$4
- Subjectivity: the AI's "better" may not match your specific Postgres schema
- Latency: API round-trips for every pair

Execution-based labeling bypasses all three issues.

### Constitutional AI Principles for SQL

Constitutional AI (CAI, Anthropic 2022) provides a framework for defining quality principles that can be applied consistently. For SQL generation, your "constitution" might be:

**Principles for "good SQL":**
1. The query executes without syntax errors on the target Postgres database
2. The query returns the correct rows for the given prompt (verified by comparing to expected output)
3. The query uses the schema correctly (correct table/column names, no hallucinations)
4. The query is efficient (avoids unnecessary subqueries, uses indexes where appropriate)
5. The query handles NULL values and edge cases correctly

You can use these principles both to label preferences (execution-based checking of principles 1-3) and to generate "critique and revision" pairs for additional training data.

### Execution-Based Preference Labeling Pipeline

The pipeline for this week:

```
For each prompt p in your prompt set:
  1. Generate SQL_A from your SFT model (postgres-sqlcoder-7b-v1)
  2. Generate SQL_B from the base model (Qwen2.5-Coder-7B)
  3. Execute SQL_A against test Postgres DB → result_A
  4. Execute SQL_B against test Postgres DB → result_B
  5. Compare to expected_output:
     - If result_A matches and result_B does not: (p, SQL_A, SQL_B) is a pair with SQL_A chosen
     - If result_B matches and result_A does not: (p, SQL_B, SQL_A) is a pair with SQL_B chosen
     - If both match: discard (no preference signal)
     - If neither matches: discard, OR use as a "both bad" example for future GRPO
```

The discard rate will be high initially (~60–70%). This is fine — quality over quantity.

### Prompt Set Design

Your prompts should cover the full distribution of queries you want your model to handle:
- Single-table SELECT with WHERE filters
- Multi-table JOINs (INNER, LEFT, RIGHT)
- Aggregations (GROUP BY, HAVING)
- Subqueries and CTEs
- TimescaleDB-specific: time_bucket, continuous aggregates, hyperfunctions
- Schema-specific: your actual PostgreSQL/TimescaleDB tables

For the prompt set, adapt from the Spider benchmark, WikiSQL, or generate synthetic prompts using your base model. You need at least 3000–5000 prompts to get 2000 clean preference pairs after filtering.

### Comparing Model vs. Model

Using one SFT model (v1) against the base Qwen model is a good starting strategy because:
- They have different failure modes (v1 knows your schema; base model does not)
- The winning pair will usually be v1 winning (this reinforces v1's strengths in DPO)
- When base model wins, it reveals cases where fine-tuning degraded capability

You can also generate two completions from the same model with different temperatures (T=0.3 and T=1.0) to get within-model preference pairs. This tends to give less signal but works well for edge cases.

### Data Quality Considerations

Not all preference pairs are equal:
- **High confidence pairs:** SQL_A executes correctly, SQL_B has a syntax error. Clear signal.
- **Medium confidence pairs:** Both execute but return different row counts. You need expected_output to label correctly.
- **Low confidence pairs:** Both execute and return the same rows. Discard.

For medium confidence pairs: you need a reference SQL (ground truth) that you know is correct. Run the reference SQL to get expected_output, then compare both candidates.

### Dataset Format

The HuggingFace preference dataset format for DPO training requires:

```python
{
    "prompt": "List all users who logged in after 2024-01-01",
    "chosen": "SELECT * FROM users WHERE last_login > '2024-01-01'",
    "rejected": "SELECT user_id, name FROM users INNER JOIN logins ON ..."  # wrong/broken
}
```

Push to HuggingFace Hub with `datasets.push_to_hub()`.

## Connections

Builds on: Week 43 (DPO dataset format, what a preference pair is).

Week 45 directly uses this dataset to train DPO. The quality of Week 45's model depends entirely on the quality of what you build this week.

Weeks 47–48 (GRPO) use a similar execution loop but inline during training — the preference labeling you are doing this week is a good preview of the reward function logic.

## Common Misconceptions

- "I need humans to label preferences for SQL." No — execution is ground truth. If the SQL runs and returns the right rows, it is preferred. This is one of the great advantages of code/SQL domains over open-ended generation.
- "I should include pairs where both are wrong." For DPO, both-wrong pairs are noise. DPO assumes y_w is actually good. For GRPO, you can use all completions because the reward distinguishes degrees of correctness.
- "More pairs is always better." 2000 clean pairs outperform 10,000 noisy pairs for DPO. Filter aggressively.
- "I should generate only from my SFT model." Using two different models (SFT vs. base) gives more diverse pairs and covers different failure modes.

## Time Allocation (6–8 hours)

- 1 hour: Read Constitutional AI paper abstract + Section 2. Read RLAIF paper abstract + Section 2.
- 30 min: Design your SQL constitution (5–7 principles). Write them down.
- 1 hour: Set up Postgres connection and execute function in Python.
- 3–4 hours: Build the pipeline end-to-end; generate and label 2000+ pairs.
- 30 min: Push dataset to HuggingFace Hub; validate the schema.
- 30 min: Analyze the dataset — what is the discard rate? Which query types are hardest?
