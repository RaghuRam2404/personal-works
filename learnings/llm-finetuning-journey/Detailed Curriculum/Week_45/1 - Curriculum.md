# Week 45 — DPO on Your Domain Model

## Learning Objectives

By the end of this week, you will be able to:

- Apply DPO using Unsloth's DPO trainer to your SFT model with your own preference dataset
- Diagnose and fix common DPO training issues specific to the SQL domain
- Produce a quantitative eval report comparing v1 (SFT only) vs. v2 (SFT + DPO)
- Interpret the reward_margin metric and relate it to downstream SQL execution correctness
- Understand why DPO might not improve on hard queries and what that implies for GRPO

## What This Week Is

This is an applied training week. The conceptual groundwork is done. Your job is to take the preference dataset you built in Week 44 (`<your-handle>/postgres-sql-preferences-v1`) and fine-tune `postgres-sqlcoder-7b-v1` using DPO to produce `postgres-sqlcoder-7b-v2-dpo`.

The week ends with a real evaluation: does v2 outperform v1 on execution correctness on a held-out test set? If not, you need to diagnose why before moving to GRPO.

## Concepts

### DPO Training on SQL Preferences — What to Expect

DPO trains the model to increase the probability of chosen SQL completions relative to the reference model (v1) and decrease the probability of rejected ones. In the SQL domain, the key effects you should observe:

- Reduction in syntax errors: rejected examples are often syntactically broken
- Better schema adherence: the model reduces the probability of hallucinated table/column names seen in rejected examples
- Marginal improvement on complex queries: DPO's offline nature limits it here — the model cannot explore new SQL patterns it did not see in the dataset

If v2-dpo gets worse on complex queries, this is expected and is the motivation for GRPO.

### Unsloth DPO

Unsloth's DPO trainer wraps TRL's `DPOTrainer` with memory optimizations for the 7B model class. On Colab Pro (A100 40GB), you can run DPO without quantization. With 4-bit quantization (bitsandbytes), you can run it on a 16GB GPU.

Key differences from the generic TRL DPO you ran in Week 43:
- Unsloth uses a patched `DPOTrainer` that avoids the memory spike from dual-model forward passes
- You apply LoRA to the training model; the reference model stays as the base 7B in 4-bit
- Training speed is 2–3× faster than vanilla TRL for equivalent output quality

### β Calibration for SQL

The β hyperparameter requires domain-specific calibration. Your Week 44 dataset has:
- "Easy" pairs: syntax error rejected vs. correct chosen → strong signal, aggressive β is fine
- "Hard" pairs: both execute, one is semantically wrong → subtle signal, high β (conservative) needed

Start with β = 0.1 (TRL default). Watch:
- `reward_margin` > 0.5 at convergence: good
- `reward_margin` < 0.2: increase learning rate or decrease β
- Model producing refusals mid-training: β is too low, increase to 0.2–0.3

### Evaluation Design

You need a held-out test set that was NOT used in preference pair generation. Evaluation should report:

1. **Execution accuracy:** What fraction of generated SQL queries execute without error on the Postgres DB?
2. **Exact match accuracy:** What fraction of generated SQL queries exactly match the reference SQL?
3. **Semantic accuracy (preferred):** What fraction return the same rows as the reference SQL?

For a 200-query test set, this takes about 20 minutes to run. Log all results.

The acceptance criterion for this week: v2 outperforms v1 on execution accuracy. If v2 is worse, you have a problem to fix before Week 46.

### Common DPO Failure Modes in the SQL Domain

**Mode 1: DPO loss goes negative.** This happens when the reward_margin inverts — rejected SQL has higher log-probability than chosen SQL under the training model. Usually caused by: β too high (over-constraining), or the preference data is mislabeled (chosen is actually worse). Fix: verify 10 random pairs manually; lower β to 0.05.

**Mode 2: Model starts generating schema-unrelated SQL.** The model discovers that certain generic SQL patterns (like `SELECT * FROM information_schema.tables`) always execute and score as "chosen." Fix: filter out any "chosen" examples that use information_schema or pg_catalog tables.

**Mode 3: SQL output length changes dramatically.** DPO can cause the model to produce very short or very long completions because it is optimizing log-probability of specific tokens. If rejected examples are short ("syntax error in line 1") and chosen are long, the model may learn to be verbose regardless of query complexity. Fix: add a length normalization when computing log-probabilities (TRL has `length_normalization` flag).

**Mode 4: val loss increasing while train loss decreases.** Classic overfitting on a small preference dataset. DPO datasets are typically smaller than SFT datasets. Fix: reduce training epochs (1 epoch is often enough), add dropout to LoRA, or reduce LoRA rank.

### Building the Eval Pipeline

```python
# Pseudocode for eval
results = []
for prompt, reference_sql in test_set:
    generated_sql = model.generate(prompt)
    exec_result = execute_sql(generated_sql, db_conn)
    ref_result = execute_sql(reference_sql, db_conn)
    results.append({
        "prompt": prompt,
        "generated": generated_sql,
        "executes": exec_result["success"],
        "correct": exec_result["rows"] == ref_result["rows"],
    })

exec_acc = sum(r["executes"] for r in results) / len(results)
sem_acc = sum(r["correct"] for r in results) / len(results)
print(f"Execution accuracy: {exec_acc:.1%}")
print(f"Semantic accuracy: {sem_acc:.1%}")
```

## Connections

Builds on: Week 43 (DPO mechanics), Week 44 (the preference dataset).

Week 46: If DPO succeeds on easy queries but not hard queries, this motivates GRPO's online reward approach — GRPO will see new rollouts for hard queries at training time.

Week 50 (Iteration): If v2-dpo is worse than v1, you will spend Week 50 fixing it. The diagnosis notes you write this week are the input to that iteration.

## Common Misconceptions

- "DPO will automatically generalize to query types not in the preference dataset." It will not. DPO optimizes the log-probability of patterns seen in training. Novel query structures require novel data.
- "A higher reward_margin always means a better model." Not if the margin was achieved by making rejected SQL even less likely rather than making chosen SQL better. Check whether chosen log-probs increase, not just whether the margin grows.
- "DPO loss going negative is a training failure." It is a warning sign, not necessarily a failure. Inspect the data first.

## Time Allocation (6–8 hours)

- 30 min: Review Week 43 training config and adapt for 7B model + your preference dataset
- 4–5 hours: Run DPO training (this is compute-heavy; Colab Pro A100 needed)
- 1–1.5 hours: Build eval pipeline and run it on v1 and v2
- 30 min: Write eval report comparing v1 vs. v2 across query difficulty tiers
