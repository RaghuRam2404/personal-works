# Week 50 — Iteration Week 1: Fix Bugs, Expand Dataset, Retry

## Learning Objectives

By the end of this week, you will be able to:

- Diagnose why your v3 model did not meet eval targets (if applicable) using a structured debugging framework
- Apply specific targeted fixes to the reward function, training data, or hyperparameters based on diagnosis
- Expand your SQL training dataset to better cover complex query types
- Run one or more targeted GRPO experiments to improve on the identified weaknesses
- Produce a documented iteration log with hypothesis, experiment, and result for each change

## The Purpose of Iteration Weeks

Real ML work is not a straight line from "design" to "working model." This week and next (Week 51) are structured time for the iteration loop that real training runs require. The curriculum allocates two full weeks here because:

1. Your first GRPO run (Week 48) produced v3, which may or may not meet the acceptance criterion.
2. Even if v3 passes, you can likely get another 5–10pp by fixing specific weaknesses.
3. The iteration process itself is the skill — knowing how to diagnose and fix training issues is more valuable than any single model.

## Iteration Framework

Use this structured process for every experiment:

```
1. Hypothesis: "I believe v3 fails on complex queries because the GRPO prompt set contained only 10% complex examples."
2. Experiment: "I will add 200 complex queries to the GRPO training set and retrain for 300 steps from the v3 checkpoint."
3. Metric: "Complex query execution accuracy should increase from X% to X+5%."
4. Result: [run the experiment] "Complex query accuracy went from 48% to 57%."
5. Analysis: "The hypothesis was correct. 300 steps was enough. Stopping here."
```

Every experiment must have a hypothesis. Experiments without hypotheses are wasted compute.

## Common v3 Failure Modes and Fixes

### Failure Mode 1: v3 not better than v2 on complex queries

**Diagnosis:** GRPO training prompt set is too simple. Most of the gradient came from easy/medium prompts.

**Fix:** Filter the training prompt set to include only complex queries (3+ JOINs, CTEs, window functions, TimescaleDB hyperfunctions) and run 300 additional GRPO steps from the v3 checkpoint. Do not restart from v2 — v3 already has improvements on easy/medium queries.

**Expected result:** Complex query execution accuracy increases by 5–15pp over 300 steps.

### Failure Mode 2: v3 has higher execution accuracy but lower semantic accuracy

**Diagnosis:** Reward function was optimizing for execution success, not correctness. The model learned to generate queries that execute but return wrong data.

**Fix 1 (reward):** Add reference SQL for more training prompts. Change the reward function so that level-0.2 (executes, unverified) becomes level-0.05 and level-1.0 (exact match) becomes the dominant signal. Retrain for 500 steps.

**Fix 2 (data):** Add prompt/expected_output pairs where the expected output is a small, specific set of rows (not `SELECT *` results). This forces the model to generate precise WHERE clauses.

**Expected result:** Semantic accuracy increases 3–8pp; execution accuracy stays flat or improves.

### Failure Mode 3: reward_std collapsed to near-zero after 500 steps

**Diagnosis:** Model mode collapse — it found one reliable SQL pattern and repeats it. The group of K completions is homogeneous.

**Fix 1 (generation):** Increase temperature from 0.7 to 0.9–1.0 during GRPO rollout. This forces more diverse completions.

**Fix 2 (data):** Switch to harder prompts where the current model succeeds 20–50% of the time (not 90%+).

**Fix 3 (algorithmic):** Apply the DAPO "entropy bonus" — add a small entropy regularization term to the GRPO loss: `L = L_GRPO - c · H(π_θ)`. This explicitly prevents the model from becoming too deterministic.

**Expected result:** reward_std recovers to > 0.1, and mean_reward starts improving again.

### Failure Mode 4: High KL divergence (> 10 nats) causing refusals or incoherence

**Diagnosis:** Model drifted too far from the SFT reference.

**Fix:** Increase β from 0.05 to 0.15 and retrain from the last checkpoint before KL exploded. Also, consider adding a "grammar" check to the reward function: if the model's output is not parseable as SQL, return 0.0 (this prevents exploiting the reward by generating gibberish that the tokenizer interprets as SQL).

### Failure Mode 5: v3 failed on the acceptance criterion but improved on specific domains

**Diagnosis:** The evaluation metric is too coarse. v3 may be significantly better on TimescaleDB-specific queries but worse on multi-table JOINs.

**Fix:** Disaggregate the evaluation. Split the 200-query test set into sub-categories (single-table, JOIN, CTE, TimescaleDB) and report separately. If v3 is better on 3/4 sub-categories, run targeted GRPO for the failing sub-category.

## Dataset Expansion

If your Week 44 preference dataset is a limiting factor, expand it:

1. Add 500 more prompts focused on your identified weakness (complex JOINs, TimescaleDB)
2. Generate candidates from both v1 and v3 (not just v1 and base Qwen)
3. Label using execution and reference SQL comparison
4. Add to the existing dataset (do not replace — stack the new pairs)

For GRPO prompt expansion:
1. Generate 300 complex SQL prompts synthetically (using GPT-4o if budget allows, or by adapting Spider's hard split)
2. Filter to prompts where v3 succeeds 20–60% of the time (use diagnostic testing from Week 47)
3. Add to the GRPO training set

## Running Targeted Experiments

For each fix, estimate the compute cost before running:
- 300 GRPO steps with K=8 on A100 80GB: ~2 hours (~$6–8 on RunPod)
- 500 DPO steps on 2000 pairs: ~1 hour on Colab Pro (~$1 at pro-tier compute)

Total iteration budget: use remaining from Phase 5 allocation (Phase 5 had $60 budget; Weeks 47–48 spent ~$10–15; Week 50–51 have ~$15–25 remaining).

## Connections

Uses: All of Phase 5 (v3 model, reward function, eval pipeline).

Week 51: Continues this iteration. Pick the best model from Weeks 50–51 combined.

Week 52 (Gate): The best model from Weeks 50–51 is evaluated at the Gate. The Gate requires v3 to beat v1 on execution correctness.

## Common Misconceptions

- "Iteration means running the same experiment again and hoping for different results." No. Every iteration must have a hypothesis and a specific change.
- "I should restart from v1 if v3 is not good enough." Do not throw away 1000 GRPO steps. Patch from v3 unless you have evidence of catastrophic forgetting.
- "The reward function is correct if it compiled without errors." Test it diagnostically on 100 completions. Bugs in reward functions are common and subtle.

## Time Allocation (6–8 hours)

- 1 hour: Diagnose v3 failures using the eval report from Week 48.
- 30 min: Write your iteration hypothesis for the highest-impact fix.
- 1 hour: Implement the fix (reward function change or dataset expansion).
- 3–4 hours async: Run the targeted GRPO experiment on RunPod.
- 1 hour: Evaluate the result. Write the iteration log.
- 30 min: Plan Week 51 based on this week's result.
