# Week 47 — GRPO with Executable Rewards on SQL (Week 1 of 2)

## Learning Objectives

By the end of this week, you will be able to:

- Design a multi-level SQL reward function that resists reward hacking
- Implement and test the reward function against a real Postgres instance
- Set up the GRPO training pipeline with Unsloth and TRL
- Validate that the reward function produces correct reward distributions on a diagnostic sample
- Identify reward hacking patterns specific to SQL and build defenses against them

## This Week's Focus: Reward Function Design

Week 47 and Week 48 together form the most important two-week sprint in the course. The model you produce (v3-grpo) is the one most likely to outperform GPT-4 on your domain. This week you design and validate the reward function. Week 48 you run it.

Your reward function is the most critical design decision in GRPO training. A poorly designed reward can cause:
- Reward hacking: the model discovers SQL patterns that score high without being correct
- Sparse signal: 95% of training steps produce zero gradient
- Wrong optimization target: the model optimizes what you measure, not what you want

Design it carefully.

## Concepts

### Multi-Level Reward Hierarchy

A binary {0, 1} reward for SQL execution is too sparse — if only 10% of your model's completions execute correctly, 90% of steps get zero gradient. Design a hierarchy:

**Level 4 (reward = 1.0):** SQL executes AND returns rows matching the expected output (semantic correctness). This is the gold standard.

**Level 3 (reward = 0.5):** SQL executes AND returns the correct number of rows, but some values differ (structural correctness). Useful signal even when not exactly right.

**Level 2 (reward = 0.2):** SQL executes without error AND returns some rows (execution correctness without semantic verification). This is the minimum bar.

**Level 1 (reward = 0.1):** SQL parses without syntax error but fails at execution (runtime error — e.g., wrong column name). Shows the model got the structure right.

**Level 0 (reward = 0.0):** SQL has a syntax error. Unparseable output. Starting point.

This hierarchy gives the model gradient signal at every level of quality, reducing the sparse-reward problem.

### Reward Hacking in SQL — Known Patterns

Before you implement, anticipate what a model might do to maximize reward without generating good SQL:

**Hack 1: Always SELECT * with no WHERE clause.** `SELECT * FROM users` often executes (reward ≥ 0.2) and may accidentally match row counts. Defense: reward 0 for queries that return more than 2× the expected row count.

**Hack 2: information_schema queries.** `SELECT table_name FROM information_schema.tables` always executes. Defense: reward 0 for queries touching information_schema, pg_catalog, pg_stat, or system schemas.

**Hack 3: SELECT NULL or literal values.** `SELECT 1` always executes and returns a row. Defense: require that the query references at least one actual table from your schema.

**Hack 4: Repeat the best known pattern.** After GRPO training, the model might converge to a single SQL template that covers many cases. Defense: enforce a diversity metric across the K completions (optional but powerful).

**Hack 5: Timeout exploitation.** If your timeout is generous, a model might generate slow queries that time out but do not get penalized (because timeout = non-execution = reward 0, which is the baseline). This is not a gain for the model but it slows training. Defense: keep timeout to 2 seconds.

### The Reward Function Contract

Your reward function must be:
- **Deterministic:** Same (prompt, completion) pair always returns the same reward.
- **Fast:** GRPO generates K completions per prompt and evaluates all K. With K=8, your reward function is called 8× per prompt per step. At 1000 steps, that is 8000 SQL executions. Each must take < 500ms.
- **Safe:** The reward function cannot corrupt the database. Only SELECT is allowed.
- **Correct:** A reward of 1.0 must mean the SQL is genuinely correct.

### Implementing the Reward Function

Your reward function signature for TRL's GRPOTrainer:

```python
def sql_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    completions: list of K generated SQL strings for K prompts (flattened)
    prompts: list of K prompts (same prompt repeated K times for GRPO)
    Returns: list of K float rewards
    """
```

TRL's GRPOTrainer passes the `reward_fn` a flat list of completions (all K completions for all prompts in the batch, concatenated). You must reshape and evaluate.

### Extracting SQL from Model Output

Your model is a chat model — it will generate SQL surrounded by markdown fences and explanations:

```
Here is the SQL query:

```sql
SELECT * FROM users WHERE id = 1
```

You need to extract just the SQL:

```python
import re

def extract_sql(text: str) -> str | None:
    # Try SQL code fence first
    m = re.search(r'```(?:sql)?\s*\n(.*?)\n```', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Try bare SQL starting with SELECT/WITH
    m = re.search(r'\b(SELECT|WITH)\b.*', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return None
```

If extraction fails (the model generated no SQL), return reward = 0.

### Reasoning Bonus (Optional)

If your model generates a chain-of-thought before the SQL (as DeepSeek-R1 does), you can add a small bonus for having a reasoning chain:

```python
def reasoning_bonus(text: str) -> float:
    """Small bonus for generating reasoning steps before SQL."""
    has_reasoning = len(text) > 100  # Rough proxy: longer text = more reasoning
    has_sql = extract_sql(text) is not None
    if has_reasoning and has_sql:
        return 0.1  # Small bonus — do not dominate the main reward
    return 0.0
```

Be careful with reasoning bonuses: if the bonus is too large, the model learns to generate verbose text regardless of SQL quality. Keep it at ≤ 10% of the max reward.

### Reward Normalization

GRPO normalizes rewards within each group. If all your rewards are in [0, 1], the normalization works correctly. Do not scale rewards to [0, 100] — the normalization will still work but the KL loss might need re-tuning.

### Diagnostic Testing

Before connecting your reward function to the GRPO trainer, test it:
- Sample 50 prompts from your training set
- Generate 8 completions each (manually, not through GRPO)
- Run the reward function
- Verify: reward distribution is not degenerate (not all 0 or all 1)
- Verify: level-1 rewards appear (some syntax errors) and level-4 rewards appear (some correct)
- Verify: no reward hacking patterns are scoring above 0

## Connections

Builds on: Week 46 (GRPO algorithm and TRL GRPOTrainer), Week 44 (execution harness), Week 45 (eval framework).

Week 48: This is the week you actually run GRPO. The reward function you finalize this week is used there.

Week 50 (Iteration): If GRPO does not improve over DPO, the reward function is the first place to look.

## Common Misconceptions

- "The reward function should only reward perfectly correct SQL." Too sparse — you need intermediate rewards for gradient signal, especially early in training.
- "A reasoning bonus will make the model explain its SQL better." Only if the bonus is calibrated correctly. Too large and the model learns to be verbose; too small and it is ignored.
- "I should penalize (negative reward) bad SQL." In GRPO with group normalization, below-average performance already gets negative advantages. Do not add explicit negative rewards unless you have specific hack patterns to block.
- "Testing the reward function on 10 examples is enough." Test on at least 50. SQL hacks are rare by design — you need enough samples to detect them.

## Time Allocation (6–8 hours)

- 1 hour: Design your reward hierarchy on paper. List the levels, thresholds, and anti-hack rules.
- 1 hour: Implement `extract_sql()` and the execution harness.
- 2 hours: Implement the full `sql_reward_fn()` with all 4 levels.
- 1 hour: Diagnostic testing on 50 prompts. Debug any extraction or execution issues.
- 1–2 hours: Set up the GRPO training script (Unsloth + TRL). Do NOT start training yet — that is Week 48.
- 30 min: Write your reward hacking audit in `week-47-grpo/reward_hacking_audit.md`.
