# Week 47 Assignment — Design the SQL Reward Function

## Setup Checklist

- [ ] Postgres DB accessible (local or cloud)
- [ ] `execute_sql()` from Week 44 available
- [ ] At least 200 prompts with reference SQL for diagnostic testing
- [ ] Python environment with `psycopg2-binary`, `re`, `torch`
- [ ] Your v2-dpo model loaded (this is the GRPO starting checkpoint)

---

## Task 1 — Implement the Full SQL Reward Function

**Goal:** Write a production-quality reward function ready for GRPO training.

**Requirements:**
- Implement `extract_sql(text: str) -> str | None` — extracts SQL from model output (handles code fences, bare SQL)
- Implement `sql_reward_fn(completions, prompts, expected_outputs=None, **kwargs) -> list[float]`
- The reward function must implement the 4-level hierarchy:
  - 0.0: syntax error or no SQL extracted
  - 0.1: SQL parsed (no syntax error) but runtime error
  - 0.2: SQL executes and returns at least 1 row
  - 0.5: SQL executes and row count matches expected (within ±1)
  - 1.0: SQL executes and rows exactly match expected (set equality)
- Add anti-hack guards:
  - Return 0.0 if SQL contains information_schema, pg_catalog, or pg_stat
  - Return 0.0 if SQL returns more than 5× the expected row count
  - Return 0.0 if SQL does not start with SELECT or WITH (after extraction)
- Timeout: 2000ms per query
- Return 0.0 (not exception) for any error

**Deliverable:** `week-47-grpo/reward_fn.py`

**Hints:**
```python
def sql_reward_fn(completions, prompts, expected_rows=None, db_dsn="", **kwargs):
    rewards = []
    for i, (completion, prompt) in enumerate(zip(completions, prompts)):
        sql = extract_sql(completion)
        expected = expected_rows[i] if expected_rows else None
        reward = _score_sql(sql, expected, db_dsn)
        rewards.append(reward)
    return rewards

def _score_sql(sql, expected, db_dsn):
    if sql is None: return 0.0
    if any(blocked in sql.upper() for blocked in ["INFORMATION_SCHEMA", "PG_CATALOG"]):
        return 0.0
    result = execute_sql(sql, db_dsn)
    if not result["success"]:
        # Differentiate syntax vs runtime
        if "syntax" in result["error"].lower(): return 0.0
        return 0.1  # Runtime error (wrong table/column) but parseable
    if result["row_count"] == 0: return 0.1
    if expected is None: return 0.2   # Can't verify semantics
    if result["row_count"] == len(expected): return 0.5
    if set(result["rows"]) == set(expected): return 1.0
    return 0.2  # Executes but wrong
```

---

## Task 2 — Diagnostic Test

**Goal:** Validate the reward function produces non-degenerate rewards on your training distribution.

**Requirements:**
- Load 100 prompts from your training set (with reference SQL)
- For each prompt, generate 4 completions from your v2-dpo model (use temperature=0.7, top_p=0.9)
- Run all 400 completions through `sql_reward_fn`
- Report:
  - % of completions scoring 0.0 (syntax/extraction failure)
  - % scoring 0.1 (runtime error)
  - % scoring 0.2 (executes, unverified)
  - % scoring 0.5 (correct row count)
  - % scoring 1.0 (exact match)
  - % of prompts where all 4 completions got the same reward (zero-gradient prompts)
- Reward distribution goal: at least 10% scoring > 0.2, at most 60% scoring 0.0

**Deliverable:** `week-47-grpo/reward_diagnostics.md` — table with the statistics above

---

## Task 3 — Reward Hacking Audit

**Goal:** Actively try to hack your own reward function before training does.

**Requirements:**
- Write 5 SQL queries specifically designed to score high on your reward function WITHOUT being good SQL. Examples:
  - `SELECT * FROM your_largest_table LIMIT 1000`
  - `WITH t AS (SELECT 1) SELECT * FROM t`
  - `SELECT NULL::integer AS id, NULL::text AS name`
- Run each through your reward function
- For any that score > 0.1: add a guard to your reward function to block it
- Document each hack and its guard in `week-47-grpo/reward_hacking_audit.md`

**Deliverable:** `week-47-grpo/reward_hacking_audit.md` + updated `reward_fn.py`

---

## Task 4 — GRPO Training Script (Setup Only — Do Not Train)

**Goal:** Write the complete GRPO training script. Verify it initializes without errors. Save it for Week 48.

**Requirements:**
- Model: your v2-dpo checkpoint (`<your-handle>/postgres-sqlcoder-7b-v2-dpo`)
- Trainer: Unsloth + TRL GRPOTrainer
- Config:
  - num_generations (K): 8
  - max_completion_length: 256
  - learning_rate: 5e-7
  - batch_size: 1 per device (GRPO with K=8 requires ~40GB VRAM with 7B model)
  - temperature: 0.7 (for rollout generation)
  - KL penalty β: 0.05
  - W&B project: `week-48-grpo-sql`
- Verify: `trainer.train()` starts and completes at least 1 step without crashing
- Save: `week-47-grpo/grpo_train.py`

---

## Stretch Goals

- Add a "reasoning bonus": small +0.1 if the completion contains a reasoning step before the SQL (e.g., "-- This query joins orders with users..."). Verify it does not dominate the main reward.
- Implement a diversity check: if 6 of 8 completions in a group are identical (exact string match), reduce their reward by 0.1. This prevents mode collapse.
- Plot the reward distribution from Task 2 as a histogram. Does it look like a multi-modal distribution (spikes at 0, 0.2, 0.5, 1.0)? If it is all at 0, your model needs more SFT before GRPO.
