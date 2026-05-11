# Week 47 Assignment Solutions

## Task 1 — Full Reward Function

```python
import re
import psycopg2

def extract_sql(text: str) -> str | None:
    """Extract SQL from a model completion that may include markdown fences."""
    # Try ```sql ... ``` fence first
    m = re.search(r'```(?:sql|SQL)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m:
        sql = m.group(1).strip()
        if sql.upper().startswith(("SELECT", "WITH")):
            return sql
    # Try bare SQL (SELECT or WITH at start of a line)
    m = re.search(r'(?:^|\n)((?:SELECT|WITH)\b.*?)(?:\n\n|\Z)', text, 
                  re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None

BLOCKED_SCHEMAS = {"INFORMATION_SCHEMA", "PG_CATALOG", "PG_STAT", 
                   "PG_CLASS", "PG_NAMESPACE"}

def _score_sql(sql: str | None, expected: list | None, db_dsn: str) -> float:
    if sql is None:
        return 0.0
    # Anti-hack: block system schema queries
    sql_upper = sql.upper()
    if any(blocked in sql_upper for blocked in BLOCKED_SCHEMAS):
        return 0.0
    # Anti-hack: require real SQL operation
    if not sql_upper.lstrip().startswith(("SELECT", "WITH")):
        return 0.0
    
    result = execute_sql(sql, db_dsn, timeout_ms=2000)
    
    if not result["success"]:
        err = result["error"].lower()
        if "syntax" in err or "parse" in err:
            return 0.0  # Syntax error
        return 0.1  # Runtime error (table not found, wrong column, etc.)
    
    if result["row_count"] == 0:
        return 0.1  # Executes but empty — suspicious
    
    if expected is None:
        return 0.2  # Executes, can't verify semantics
    
    # Anti-hack: penalize wildly wrong row counts
    if result["row_count"] > max(len(expected) * 5, 10):
        return 0.1  # Too many rows — SELECT * without WHERE
    
    # Semantic verification
    actual_rows = set(tuple(r) for r in result["rows"])
    expected_rows = set(tuple(r) for r in expected)
    
    if actual_rows == expected_rows:
        return 1.0  # Exact match
    if result["row_count"] == len(expected):
        return 0.5  # Right count, wrong values
    return 0.2  # Executes but wrong


def sql_reward_fn(completions: list[str], prompts: list[str],
                  expected_rows: list | None = None, 
                  db_dsn: str = "", **kwargs) -> list[float]:
    rewards = []
    for i, (completion, prompt) in enumerate(zip(completions, prompts)):
        sql = extract_sql(completion)
        expected = expected_rows[i] if expected_rows is not None else None
        reward = _score_sql(sql, expected, db_dsn)
        rewards.append(float(reward))
    return rewards
```

**Common gotchas:**
- `extract_sql` must handle the model repeating the prompt before the SQL — use `\n` anchors to find SQL at the start of a line
- Empty result (row_count=0) should NOT be reward 0.2 — it is almost always wrong for SQL questions; use 0.1
- Row set equality requires hashable types: convert each row to a tuple before set operations
- The anti-hack for row count needs a minimum floor (`max(len(expected)*5, 10)`) — if expected has 0 rows (aggregate with no matches), do not penalize a query that returns 1 row

---

## Task 2 — Expected Diagnostic Distribution

For a well-trained v2-dpo model on SQL:

| Reward | Expected % |
|---|---|
| 0.0 (syntax/no SQL) | 20–35% |
| 0.1 (runtime error or empty) | 15–25% |
| 0.2 (executes, unverified) | 10–20% |
| 0.5 (correct row count) | 5–15% |
| 1.0 (exact match) | 15–30% |

If 0.0 rate > 60%: your model is not yet ready for GRPO. Do another SFT epoch or use a lower temperature.

If 1.0 rate > 50%: GRPO will have low gradient (mostly zero advantages for easy prompts). Increase prompt difficulty.

Ideal distribution: 30% scoring 0.0, 40% scoring 0.1–0.2, 30% scoring 0.5–1.0.

---

## Task 4 — GRPO Script Minimal Config

```python
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "<your-handle>/postgres-sqlcoder-7b-v2-dpo",
    max_seq_length=1024,
    dtype=torch.bfloat16,
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, target_modules="all-linear"
)

config = GRPOConfig(
    num_generations=8,          # K
    max_completion_length=256,
    learning_rate=5e-7,
    per_device_train_batch_size=1,
    temperature=0.7,
    kl_coef=0.05,
    report_to="wandb",
    run_name="week-48-grpo-sql",
    bf16=True,
)

trainer = GRPOTrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,  # Must have 'prompt' column
    reward_fn=sql_reward_fn,       # Your reward function from Task 1
    processing_class=tokenizer,
)
# Do NOT call trainer.train() this week — save for Week 48
```

---

## How to Verify You Did It Right

1. `extract_sql("```sql\nSELECT * FROM users\n```")` returns `"SELECT * FROM users"`.
2. `sql_reward_fn(["SELECT * FROM information_schema.tables"], ["..."], ...)` returns `[0.0]`.
3. Diagnostic distribution is not all-zero (your model generates at least some valid SQL).
4. Reward hacking audit: each of the 5 hack queries you wrote gets reward ≤ 0.1.
5. GRPO training script initializes without import errors or shape mismatches.
