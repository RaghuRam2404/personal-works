# Week 60 Assignment Solutions

## Task 1 — Reward Function Correctness Test

```python
# Sanity checks for reward function
test_cases = [
    # (description, sql, expected_range)
    ("Reference SQL", reference_sql, (2.0, 2.2)),
    ("Empty string", "", (-1.1, -0.9)),
    ("Syntax error", "SELECT FROM;", (-0.6, -0.4)),
    ("Executes but wrong rows", "SELECT * FROM sensors LIMIT 0", (0.9, 1.1)),
    ("Correct result", reference_sql, (2.0, 2.2)),
]

for desc, sql, (lo, hi) in test_cases:
    completion = f"```sql\n{sql}\n```"
    r = compute_reward(prompt, completion, reference_sql, schema_ddl, conn)
    assert lo <= r <= hi, f"FAILED {desc}: reward={r}, expected [{lo},{hi}]"
    print(f"PASS {desc}: reward={r:.2f}")
```

## Task 3 — extract_sql Helper

```python
import re

def extract_sql(completion: str) -> str | None:
    """Extract SQL from markdown fences or raw text."""
    # Try markdown fence first
    match = re.search(r"```(?:sql)?\s*(.*?)```", completion, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try raw SQL
    stripped = completion.strip()
    if stripped.upper().startswith(("SELECT","WITH","INSERT","UPDATE","DELETE","CREATE")):
        return stripped
    return None
```

## Merge and Push Final Model

```python
# After GRPO training completes
model.save_pretrained_merged(
    "postgres-sqlcoder-7b-final",
    tokenizer,
    save_method="merged_16bit",
)
# Push to Hub
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="postgres-sqlcoder-7b-final",
    repo_id="<your-handle>/postgres-sqlcoder-7b-final",
    repo_type="model",
)
```

---

## Common Gotchas

- **Reward function crashes in background thread.** TRL's GRPOTrainer may call the reward function in parallel. Use a connection pool (`psycopg2.pool.ThreadedConnectionPool`) instead of a single connection.
- **Model generates natural language instead of SQL.** The format reward should catch this. But also: ensure your prompts include "Return only the SQL query, no explanation" in the system prompt.
- **KL divergence explodes after step 200.** Increase `kl_coef` from 0.05 to 0.1 and reload from step-100 checkpoint.
- **extract_sql returns None too often.** The model has learned to not use markdown fences. Adjust your extraction logic to handle raw SQL output without fences.
- **GRPO is slower than expected.** Group size 8 means 8 forward passes per prompt. If speed is a bottleneck: reduce to group size 4 (less variance, still works) or use vLLM for generation.

---

## How to Verify You Did It Right

1. `final_model_comparison.md` is populated with actual numbers (not placeholders)
2. GRPO-final ≥ DPO-v3 on custom benchmark — if not, GRPO regressed (use DPO-v3 as the final model)
3. `postgres-sqlcoder-7b-final` on HuggingFace is a merged model (not adapter-only), with README
4. Reward function test cases all pass before training
5. W&B run `week-60-grpo-final` shows mean_reward increasing from ~0.8 to ~1.5+ over training
6. RunPod instance terminated

**If GRPO regresses:** This happens. The DPO-v3 model is your backup capstone. Report the regression in your technical report honestly — this is scientifically valid and shows you ran a real experiment, not a cherry-picked demo.
