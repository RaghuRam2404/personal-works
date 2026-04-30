# Week 62 Assignment Solutions

## Task 1 — API Caching Pattern

```python
import json, hashlib, os

CACHE_FILE = "api_cache.json"
cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}

def call_api_cached(model_type, model_name, prompt, system_prompt):
    cache_key = hashlib.md5(
        f"{model_type}:{model_name}:{system_prompt}:{prompt}".encode()
    ).hexdigest()
    
    if cache_key in cache:
        return cache[cache_key]
    
    result = call_api_model(model_type, model_name, prompt, system_prompt)
    cache[cache_key] = result
    
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
    
    return result
```

This pattern ensures: (a) you never pay twice for the same API call; (b) if the script crashes mid-run, you can resume without losing progress; (c) results are deterministic and reproducible.

## Task 3 — Prompt Differences to Document

Different models need slightly different system prompts for best performance:

```python
SYSTEM_PROMPTS = {
    "local": "You are an expert PostgreSQL/TimescaleDB engineer. Return only the SQL query, no explanation.",
    "gpt-4o": "You are an expert PostgreSQL/TimescaleDB SQL engineer. Respond with ONLY the SQL query. No explanation, no markdown fences, no comments.",
    "claude": "You are an expert PostgreSQL/TimescaleDB engineer. Return only the SQL query as plain text, nothing else.",
    "sqlcoder": "",  # SQLCoder has its own prompt format — use their official template
}
```

Document these differences in `head_to_head_comparison.md`. Reviewers will ask if prompts were fair.

---

## Common Gotchas

- **SQLCoder expects a specific prompt template.** Defog's SQLCoder was trained with a specific prompt format (not generic instruction-following). Use their official template from the model card, not your generic system prompt. Using the wrong template can reduce SQLCoder accuracy by 10–20%.
- **API rate limits during 200-example runs.** Add `time.sleep(0.5)` between API calls. Use `tenacity` library for automatic retry with exponential backoff.
- **DeepSeek-Coder-V2-Lite is 16B — won't fit on 24GB GPU at fp16.** Use 4-bit loading or route through Together AI's API (~$0.0002/1K tokens, negligible cost).
- **GPT-4o sometimes adds markdown fences even with explicit instructions.** Add SQL extraction logic that strips ```sql ... ``` fences before execution.
- **Result comparison edge cases.** GPT-4o may return `SELECT NULL` for empty results; your model may return `SELECT * FROM ... WHERE 1=0`. Both are technically "no result" but compare differently. Add a normalization step.

---

## How to Verify You Did It Right

1. 6 × `results_*.json` files exist, each with the same schema and 200 entries
2. Identical example IDs appear in all 6 result files (same questions evaluated for all models)
3. GPT-4o results use `api_cache.json` so they can be reproduced without additional API cost
4. `head_to_head_comparison.md` has all 6 models with bootstrapped CI on your model's score
5. "Where we win" section has ≥ 5 genuine examples where your model is correct and GPT-4o is not
6. Total API spend documented (ask for receipt from OpenAI/Anthropic dashboard)
