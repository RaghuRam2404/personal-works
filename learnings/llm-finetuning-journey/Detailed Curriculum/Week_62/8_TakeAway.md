# Week 62 TakeAway — Head-to-Head Evaluation

**One-liner:** Same prompts, same databases, same result comparison for all models; cache all API calls; report CI.

---

## API Cache Pattern

```python
cache_key = hashlib.md5(f"{model}:{system}:{prompt}".encode()).hexdigest()
if cache_key in cache: return cache[cache_key]
result = call_api(...)
cache[cache_key] = result
json.dump(cache, open("api_cache.json","w"))
```

## McNemar's Test (2-model comparison)

```python
# b = your model correct, other wrong
# c = your model wrong, other correct
stat = (abs(b-c)-1)**2 / (b+c)
# chi2 critical at p=0.05: 3.84
# significant if stat > 3.84
```

## Cost-Per-Correct-Query

```python
def cost_per_correct(accuracy, cost_per_query):
    return cost_per_query / accuracy

# GPT-4o: $0.03 / 0.83 = $0.036 per correct query
# Your model (local): $0 / 0.76 = $0 per correct query
```

---

## Evaluation Table Template

| Model | Params | Custom | TimescaleDB | BIRD-100 | $/query |
|-------|--------|--------|-------------|----------|---------|
| Yours | 7B | X±Y | X | X | $0 |
| Base Qwen | 7B | X | X | X | $0 |
| SQLCoder | 7B | X | X | X | $0 |
| DeepSeek-Lite | 16B | X | X | X | $0 |
| Claude 3.5 | — | X | X | X | $0.02 |
| GPT-4o | — | X | X | X | $0.03 |

---

## Decision Rules

- Use identical system prompts for all local models; document any API model prompt differences
- Use SQLCoder's official prompt template — do NOT use your generic template for it
- Report wins/losses with McNemar's test, not just raw accuracy
- 200 examples → ±14pp CI; need 500+ for ±7pp CI
- TimescaleDB subset is your domain claim — report it separately even if n=50

---

## Numbers to Remember

- McNemar's chi-squared critical value (p=0.05): 3.84
- 200 examples → 95% CI ≈ ±7pp (bootstrapped)
- 50 examples → 95% CI ≈ ±14pp
- GPT-4o typical cost for 200 SQL queries: ~$6–12
- Temperature for eval: 0 (greedy, deterministic)

---

## Red Flags

- SQL-Coder accuracy using your generic prompt: likely 10–20pp below official template
- GPT-4o outputs have markdown fences: strip before execution
- Comparison table with no CI: appears precise but is not trustworthy
- "We win on TimescaleDB" claim without statistical significance note: reviewers will catch it
