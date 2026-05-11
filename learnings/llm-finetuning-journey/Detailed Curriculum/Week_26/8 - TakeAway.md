# Week 26 TakeAway — Building the PostgreSQL Dataset v1

**One-liner:** 3 tiers (Spider+BIRD + hand-written + self-instruct), sqlglot validation, execute on real Postgres, MinHash dedup, push to Hub.

---

## Dataset Build Pipeline

```
Spider/BIRD → filter (sqlglot + SQLite-compat) → MinHash dedup → 2,000 examples
Hand-written → verify against Postgres 16 → 100 examples
Self-Instruct → generate → filter → cross-dedup → 2,900 examples
→ Merge → shuffle(seed=42) → 80/20 split → push_to_hub(private=True)
```

---

## Key Code Patterns

```python
# Publish dataset to HuggingFace Hub
from datasets import Dataset, DatasetDict
ds = DatasetDict({
    "train": Dataset.from_list(train_examples),
    "validation": Dataset.from_list(val_examples)
})
ds.push_to_hub("<handle>/postgres-sql-v1", private=True)

# Docker PostgreSQL verification
import psycopg2
conn = psycopg2.connect("dbname=postgres user=postgres password=test host=localhost")

# Run Docker: docker run -d -e POSTGRES_PASSWORD=test -p 5432:5432 postgres:16
```

---

## Dataset Quality Checklist

- [ ] ≥ 98% of SQL examples parse with `sqlglot.parse(dialect="postgres")`
- [ ] ≥ 90% of hand-written examples execute on Postgres 16 without error
- [ ] Zero examples with SQLite-only syntax (GROUP_CONCAT, AUTOINCREMENT)
- [ ] Each SQL construct (JOIN, CTE, window, time_bucket, JSONB) ≥ 2% of examples
- [ ] No example appears in both train and val (shuffle before split, then no leakage)
- [ ] Assistant turn contains SQL only (no "Here is your query:" prefix)
- [ ] Every example has a schema in the user message

---

## Numbers to Remember

| Metric | Target |
|---|---|
| Total examples | 5,000 |
| Train / Val split | 80% / 20% (4,000 / 1,000) |
| Tier 1 (Spider+BIRD) | 2,000 |
| Tier 2 (hand-written) | 100 |
| Tier 3 (self-instruct) | 2,900 |
| sqlglot pass rate | ≥ 98% |
| MinHash threshold | 0.7 (Jaccard) |
| Self-Instruct generation cost | ~$5 (GPT-3.5) or free (Ollama) |

---

## Red Flags

- 85% val accuracy but 45% on fresh test questions → distribution mismatch, not true learning
- All time_bucket examples use the same interval → add diverse time intervals
- Schema name appears in > 20% of examples → not diverse enough, add more schemas
- Average SQL length < 15 tokens → too many trivial queries, apply complexity filter
- Model generates explanation text before SQL → explanation text leaked into training data
