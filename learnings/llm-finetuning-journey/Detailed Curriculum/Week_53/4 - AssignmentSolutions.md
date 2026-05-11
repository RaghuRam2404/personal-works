# Week 53 Assignment Solutions

## Task 2 — Key Snippets for v2 Audit

```python
import json
import re
from datasets import load_dataset
from collections import defaultdict
import sqlglot
from datasketch import MinHash, MinHashLSH

def get_sql_features(sql: str) -> dict:
    sql_upper = sql.upper()
    return {
        "has_join": "JOIN" in sql_upper,
        "has_group_by": "GROUP BY" in sql_upper,
        "has_window": "OVER (" in sql_upper or "OVER(" in sql_upper,
        "has_cte": sql_upper.strip().startswith("WITH"),
        "has_subquery": sql_upper.count("SELECT") > 1,
        "has_timescale": any(kw in sql_upper for kw in [
            "TIME_BUCKET", "FIRST(", "LAST(", "HISTOGRAM(",
            "LOCF(", "INTERPOLATE(", "COMPRESS_CHUNK"
        ]),
    }

def make_minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    tokens = text.lower().split()
    for i in range(len(tokens) - 4):
        ngram = " ".join(tokens[i:i+5])
        m.update(ngram.encode("utf8"))
    return m

# Load dataset
ds = load_dataset("json", data_files="v2_dataset.jsonl")["train"]

# Collect features
feature_counts = defaultdict(int)
schemas = set()
lsh = MinHashLSH(threshold=0.85, num_perm=128)

for i, ex in enumerate(ds):
    feats = get_sql_features(ex["sql"])
    for k, v in feats.items():
        if v:
            feature_counts[k] += 1
    schemas.add(frozenset(ex.get("tables", [])))
    mh = make_minhash(ex["question"] + " " + ex["sql"])
    lsh.insert(f"ex_{i}", mh)
```

**Expected output shape for `v2_audit_results.json`:**
```json
{
  "total": 18432,
  "features": {"has_join": 7201, "has_cte": 1203, "has_timescale": 412, ...},
  "distinct_schemas": 23,
  "near_duplicate_pairs": 1847,
  "contamination_estimate": 134
}
```

---

## Task 3 — Common Gotchas

- **Skill taxonomy too coarse.** "TimescaleDB" is one skill? No — break it into `time_bucket`, `continuous_agg`, `hyperfunctions`, `compression`, `data_retention`. Each needs its own count.
- **Contamination check too strict.** 5-gram overlap on code is expected for common SQL keywords. Use 8-gram overlap to reduce false positives, or filter at the question (NL) level only.
- **Data card is vague on schemas.** "We use PostgreSQL schemas" is not enough. Name the schemas: TPC-H, IMDB, your TimescaleDB IoT schema, etc.
- **Forgetting the label format.** v3 examples must specify the exact prompt template you plan to use for training (system prompt + user turn format). Changing this later means re-generating.

---

## How to Verify You Did It Right

1. `v2_audit_results.json` exists and all keys are populated with non-zero values
2. `data_card_v3.md` has a skill taxonomy table with at least 12 distinct SQL/TimescaleDB skills, each with a numeric target
3. `gap_analysis.md` identifies at least 5 skills where current count < 500 examples
4. Your near-duplicate count from the audit is actionable — if >20% of your v2 is near-duplicates, your v2 was weaker than you thought
5. The contamination check runs to completion (does not hang — MinHash LSH makes it O(n) not O(n²))

**A "passing" data card for v3 has:**
- At least 12 skill categories with numeric targets
- At least 40 distinct schemas planned
- A contamination exclusion list that names Spider test set, BIRD test set, and WikiSQL test set explicitly
- Difficulty targets that are not all "medium"
