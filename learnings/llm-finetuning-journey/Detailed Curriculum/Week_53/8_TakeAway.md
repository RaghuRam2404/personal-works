# Week 53 TakeAway — Data Quality Strategy

**One-liner:** Quality, diversity, and contamination-free curation beat raw data volume every time.

---

## Key Principles

- LIMA: 1,000 diverse, verified examples ≈ 50,000 mediocre ones for alignment fine-tuning
- Tulu 3: track skill coverage explicitly; balanced taxonomy > raw count
- Execution correctness is necessary but not sufficient — add semantic correctness checks

---

## Deduplication (MinHash)

```python
from datasketch import MinHash, MinHashLSH

def make_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    tokens = text.lower().split()
    for i in range(len(tokens) - 4):
        m.update(" ".join(tokens[i:i+5]).encode("utf8"))
    return m

lsh = MinHashLSH(threshold=0.85, num_perm=128)
# Insert examples, then query for near-duplicate clusters
```

## SQL Feature Audit

```python
import sqlglot

def ast_depth(sql):
    try:
        tree = sqlglot.parse_one(sql, dialect="postgres")
        return sum(1 for _ in tree.walk())
    except:
        return -1  # parse failure = likely bad SQL
```

---

## Decision Rules

- If skill count < 500 examples → target ≥ 2,000 in v3
- If near-duplicate rate > 20% → audit before adding more data
- If teacher execution rate < 60% for a skill → switch to few-shot augmentation mode
- If any training example shares 8+ consecutive tokens with a benchmark test question → exclude
- Difficulty target: 20% Easy / 35% Medium / 30% Hard / 15% Expert

---

## Numbers to Remember

- LIMA used exactly 1,000 examples across 19 task categories
- Tulu 3 deduplication threshold: exact + near-duplicate (Jaccard > 0.8)
- MinHash num_perm = 128: good balance of accuracy vs. speed for < 1M examples
- Target at least 40 distinct schemas in v3
- TimescaleDB target: ≥ 3,000 examples (from 412 in v2)

---

## Red Flags

- Any single SQL construct > 40% of your dataset (it is then a one-trick model)
- All training examples from ≤ 5 schemas (memorizes column names, not SQL reasoning)
- Teacher execution rate < 50% overall (your prompts are under-constrained)
- Data card missing contamination exclusion list (evaluation will be untrustworthy)
