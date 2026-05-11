# Week 37 TakeAway — Building a Domain SQL Dataset

**One-liner:** 15K = 8–10K public + 3–5K synthetic + 50–100 domain-specific. Filter aggressively; diversity > volume.

---

## Dataset Quality Checklist

```python
# 1. Length filter (max 400 tokens for schema+question+sql)
def is_short_enough(item, tokenizer, max_tokens=400):
    total = len(tokenizer.encode(item["schema"] + item["question"] + item["sql"]))
    return total <= max_tokens

# 2. MySQL filter
MYSQL_MARKERS = ["AUTO_INCREMENT", "TINYINT(1)", "ENGINE=", "`", "INT UNSIGNED"]
def is_postgresql(sql):
    return not any(m in sql.upper() for m in MYSQL_MARKERS)

# 3. SQL validation
import sqlparse
def is_valid_sql(sql):
    parsed = sqlparse.parse(sql.strip())
    return len(parsed) > 0 and parsed[0].get_type() is not None

# 4. Deduplication
import hashlib
seen = set()
def is_unique(item):
    key = hashlib.md5(f"{item['question']}|||{item['sql']}".encode()).hexdigest()
    if key in seen: return False
    seen.add(key); return True
```

---

## Dataset Split Pattern

```
Total: 15,000
├── train_15k.jsonl   → 14,500 examples
└── val_500.jsonl     →    500 examples (NOT from held-out test set)

Held-out test (Week 32): 100 examples — NEVER in train or val
```

---

## Source Strategy

| Source | Target N | Quality |
|---|---|---|
| sql-create-context | 8,000–10,000 | High — expert labeled |
| gretel/gretel-text-to-sql | 3,000–5,000 | Medium — synthetic |
| Hand-crafted TimescaleDB | 50–100 | Highest — domain exact |
| wikisql (fallback) | 1,500–2,000 | Medium — simple queries |

---

## SQL Type Balance Targets

| Type | Target % |
|---|---|
| Simple SELECT (no JOIN) | 25–35% |
| SELECT with JOIN | 30–35% |
| SELECT with GROUP BY/HAVING | 20–25% |
| Subqueries / CTEs | 10–15% |
| Window functions / advanced | 5–10% |

---

## Numbers to Remember

- sqlparse validation: filter anything where `parsed[0].get_type()` returns None
- Token budget per example: aim for ≤ 400 tokens (leaves 112 for SQL output at max_seq_length 512)
- Contamination check: dedup against held-out test set by schema hash

---

## Decision Rules

- If sqlparse rejects example → remove (4% failure rate is normal for LLM-generated SQL)
- If >60% of dataset is same SQL type → resample for balance
- If any held-out test schema appears in training set → remove it from training
- Keep TimescaleDB examples weighted 2–3× (repeat them) to compensate for rarity
