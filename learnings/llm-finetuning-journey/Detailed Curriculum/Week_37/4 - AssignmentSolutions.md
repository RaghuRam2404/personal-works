# Week 37 Assignment Solutions

## Task 1 — Data Loading and Filtering: Key Snippets

```python
from datasets import load_dataset
import sqlparse, hashlib, json

def is_postgresql_compatible(sql):
    """Filter out MySQL-specific syntax."""
    mysql_markers = ["AUTO_INCREMENT", "TINYINT(1)", "ENGINE=", 
                     "`", "INT UNSIGNED", "DATETIME("]
    return not any(marker in sql.upper() for marker in mysql_markers)

def is_valid_sql(sql):
    try:
        parsed = sqlparse.parse(sql.strip())
        return len(parsed) > 0 and parsed[0].get_type() is not None
    except:
        return False

def dedup_key(item):
    return hashlib.md5(f"{item['question']}|||{item['answer']}".encode()).hexdigest()

# Load sql-create-context
dataset = load_dataset("b-mc2/sql-create-context", split="train")
print(f"Original: {len(dataset)}")

# Filter
dataset = dataset.filter(lambda x: is_postgresql_compatible(x["answer"]))
print(f"After MySQL filter: {len(dataset)}")

dataset = dataset.filter(lambda x: is_valid_sql(x["answer"]))
print(f"After SQL validation: {len(dataset)}")

# Dedup
seen = set()
filtered = []
for item in dataset:
    key = dedup_key(item)
    if key not in seen:
        seen.add(key)
        filtered.append(item)
print(f"After dedup: {len(filtered)}")
```

**Expected counts for sql-create-context:**
- Original: 78,577
- After MySQL filter: ~65,000 (most are compatible)
- After SQL validation: ~63,000
- After dedup: ~60,000

Sample 8,000–10,000 from this pool: `random.sample(filtered, 8000)`.

---

## Task 2 — Synthetic Generation: Prompt Example

```python
import anthropic

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_example(schema_type, feature):
    prompt = f"""Generate a PostgreSQL training example.
Schema domain: {schema_type}
SQL feature: {feature}

Output valid JSON only:
{{"schema": "CREATE TABLE ...", "question": "...", "sql": "SELECT ..."}}

Rules:
- PostgreSQL 15+ syntax only
- 2-3 tables with foreign keys
- SQL must be executable"""
    
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response.content[0].text)
```

**Cost estimate:** 200 examples × 15 features = 3K examples, ~400 tokens each ≈ 1.2M tokens.
Claude 3 Haiku: ~$0.001/1K tokens input → ~$1–3 total. Very affordable.

---

## Task 3 — TimescaleDB Examples: Template

```json
{
  "schema": "CREATE TABLE sensor_readings (time TIMESTAMPTZ NOT NULL, device_id INT, temperature FLOAT, humidity FLOAT);\nSELECT create_hypertable('sensor_readings', 'time');",
  "question": "What is the average temperature per hour for the last 24 hours?",
  "sql": "SELECT time_bucket('1 hour', time) AS hour, AVG(temperature) as avg_temp FROM sensor_readings WHERE time > NOW() - INTERVAL '24 hours' GROUP BY hour ORDER BY hour;"
}
```

---

## Task 4 — Dataset Statistics: Expected Values

After combining and deduplicating 15K examples:

| Metric | Target Range |
|---|---|
| Total examples | 15,000 |
| Training split | 14,500 |
| Validation split | 500 |
| Average tokens (full example) | 180–280 |
| Examples with JOINs | 35–45% |
| Examples with GROUP BY | 30–40% |
| Examples with subqueries | 10–20% |
| Examples with window functions | 5–15% |
| PostgreSQL-specific (JSONB, ARRAY, etc.) | 5–10% |
| TimescaleDB-specific | 0.3–1% |

---

## Common Gotchas

- **SQLite-specific syntax slips through**: `PRAGMA`, `ROWID`, `WITHOUT ROWID` — add to the MySQL filter list.
- **Synthetic LLM examples sometimes output invalid JSON**: Wrap generation in try/except and retry up to 3 times per example.
- **Token length distribution is skewed**: Long schemas (20+ columns) can push total length over 512 tokens. Use `tokenizer(text, return_length=True)` to filter before training, not just estimate by word count.
- **Week 33 held-out test set overlap**: Verify that none of the 100 held-out test examples appear in your training set. Use the same dedup hash check across the combined dataset and held-out set.

---

## How to Verify You Did It Right

- `train_15k.jsonl` contains exactly 14,500 lines
- `val_500.jsonl` contains exactly 500 lines
- No example in train or val matches any example in `held_out_test.json` (dedup check)
- `week37_dataset_stats.md` shows diversity: no SQL type exceeds 60% of the dataset
- Quick 10-step training run shows loss < 2.5 at step 10 (confirming valid formatting)
- At least 30 TimescaleDB examples in `dataset_timescale.jsonl`
