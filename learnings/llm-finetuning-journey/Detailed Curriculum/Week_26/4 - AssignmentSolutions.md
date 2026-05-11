# Week 26 Assignment Solutions

## Task 2 — Key Snippet: Tier 1 Processing Pipeline

```python
from datasets import load_dataset
from quality_filter import sql_quality_filter, filter_for_postgres_compat
from converters import spider_to_chatml
from datasketch import MinHash, MinHashLSH
import json, re

def get_spider_schema_str(tables_dict: dict, db_id: str) -> str:
    """Build a CREATE TABLE schema string from Spider's table metadata."""
    # Spider stores schema as dict with 'column_names_original', 'table_names_original', etc.
    schema_lines = []
    for i, table in enumerate(tables_dict.get('table_names_original', [])):
        cols = [(c[1], tables_dict['column_types'][j])
                for j, c in enumerate(tables_dict['column_names_original'])
                if c[0] == i]
        col_defs = ", ".join(f"{name} {dtype.upper()}" for name, dtype in cols)
        schema_lines.append(f"CREATE TABLE {table} ({col_defs});")
    return "\n".join(schema_lines)

def process_tier1():
    ds = load_dataset("spider")
    results = []
    seen_questions = set()
    lsh = MinHashLSH(threshold=0.7, num_perm=128)

    for ex in ds['train']:
        schema_str = get_spider_schema_str(ex['db_id'], ex['db_id'])
        # Note: actual Spider schema extraction requires tables metadata
        # The simplified version above illustrates the pattern

        chatml = spider_to_chatml(ex, schema_str)
        sql = chatml['messages'][2]['content']
        question = chatml['messages'][1]['content']

        # Quality filter
        if not sql_quality_filter({'output': sql, 'instruction': question}):
            continue
        if not filter_for_postgres_compat(sql):
            continue

        # MinHash dedup
        m = MinHash(num_perm=128)
        for w in question.lower().split():
            m.update(w.encode())
        key = f"spider_{len(results)}"
        if len(lsh.query(m)) == 0:
            lsh.insert(key, m)
            results.append(chatml)

        if len(results) >= 1500:
            break

    with open("tier1_examples.jsonl", "w") as f:
        for ex in results:
            f.write(json.dumps(ex) + "\n")
    print(f"Tier 1 complete: {len(results)} examples")
```

**Common gotchas:**
- Spider's schema metadata is complex — `db_id` is just a name; the actual schema tables are loaded separately from the Spider database files
- The simplest approach: use the `spider_schema` field if present, or pre-generate schema strings from Spider's JSON tables
- BIRD format differs: uses `SQL` (not `query`), has `evidence` field, and has actual database files you can query

---

## Task 4 — Key Snippet: Dataset Build and Publish

```python
from datasets import Dataset, DatasetDict
import json, random

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def build_and_publish():
    tier1 = load_jsonl("tier1_examples.jsonl")    # 2000
    tier2 = load_jsonl("hand_written_examples.jsonl")  # 100
    tier3 = load_jsonl("tier3_examples.jsonl")    # 2900

    all_examples = tier1 + tier2 + tier3
    random.seed(42)
    random.shuffle(all_examples)

    split_idx = int(0.8 * len(all_examples))
    train = all_examples[:split_idx]
    val   = all_examples[split_idx:]

    print(f"Train: {len(train)}, Val: {len(val)}, Total: {len(all_examples)}")

    # Statistics
    sql_lens = [len(ex['messages'][2]['content'].split()) for ex in train]
    print(f"Avg SQL tokens: {sum(sql_lens)/len(sql_lens):.1f}")

    # Publish
    train_ds = Dataset.from_list(train)
    val_ds   = Dataset.from_list(val)
    ds_dict  = DatasetDict({"train": train_ds, "validation": val_ds})
    ds_dict.push_to_hub("<your-handle>/postgres-sql-v1", private=True)
    print("Published to HuggingFace Hub")
```

**Expected dataset statistics (approximate):**
- Train: 4,000 examples
- Val: 1,000 examples
- Average SQL tokens: 25–60 tokens (short queries dominate; some CTEs reach 100+)
- SQL keyword distribution: ~70% SELECT, ~10% WITH (CTE), ~8% INSERT, ~5% UPDATE, ~4% DELETE, ~3% CREATE
- Hand-written examples: 100 (2% of total but highest quality)

**Red flags in dataset:**
- Average SQL tokens < 15: too many trivially simple queries ("SELECT * FROM table")
- Single schema appears in > 20% of examples: not diverse enough
- All examples have identical system prompt but different user content — this is correct, not a red flag
