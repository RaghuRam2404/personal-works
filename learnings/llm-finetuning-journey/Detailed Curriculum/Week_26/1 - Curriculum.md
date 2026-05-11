# Week 26 — Domain Dataset Construction Part 2: Build the PostgreSQL/TimescaleDB v1 Dataset

## Learning Objectives

By the end of this week, you will be able to:

- Execute the full 3-tier dataset construction pipeline to produce 5,000 training examples
- Validate every example with SQL parsing and schema consistency checks
- Apply MinHash deduplication to remove near-duplicate examples
- Document and publish a dataset to HuggingFace Hub with a proper dataset card
- Analyze your v1 dataset's coverage gaps and plan v2 improvements

---

## Concepts

### The Dataset Construction Workflow

This week you execute the plan from Week 25. The order of operations matters:

```
Step 1: Process Tier 1 (Spider + BIRD)
  → filter for PostgreSQL compatibility
  → convert to ChatML
  → deduplicate with MinHash (threshold=0.7)
  → target: 2,000 examples

Step 2: Finalize Tier 2 (Hand-written)
  → complete 100 hand-written examples (you wrote 20 in Week 25)
  → verify every SQL with sqlglot AND execute against a local Postgres instance
  → target: 100 examples

Step 3: Generate Tier 3 (Self-Instruct)
  → use Week 25's self_instruct.py
  → generate 5,000 candidates using your 100 hand-written examples as seeds
  → filter: sqlglot + length + SQLite-only keywords
  → deduplicate within Tier 3 AND against Tier 1+2 combined
  → target: 2,900 examples

Step 4: Merge and Final Quality Check
  → combine all tiers: 2,000 + 100 + 2,900 = 5,000 examples
  → shuffle
  → 80/20 train/val split
  → produce train.jsonl and val.jsonl

Step 5: Publish to HuggingFace Hub
  → create dataset repository
  → write dataset card (README.md)
  → push with `datasets.push_to_hub()`
```

### Building Realistic TimescaleDB Schemas

Your hand-written examples need realistic PostgreSQL/TimescaleDB schemas. Use these as templates:

**IoT Sensor Schema:**
```sql
CREATE TABLE sensors (
    sensor_id  SERIAL PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    location   VARCHAR(50),
    type       VARCHAR(30)
);

SELECT create_hypertable('sensor_readings', 'recorded_at');
CREATE TABLE sensor_readings (
    sensor_id  INTEGER REFERENCES sensors(sensor_id),
    recorded_at TIMESTAMPTZ NOT NULL,
    temperature DOUBLE PRECISION,
    humidity    DOUBLE PRECISION,
    pressure    DOUBLE PRECISION
);
```

**E-Commerce Schema:**
```sql
CREATE TABLE users (
    user_id    SERIAL PRIMARY KEY,
    email      VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE orders (
    order_id   SERIAL PRIMARY KEY,
    user_id    INTEGER REFERENCES users(user_id),
    amount     NUMERIC(10, 2),
    status     VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE order_items (
    item_id    SERIAL PRIMARY KEY,
    order_id   INTEGER REFERENCES orders(order_id),
    product_id INTEGER,
    quantity   INTEGER,
    unit_price NUMERIC(10, 2)
);
```

**Application Events Schema (TimescaleDB):**
```sql
SELECT create_hypertable('events', 'occurred_at');
CREATE TABLE events (
    event_id    UUID DEFAULT gen_random_uuid(),
    user_id     INTEGER,
    event_type  VARCHAR(50),
    occurred_at TIMESTAMPTZ NOT NULL,
    metadata    JSONB
);
```

### Completing the 100 Hand-Written Examples

You wrote 20 in Week 25. You need 80 more. Distribute them:

| Category | Count |
|---|---|
| Basic TimescaleDB (time_bucket, continuous agg) | 15 |
| PostgreSQL-specific (JSONB, ON CONFLICT, RETURNING) | 15 |
| Window functions (LAG, LEAD, RANK, NTILE) | 15 |
| Multi-table JOINs with 3+ tables | 15 |
| Advanced patterns (recursive CTE, LATERAL, correlated subquery) | 10 |
| Performance-focused (EXPLAIN ANALYZE guidance, index hints) | 10 |
| Total new | 80 |

For each example, the workflow is:
1. Write the question and schema
2. Write the SQL by hand
3. Execute against a local Postgres instance (Docker if needed)
4. Fix any runtime errors
5. Verify the result is what the question asked for
6. Convert to ChatML format

**The local Postgres verification step is non-negotiable.** Only examples that actually execute correctly belong in your training data.

### Running and Filtering Self-Instruct Generation at Scale

With 100 seed examples, run Self-Instruct:

```python
# In self_instruct.py from Week 25

import concurrent.futures

def run_self_instruct_pipeline(seeds, target_count=5000, batch_size=20):
    all_instructions = []
    all_examples = []

    # Step 1: Generate instructions
    while len(all_instructions) < target_count:
        batch = generate_instructions(seeds, n=batch_size)
        all_instructions.extend(batch)
        print(f"Generated {len(all_instructions)} instructions so far")

    # Step 2: Generate SQL responses (parallelizable)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(generate_sql_response, inst): inst
                   for inst in all_instructions[:target_count]}
        for future in concurrent.futures.as_completed(futures):
            inst = futures[future]
            try:
                sql = future.result()
                all_examples.append({
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": inst},
                        {"role": "assistant", "content": sql}
                    ]
                })
            except Exception as e:
                print(f"Failed: {e}")

    return all_examples
```

**Cost estimate:**
- Using GPT-3.5-turbo: 5,000 examples × ~600 tokens average = 3M tokens ≈ $4.50
- Using Ollama + Qwen2.5-Coder-7B locally: free (needs 8GB RAM)
- Using Claude Haiku API: ~$2.50

For cost reasons, use Ollama + local model unless you need API-level quality.

### Deduplication Strategy

After generating all examples, deduplicate across all tiers:

```python
from datasketch import MinHash, MinHashLSH

def dedup_examples(examples, threshold=0.7):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    deduped = []

    for i, ex in enumerate(examples):
        question = ex['messages'][1]['content']  # user message
        m = get_minhash(question)
        key = f"ex_{i}"
        try:
            neighbors = lsh.query(m)
            if len(neighbors) == 0:  # no near-duplicates found
                lsh.insert(key, m)
                deduped.append(ex)
        except Exception:
            deduped.append(ex)  # on error, keep the example

    return deduped
```

Run deduplication on the question (user message) only, not the SQL, since different questions can legitimately produce the same SQL pattern.

### Dataset Documentation: The Dataset Card

A dataset card on HuggingFace Hub must include:

```markdown
# postgres-sql-v1

## Dataset Summary
5,000 PostgreSQL/TimescaleDB text-to-SQL pairs in ChatML format for fine-tuning code LLMs.

## Data Sources
- Tier 1: Spider (filtered, PostgreSQL-converted) and BIRD benchmark (~2,000 examples)
- Tier 2: Hand-written PostgreSQL/TimescaleDB examples (~100 examples)
- Tier 3: Self-Instruct synthetic with Qwen2.5-Coder-7B (~2,900 examples)

## Schema
Each example has a `messages` field (list of role/content dicts).

## Statistics
- Total examples: 5,000
- Train split: 4,000 examples
- Val split: 1,000 examples
- Average SQL tokens: [computed from data]
- Coverage: SELECT, INSERT, UPDATE, DELETE, CTEs, Window functions, TimescaleDB functions

## Quality Assurance
- All SQL validated with sqlglot (PostgreSQL dialect)
- Tier 1 filtered for PostgreSQL compatibility (no SQLite-only syntax)
- Tier 2 executed against PostgreSQL 16
- Deduplicated with MinHash (Jaccard threshold 0.7)

## Intended Use
Fine-tuning code LLMs (specifically Qwen2.5-Coder-7B) for PostgreSQL/TimescaleDB SQL generation.
```

---

## Connections

**Week 25:** All infrastructure from Week 25 is deployed this week. Do not skip Week 25.

**Phase 4+:** This dataset is the core artifact of your Phase 3–4 work. You will use it in Week 29 (first SFT), Week 31 (LoRA), and Phase 5–6 (full fine-tuning pipeline).

---

## Common Misconceptions

- **"I need perfect quality before publishing."** Publish v1 even if 10–15% of synthetic examples are imperfect. Document the quality level clearly. v2 improvements come from fine-tuning feedback.
- **"More examples in Tier 3 compensates for fewer in Tier 2."** It does not. 50 well-crafted, executed hand-written examples with realistic TimescaleDB schemas are worth more than 500 synthetic ones for teaching PostgreSQL-specific idioms.
- **"I should publish my dataset publicly."** Only if you are comfortable with it. Private HuggingFace datasets are fine for this course.
- **"I can use sqlglot parse-only validation as a guarantee of SQL correctness."** sqlglot confirms syntax, not semantics. A query can parse correctly and still return wrong results. Only execution against a database guarantees semantic correctness.

---

## Time Allocation (6–8 hrs)

- 0.5h: Set up Docker PostgreSQL for local SQL execution verification
- 2.5h: Complete 80 hand-written examples, verify each against PostgreSQL
- 1h: Run full Tier 1 processing (Spider + BIRD conversion + filter)
- 1.5h: Run Self-Instruct generation (can be background if using local Ollama)
- 0.5h: Merge, deduplicate, split, and produce final JSONL files
- 0.5h: Publish to HuggingFace Hub with dataset card
- 0.5h: Commit and journal entry
