# Week 39 Assignment Solutions

## Task 1 — Core Harness: Key Snippets

The trickiest part is creating an isolated context per test example and safely executing SQL with a timeout.

```python
import psycopg2, sqlparse, json

def get_connection():
    return psycopg2.connect(
        host="localhost", port=5432,
        user="postgres", password="postgres",
        database="postgres", connect_timeout=5
    )

def is_safe_sql(sql: str) -> bool:
    parsed = sqlparse.parse(sql.strip())
    if not parsed:
        return False
    return parsed[0].get_type() == "SELECT"

def execute_example(schema_sql: str, expected_sql: str, generated_sql: str):
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    # Isolate each example in its own schema
    schema_name = "eval_tmp"
    cur.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    cur.execute(f"CREATE SCHEMA {schema_name}")
    cur.execute(f"SET search_path TO {schema_name}")

    # Create tables
    cur.execute(schema_sql)

    # Execute expected SQL
    cur.execute("SET statement_timeout = 5000")
    cur.execute(expected_sql)
    expected_rows = cur.fetchall()

    # Execute generated SQL safely
    if not is_safe_sql(generated_sql):
        conn.close()
        return None, None, "Non-SELECT query rejected"

    try:
        cur.execute("SET statement_timeout = 5000")
        cur.execute(generated_sql)
        actual_rows = cur.fetchall()
        error = None
    except Exception as e:
        actual_rows = None
        error = str(e)

    conn.close()
    return expected_rows, actual_rows, error
```

Result comparison — normalize before sorting:

```python
def compare_results(expected, actual):
    if expected is None or actual is None:
        return False
    def norm(row):
        return tuple("NULL" if v is None else str(v) for v in row)
    return sorted(norm(r) for r in expected) == sorted(norm(r) for r in actual)
```

**Expected output (summary print):**
```
Model: Week38-v1  Total: 100  Exec Success: 94/100 (94.0%)  Exec Correct: 71/100 (71.0%)
```

---

## Task 2 — Inference: Key Snippet

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch, json

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")

# For Week 38 adapter:
# model = PeftModel.from_pretrained(base, "<your-handle>/postgres-sqlcoder-7b-v1")

def generate_sql(model, tokenizer, schema, question, max_new_tokens=256):
    prompt = (
        f"### Schema:\n{schema}\n\n"
        f"### Question:\n{question}\n\n"
        f"### SQL:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # Strip markdown fences if present
    decoded = decoded.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
    return decoded
```

**Common gotchas:**

- The model may output ```` ```sql ... ``` ```` fences — always strip them before passing to the harness
- `do_sample=False` with `temperature=1.0` is correct; setting `temperature=0` raises an error in some HF versions — use `do_sample=False` alone
- Greedy decoding can stall on long outputs; set `max_new_tokens=256` hard limit
- Each model inference pass should be done in sequence; do not load two 7B models simultaneously on a T4

---

## Task 3 — Comparison Table: Expected Ranges

| Model | Exec Success % | Exec Correct % | Exact Match % |
|---|---|---|---|
| Base Qwen2.5-Coder-7B | 70–85% | 20–35% | 5–15% |
| Week 33 QLoRA | 85–93% | 45–60% | 30–45% |
| Week 38 QLoRA (v1) | 90–97% | 60–80% | 40–60% |

If your Week 38 model exec correctness is below 55%, check: did the held-out test set accidentally overlap with training data? Run a deduplication check.

---

## Task 4 — Error Analysis

Typical distribution for a first-pass 7B model on SQL eval:

| Category | Approx % of failures |
|---|---|
| Wrong filter (WHERE clause) | 35–45% |
| Wrong aggregation | 15–25% |
| Wrong join | 15–20% |
| Syntax error | 10–15% |
| Wrong table/column | 5–10% |
| Other | 5% |

The most actionable finding: if "Wrong filter" dominates, your training data may lack variety in complex WHERE conditions. That informs Phase 5 data augmentation.

---

## How to Verify You Did It Right

1. Run `eval_harness.py` on a single trivial example (`SELECT 1 AS x`) — it should return `exec_success=True, exec_correct=True`
2. Introduce a deliberate error in `generated_sql` (e.g., `SELEC 1`) — confirm `exec_success=False, error_msg` is populated
3. Confirm `eval_results_week38.jsonl` has exactly 100 lines
4. Open 5 random `exec_correct=False` examples and manually inspect — do the expected vs. actual result sets visibly differ?
5. Confirm your exec correctness for the Week 38 model is higher than for the base model by at least 25 percentage points
