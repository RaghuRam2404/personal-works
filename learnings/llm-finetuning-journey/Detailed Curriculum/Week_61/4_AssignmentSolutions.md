# Week 61 Assignment Solutions

## Task 1 — Eval Harness Key Components

```python
import json, threading, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psycopg2

def execute_with_timeout(sql, conn, timeout=30):
    """Execute SQL with timeout; return (rows, error)."""
    result = {"rows": None, "error": None}
    def target():
        try:
            cur = conn.cursor()
            cur.execute("BEGIN")
            cur.execute(sql)
            result["rows"] = cur.fetchall()
            conn.rollback()
        except Exception as e:
            conn.rollback()
            result["error"] = str(e)[:100]
    t = threading.Thread(target=target)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        conn.cancel()
        return None, "timeout"
    return result["rows"], result["error"]

def results_match(pred_rows, ref_rows):
    """Compare result sets by sorting rows as strings."""
    if pred_rows is None or ref_rows is None:
        return False
    return sorted(str(r) for r in pred_rows) == sorted(str(r) for r in ref_rows)

def format_prompt(question, schema_ddl, system_prompt):
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\nSchema:\n{schema_ddl}"},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

## Task 2 — Batch Inference Pattern

```python
@torch.no_grad()
def batch_generate(prompts, model, tokenizer, batch_size=8):
    """Generate SQL for a list of prompts."""
    all_outputs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=2048).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,         # greedy for eval
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode only newly generated tokens (not the input)
        for j, out in enumerate(outputs):
            new_tokens = out[inputs.input_ids.shape[1]:]
            all_outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return all_outputs
```

---

## Common Gotchas

- **BIRD uses its own database files (SQLite).** You can run BIRD eval against SQLite directly for standard SQL queries; for PostgreSQL-specific queries, you must convert. Use SQLite for BIRD evaluation to compare fairly with published SOTA numbers.
- **Result set comparison fails for floating point.** Round floats to 4 decimal places before comparison: `round(float(v), 4)` for each value in rows.
- **Spider database loading fails.** Spider databases are nested in subdirectories; use `os.walk()` to find all `.sqlite` files.
- **Empty result sets.** Some reference SQL returns 0 rows (no matching data). Both predicted and reference return 0 rows → this should be counted as correct if the predicted SQL executes without error.
- **OOM during batch inference.** Reduce batch_size from 8 to 4. If still OOM, use 4-bit loading for eval: `load_in_4bit=True`.

---

## How to Verify You Did It Right

1. `bird_eval_results.json` has 1,534 entries, each with `pred_sql`, `exec_status`, `correct` fields
2. Execution accuracy is reported to 1 decimal place with question count denominator: "65.3% (1,002/1,534)"
3. Error type distribution sums to 100%
4. `spider_eval_summary.md` shows higher accuracy than BIRD (Spider is easier)
5. Custom eval shows breakdown by TimescaleDB vs standard SQL — TimescaleDB should be lower
6. All eval scripts accept `--max-examples N` to run a quick 50-example sanity check before full run
