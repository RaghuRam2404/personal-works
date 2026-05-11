# Week 76 Assignment Solutions — Multi-Turn Agentic SQL with Tool Use

## Task 1 — Tool Format Key Snippet

The trickiest part is verifying the template output. For Qwen2.5-Coder:

```python
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

tools = [{
    "type": "function",
    "function": {
        "name": "execute_sql",
        "description": "Execute SQL against PostgreSQL; returns rows or error string.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}]

messages = [
    {"role": "user", "content": "Schema: orders(id, region, amount)\nQuestion: Total amount by region?"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "execute_sql",
                      "arguments": json.dumps({"query": "SELECT region, SUM(amount) FROM orders GROUP BY region;"})}}
    ]},
    {"role": "tool", "tool_call_id": "call_1",
     "content": "region,sum\nNorth,15000\nSouth,12000"},
    {"role": "assistant", "content": "SELECT region, SUM(amount) FROM orders GROUP BY region;"}
]

formatted = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
print(formatted)
```

Expected output contains `<tool_call>` and `<tool_response>` markers. The response template for loss masking is `<|im_start|>assistant\n`.

---

## Task 2 — Trajectory Builder Key Snippet

```python
import psycopg2, json, random

def mutate_sql(sql):
    mutations = [
        lambda s: s.replace("JOIN", "/*JOIN*/", 1),         # drop join
        lambda s: s.replace("region", "regio", 1),           # wrong column
        lambda s: s.replace("SELECT", "SELECT,", 1),         # syntax error
    ]
    return random.choice(mutations)(sql)

def build_trajectory(question, schema, correct_sql, conn):
    wrong_sql = mutate_sql(correct_sql)
    try:
        cur = conn.cursor()
        cur.execute(wrong_sql)
        error_msg = "OK"  # mutation may not always error
    except Exception as e:
        conn.rollback()
        error_msg = str(e)
    
    return {"messages": [
        {"role": "user", "content": f"{schema}\nQuestion: {question}"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"function": {"name": "execute_sql",
                                      "arguments": json.dumps({"query": wrong_sql})}}]},
        {"role": "tool", "content": error_msg},
        {"role": "assistant", "content": None,
         "tool_calls": [{"function": {"name": "execute_sql",
                                      "arguments": json.dumps({"query": correct_sql})}}]},
        {"role": "tool", "content": "rows returned"},
        {"role": "assistant", "content": correct_sql}
    ]}
```

**Expected output:** JSONL file where ~65% of examples have error_msg != "OK" (syntax mutations always error; column mutations may or may not depending on schema).

---

## Task 3 — Loss Masking Key Snippet

```python
def build_labels(input_ids, tokenizer, assistant_start_tokens):
    """Return label array with -100 everywhere except assistant turns."""
    labels = [-100] * len(input_ids)
    ids = list(input_ids)
    i = 0
    while i < len(ids):
        # Detect assistant turn start
        window = ids[i:i+len(assistant_start_tokens)]
        if window == assistant_start_tokens:
            i += len(assistant_start_tokens)  # skip the template token
            # Unmask until next role boundary
            while i < len(ids) and ids[i:i+3] != [ROLE_BOUNDARY_TOKEN]:
                labels[i] = ids[i]
                i += 1
        else:
            i += 1
    return labels
```

**Verify masking:** `unmasked_fraction = (labels != -100).float().mean()`. Should be 0.30–0.50 on agentic data. If it is >0.80, your boundary detection is wrong.

---

## Task 4 — Agentic Loop Key Snippet

```python
import psycopg2, json, re

def execute_sql(conn, sql):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return str(rows[:5])  # cap output length
    except Exception as e:
        conn.rollback()
        return f"ERROR: {e}"

def agentic_sql_loop(model, tokenizer, question, schema, conn, max_rounds=3):
    messages = [{"role": "user", "content": f"{schema}\nQuestion: {question}"}]
    for round_idx in range(max_rounds):
        response = generate_response(model, tokenizer, messages)
        tool_call = extract_tool_call(response)  # returns SQL string or None
        if tool_call is None:
            return response, round_idx + 1  # final answer, rounds used
        result = execute_sql(conn, tool_call)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "tool", "content": result})
    return response, max_rounds
```

---

## Common Gotchas

- **Loss computed on tool turns:** If your label array has non-(-100) values at tool-result positions, your model will try to predict database output. Check with `assert labels[tool_turn_positions].eq(-100).all()`.
- **Conversation too long after 3 rounds:** Full multi-turn conversations can exceed 2048 tokens. Truncate schema to the top-5 most relevant tables if total length exceeds your model's context.
- **psycopg2 rollback required:** After a failed query, the connection is in an aborted state. Always call `conn.rollback()` before executing the next query.
- **extract_tool_call returning None for valid calls:** Tool call extraction depends on your model's exact format. Use the decoded token string (not just the raw logit output) and match the exact delimiter pattern.

---

## How to Verify You Did It Right

1. Open `week76/agentic_eval_results.json`. Confirm `final_em > first_attempt_em` — agentic correction is helping.
2. In W&B, inspect the `unmasked_tokens_per_batch` metric. It should be stable around 30–50% and not vary wildly across batches.
3. Print three full decoded training examples from your agentic dataset. Manually verify the turn order: user → tool_call → tool_result → tool_call → tool_result → final_answer.
4. Run the loop on one TimescaleDB time_bucket query manually. Watch the terminal output for each round. Verify the model correctly uses the error from round 1 to fix round 2.
