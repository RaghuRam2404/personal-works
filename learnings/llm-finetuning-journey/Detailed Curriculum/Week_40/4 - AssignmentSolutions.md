# Week 40 Assignment Solutions

## Task 2 — Publishing to HuggingFace Hub: Key Snippets

Pushing the adapter from Colab:

```python
from huggingface_hub import login
login()  # prompts for your HF token

# If you have the PeftModel in memory:
model.push_to_hub("your-handle/postgres-sqlcoder-7b-v1")
tokenizer.push_to_hub("your-handle/postgres-sqlcoder-7b-v1")

# If you only have a local checkpoint directory:
from huggingface_hub import upload_folder
upload_folder(
    folder_path="/content/checkpoints/week38-final",
    repo_id="your-handle/postgres-sqlcoder-7b-v1",
    repo_type="model",
)
```

Minimal model card that satisfies the requirements (write this as `README.md` on the Hub or via the web UI):

```markdown
---
base_model: Qwen/Qwen2.5-Coder-7B
license: apache-2.0
tags:
  - text-to-sql
  - postgresql
  - peft
  - qlora
---

# postgres-sqlcoder-7b-v1

QLoRA fine-tune of Qwen2.5-Coder-7B for PostgreSQL text-to-SQL.

## Training

- Base: Qwen/Qwen2.5-Coder-7B
- LoRA: rank=16, alpha=32, target_modules=[q,k,v,o,gate,up,down]
- Quantization: NF4 double-quant, BF16 compute
- Optimizer: paged_adamw_8bit, LR=2e-4
- Data: 14,500 train / 500 val PostgreSQL text-to-SQL examples

## Prompt Format

### Schema:
<CREATE TABLE statements>

### Question:
<natural language question>

### SQL:

## Evaluation (100-example held-out test)

| Metric | Score |
|---|---|
| Exec success | 94% |
| Exec correctness | 71% |
| Exact match | 52% |

## Limitations

Struggles with multi-level nested subqueries and window functions.
```

---

## Task 3 — Pushing the Dataset: Key Snippet

```python
from datasets import Dataset, DatasetDict
import json

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

train_data = load_jsonl("train_15k.jsonl")
val_data   = load_jsonl("val_500.jsonl")

ds = DatasetDict({
    "train":      Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
})
ds.push_to_hub("your-handle/postgres-text2sql-15k")
```

Verify it loaded correctly:
```python
from datasets import load_dataset
ds = load_dataset("your-handle/postgres-text2sql-15k")
assert len(ds["train"]) == 14500
assert len(ds["validation"]) == 500
```

**Common gotchas:**

- If `push_to_hub` fails with a 403 error, your HF token needs "write" permission — regenerate at huggingface.co/settings/tokens
- If column names differ between train and val (e.g., one uses `sql`, the other uses `query`), `DatasetDict` will error — normalize column names first
- Do NOT push `held_out_test.json` to HuggingFace — keep it local; it is your private benchmark for Phase 5 comparisons

---

## Task 4 — Retrospective: What Good Looks Like

A strong retrospective entry for "most important things learned":

Weak: "I learned about QLoRA and how it saves memory."

Strong: "I learned that the quantized base model in QLoRA stores weights in NF4 but dequantizes to BF16 before the forward pass — this means the base contributes BF16-precision activations to the LoRA computation even though it never updates its weights. The memory saving comes entirely from storing 4-bit base weights at rest, not from reducing activation precision."

The difference is specificity. Phase 5 demands this level of understanding when debugging GRPO training instabilities.

---

## How to Verify You Did It Right

1. Open a completely fresh Python environment (new Colab session) and load your model from the Hub — if it requires any local files not on the Hub, your push was incomplete
2. Confirm `ds = load_dataset("your-handle/postgres-text2sql-15k")["train"]` returns 14,500 rows without downloading anything local
3. Run `eval_harness.py` one more time on `held_out_test.json` with the Hub-loaded adapter — the exec correctness number should match your Week 39 report (within 1–2% due to Postgres version differences)
4. Read your own model card as if you were a stranger: does it tell you exactly what prompt to use, what to expect, and what the model cannot do? If not, revise it.
