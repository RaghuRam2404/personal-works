# Week 7 — Assignment Solutions

## Task 1 — Key Snippets (distilgpt2 Inspection)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tok   = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.eval()

prompt = "SELECT id FROM"
inputs = tok(prompt, return_tensors="pt")

# Inspect tokens
token_strs = tok.convert_ids_to_tokens(inputs.input_ids[0].tolist())
print(f"Tokens: {token_strs}")  # e.g., ['SELECT', 'Ġid', 'ĠFR', 'OM']
print(f"Count: {len(token_strs)}")

# Forward pass + top-5 predictions
with torch.no_grad():
    out    = model(**inputs)
    logits = out.logits           # (1, seq_len, 50257)
    last   = logits[0, -1, :]    # logits at final position
    probs  = torch.softmax(last, dim=-1)
    top5   = probs.topk(5)
    for prob, idx in zip(top5.values, top5.indices):
        print(f"  {tok.decode([idx.item()])!r}: {prob.item():.4f}")

# Generation
gen = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tok.eos_token_id,
)
print(tok.decode(gen[0], skip_special_tokens=True))
```

**Expected top-5 for "SELECT id FROM":** distilgpt2 is not SQL-trained, so predictions will be English words. Something like `[' the', ' a', ' an', ' this', ' your']`. This motivates why fine-tuning matters.

**Common gotchas:**
- distilgpt2 does not have a `pad_token` by default. Always set `pad_token_id=tok.eos_token_id` in `generate()` or you get a warning.
- `logits[0, -1, :]` is the prediction at the last input token, which is the prediction for the next token. `logits[0, 0, :]` is the prediction after seeing only the first token.
- Token strings for GPT-2 use `Ġ` (G with cedilla) to represent a leading space. `Ġid` means ` id` (with space). This is the GPT-2 tokenizer's way of encoding whitespace.

---

## Task 2 — Key Snippets (Spider Processing)

```python
from datasets import load_dataset
from transformers import AutoTokenizer

ds  = load_dataset("spider")
tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token   # distilgpt2 has no pad token

def format_example(ex):
    ex['text'] = (
        f"### Question: {ex['question']}\n"
        f"### Database: {ex['db_id']}\n"
        f"### SQL: {ex['query']}"
    )
    return ex

def tokenize(examples):
    return tok(
        examples['text'],
        truncation=True,
        max_length=512,
        padding=False,  # don't pad here; DataCollator handles it later
    )

ds = ds.map(format_example)
tokenized = ds.map(tokenize, batched=True, remove_columns=ds['train'].column_names)

# Token length statistics
import numpy as np
lengths = [len(x) for x in tokenized['train']['input_ids']]
print(f"Mean: {np.mean(lengths):.1f}, Median: {np.median(lengths):.1f}, Max: {max(lengths)}")
# Expected: Mean ~80, Median ~65, Max ~400+ for Spider

tokenized.save_to_disk("week_07/spider_tokenized")
tokenized.push_to_hub("YOUR_USERNAME/spider-tokenized-phase1", private=True)
```

**Expected statistics for Spider train split:** ~7,000 examples. Mean token length ~60–100 tokens with distilgpt2 (longer with Qwen due to different vocabulary). Max ~400+ for the longest queries.

**Common gotchas:**
- `load_dataset("spider")` requires accepting the dataset's terms on the Hub. If it errors, visit the dataset page on huggingface.co and click "Accept Terms."
- `tok.pad_token = tok.eos_token` — necessary for distilgpt2. Without it, `tokenize()` with `padding=True` will error (no pad token defined).
- `remove_columns=ds['train'].column_names` removes ALL original columns. Check that `input_ids` and `attention_mask` are preserved in `tokenized`.
- `save_to_disk` saves in Arrow format locally. `push_to_hub` uploads to Parquet format on the Hub. Both are reversible.

---

## Task 4 — Key Snippets (Labels for Causal LM)

```python
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilgpt2")
tok.pad_token = tok.eos_token

texts = [
    "SELECT id FROM users",
    "SELECT id, name FROM departments WHERE dept = 'Engineering'",
    "SELECT",
]
batch = tok(texts, padding=True, truncation=True, return_tensors="pt")

def prepare_causal_lm_batch(input_ids, attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels

labels = prepare_causal_lm_batch(batch.input_ids, batch.attention_mask)
print("input_ids:", batch.input_ids)
print("attention_mask:", batch.attention_mask)
print("labels:", labels)
# Labels should show -100 at all right-padding positions
```

**Expected output:** `labels` has the same shape as `input_ids`. The shortest example ("SELECT") has the most -100 positions (padding). The longest example has no -100 positions.

**Common gotchas:**
- Right-padding vs. left-padding: GPT-family models use right-padding by default. Check `tok.padding_side` — it should be `"right"` for causal LMs.
- The -100 positions must align exactly with padding positions (where `attention_mask == 0`). Misalignment corrupts the loss.

---

## How to Verify You Did It Right

1. **Task 1:** Top-5 tokens are printed with probabilities summing to less than the full probability mass (just top-5). Generation produces English text (distilgpt2 is not SQL-trained).
2. **Task 2:** `tokenized['train']` has columns `input_ids` and `attention_mask` only. Hub URL is accessible when you visit it. Token length stats printed.
3. **Task 3:** Greedy (`do_sample=False`) produces the same output every time. Temperature=0.1 produces nearly identical outputs. Temperature=0.9 produces diverse, less coherent outputs.
4. **Task 4:** Labels tensor shows -100 at all padding positions. Non-padding positions match input_ids exactly.
