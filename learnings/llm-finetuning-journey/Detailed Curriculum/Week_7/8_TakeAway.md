# Week 7 — TakeAway

**This week in 15 words:** HuggingFace wraps PyTorch; know its API but never forget what's underneath.

---

## Core Import Patterns

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import login

# Load model and tokenizer
tok   = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Load dataset
ds = load_dataset("spider")   # or local: load_from_disk("path/")

# Push to Hub
ds.push_to_hub("user/dataset-name", private=True)
model.push_to_hub("user/model-name")
tok.push_to_hub("user/model-name")   # same repo as model
```

---

## Dataset Processing Pattern

```python
# 1. Format
def format_example(ex):
    ex['text'] = f"### Question: {ex['question']}\n### SQL: {ex['query']}"
    return ex
ds = ds.map(format_example)

# 2. Tokenize (batched=True for speed)
def tokenize(batch):
    return tok(batch['text'], truncation=True, max_length=512)
ds = ds.map(tokenize, batched=True, remove_columns=['text'])

# 3. Save/load
ds.save_to_disk("local_path/")
ds = load_from_disk("local_path/")
```

---

## Labels for Causal LM Fine-tuning

```python
# Mask padding positions so loss ignores them
labels = input_ids.clone()
labels[attention_mask == 0] = -100   # -100 is CrossEntropyLoss ignore_index
# For instruction tuning (Phase 4): also mask prompt tokens to -100
```

---

## Generation API

```python
model.eval()
with torch.no_grad():
    gen = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,        # lower = more deterministic
        top_p=0.9,              # nucleus sampling
        # top_k=40,             # alternative to top_p
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
```

---

## Decision Rules

- **distilgpt2 vs Qwen2.5-Coder:** Use distilgpt2 for API exploration (fast, no gating). Use Qwen for actual SQL tasks.
- **`add_special_tokens`:** True for complete single-field inputs. False for segments in a multi-field prompt.
- **Temperature:** 0.1–0.3 for deterministic SQL. 0.7–0.9 for diverse sampling. Never > 1.2 in production.
- **top_p vs top_k:** Prefer top_p (0.9–0.95) for SQL — it adapts nucleus size to model confidence.
- **Labels and -100:** Always mask padding. Mask prompt tokens for instruction tuning.

---

## Numbers to Remember

- distilgpt2: 82M params, vocab_size=50257
- Qwen2.5-Coder-7B: 7B params, vocab_size≈150K, max_length=32768
- Spider train: ~7,000 examples; mean token length ~80 tokens with distilgpt2
- HuggingFace cache location: `~/.cache/huggingface/hub/`
- `-100`: CrossEntropyLoss ignore_index — always use this for padding in labels

---

## Red Flags

- `NaN` loss at step 1 → labels contain padding token IDs (not -100); fix `prepare_labels`.
- Generation always ends immediately → EOS token is in the first position; check `labels` don't include EOS in wrong place.
- `CUDA out of memory` on first generation call → model loaded in float32 on GPU; use `model.half()` or `torch_dtype=torch.float16` in `from_pretrained`.
- Dataset `.map()` runs every time → missing `cache_file_name` or `.map()` function changed. HuggingFace caches by function hash.
