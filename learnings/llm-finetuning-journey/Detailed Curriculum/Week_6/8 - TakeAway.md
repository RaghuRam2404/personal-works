# Week 6 — TakeAway

**This week in 15 words:** BPE merges byte pairs iteratively; tokenization choices affect SQL generation quality in production.

---

## BPE Core Algorithm

```python
def get_stats(ids):
    """Count adjacent pairs."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, new_id):
    """Replace all (non-overlapping) occurrences of pair."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids)-1 and (ids[i], ids[i+1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

# Training:
tokens = list(text.encode('utf-8'))
merges = {}
for i in range(n_merges):
    stats = get_stats(tokens)
    top   = max(stats, key=stats.get)
    merges[top] = 256 + i
    tokens = merge(tokens, top, 256 + i)
```

---

## HuggingFace Tokenizer API

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")

ids    = tok.encode("SELECT id FROM users", add_special_tokens=False)
tokens = tok.convert_ids_to_tokens(ids)
text   = tok.decode(ids)

# Batch tokenization with padding
batch = tok(["SELECT id FROM t1", "SELECT name FROM t2"],
            padding=True, truncation=True, max_length=128,
            return_tensors="pt")
# batch.input_ids: (2, 128), batch.attention_mask: (2, 128)
```

---

## Key Facts

```
GPT-4 tokenizer:  cl100k_base, vocab_size=100277
GPT-2 tokenizer:  gpt2,        vocab_size=50257
Qwen2.5-Coder:    tiktoken BPE, vocab_size≈150K
LLaMA-3:          tiktoken BPE, vocab_size=128K

SQL keyword tokenization (GPT-4 cl100k_base):
  SELECT  -> single token
  FROM    -> single token
  WHERE   -> single token
  GROUP BY -> two tokens
  1698765432000 -> 5 tokens (3 digits each)
```

---

## Decision Rules

- **Adding special tokens:** Call `tok.add_special_tokens()`, then `model.resize_token_embeddings(len(tok))`. Initialize new embeddings to mean of existing.
- **Numbers in SQL:** Long numbers → many tokens with BPE. Consider normalizing or encoding dates separately.
- **add_special_tokens=True vs False:** Use `False` for middle segments of a multi-part prompt; use `True` only at the boundary positions (BOS at start, EOS at end of target).
- **Round-trip test:** Always verify `tok.decode(tok.encode(text)) == text` for your training data.
- **`attention_mask`:** 1 for real tokens, 0 for padding. Always pass it to the model — forgetting it causes padding to affect attention.

---

## Red Flags

- `decode(encode(text)) != text` → bug in your BPE decode; check byte reconstruction.
- SQL output has double spaces → tokenizer whitespace handling mismatch; normalize training data.
- Very high token count per SQL query → check if long numbers or identifiers are being fragmented; may need SQL-specific tokenizer.
- `vocab_size` mismatch after adding tokens → forgot to call `model.resize_token_embeddings()`.
