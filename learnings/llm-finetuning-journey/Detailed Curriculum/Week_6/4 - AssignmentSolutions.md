# Week 6 — Assignment Solutions

## Task 1 — Key Snippets (BPE from Scratch)

The core BPE training loop:

```python
def get_stats(ids):
    """Count all adjacent pairs."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """Replace all occurrences of pair with new token idx."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def train(text, vocab_size):
    assert vocab_size >= 256
    num_merges = vocab_size - 256
    # Start from byte-level encoding
    tokens = list(text.encode('utf-8'))
    merges = {}
    for i in range(num_merges):
        stats = get_stats(tokens)
        if not stats: break
        top_pair = max(stats, key=stats.get)
        idx = 256 + i
        tokens = merge(tokens, top_pair, idx)
        merges[top_pair] = idx
    return merges
```

**Round-trip verification:**
```python
text = "SELECT * FROM orders WHERE status = 'completed'"
ids = encode(text)
assert decode(ids) == text, "Round trip failed!"
```

**Common gotchas:**
- The `merge` function scans left-to-right and replaces non-overlapping pairs. If you naively use string replace or regex, you may miss edge cases or create overlapping merges.
- Decode must reconstruct bytes, then decode UTF-8: `b''.join(bytes_vocab[id] for id in ids).decode('utf-8')`. Direct character concatenation fails for multi-byte UTF-8 characters.
- The `get_stats` function only counts adjacent pairs in the current token sequence — not in the original text. After each merge step, the token sequence changes.
- GPT-4 pre-tokenization regex must be applied before BPE. Without it, merges will cross word boundaries (e.g., `' SELECT'` would merge the leading space with 'S').

---

## Task 2 — Expected Comparison Output

Example output for `"SELECT AVG(response_time_ms) FROM api_logs WHERE timestamp > 1698765432000"`:

```
GPT-4 (cl100k_base):
  Tokens: ['SELECT', ' AVG', '(', 'response', '_time', '_ms', ')', ' FROM', ' api', '_logs', ' WHERE', ' timestamp', ' >', ' 1698765432000']
  Count: 14 tokens

Your SQL BPE (vocab_size=1000):
  Tokens: ['SE', 'LE', 'CT', ' ', 'AV', 'G(', 'response', '_time', '_ms)', ' FROM', ' api', '_log', 's', ' WH', 'ERE', ...]
  Count: ~18-22 tokens (depends on training corpus)
```

**Key observations to include in analysis:**
- GPT-4 keeps `timestamp` as a single token (it's common in English text). Your SQL tokenizer may split it.
- `1698765432000` (13 digits): GPT-4's pattern limits numbers to 3 digits per token → 5 tokens for this number alone. This is a significant overhead for timestamp-heavy TimescaleDB queries.
- `response_time_ms`: GPT-4 splits at underscores → 3–4 tokens. Your SQL BPE may learn to keep it together if this identifier appears frequently in Spider.

**Common gotchas:**
- `tiktoken` uses `cl100k_base` for GPT-4 (not `gpt2` or `p50k_base`). Use `tiktoken.get_encoding("cl100k_base")`.
- Your BPE's `encode()` must handle non-ASCII gracefully (fall back to byte-level). Test with `"SELECT id FROM café"`.

---

## Task 3 — Expected HuggingFace Output

```python
# Qwen2.5-Coder-7B tokenizer (approximate values)
tok.vocab_size          # 151643
tok.model_max_length    # 32768
tok.special_tokens_map  # {'eos_token': '<|endoftext|>', 'pad_token': '<|endoftext|>', ...}

tok.encode("SELECT")    # e.g., [21382]  (single token — good for SQL)
tok.encode("FROM")      # e.g., [4198]   (single token)
tok.encode("WHERE")     # e.g., [13358]  (single token)
tok.encode("GROUP BY")  # may be [13389, 16531] (two tokens)
```

**What `add_special_tokens=True` adds:** Typically `<|im_start|>user\n` and `<|im_end|>` for chat-format models. For base models, it usually adds `<|endoftext|>` (BOS). The difference matters for fine-tuning data formatting — always check what special tokens your model adds by default.

**Common gotchas:**
- `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")` requires HuggingFace login for gated models. Use `huggingface-cli login` first. If the model is not yet accessible, use `distilgpt2` as a fallback for exploring the API.
- `tok.decode([id])` for a single token may include a leading space — that is the SentencePiece `▁` prefix. This is expected behavior.

---

## How to Verify You Did It Right

1. **Task 1:** `decode(encode(text)) == text` for at least 10 different SQL queries, including ones with special characters, numbers, and identifiers. Zero assertion failures.
2. **Task 2:** `tokenizer_comparison.txt` exists and shows clear numerical differences between SQL BPE and GPT-4 token counts on the 5 test queries.
3. **Task 3:** Output shows Qwen's vocabulary size (~150K), confirms SQL keywords tokenize to single tokens, and demonstrates the special token difference with `add_special_tokens`.
