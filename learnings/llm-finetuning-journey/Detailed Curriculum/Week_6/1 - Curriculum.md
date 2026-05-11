# Week 6 — Tokenization Deep Dive

## Learning Objectives

By the end of this week, you will be able to:

- Explain why character-level and word-level tokenization are suboptimal for LLMs, and why subword tokenization is the current standard.
- Implement byte-pair encoding (BPE) from scratch, including the merge loop and vocabulary construction.
- Reproduce GPT-4's tokenization regex patterns from Karpathy's `minbpe` and explain each component.
- Train a BPE tokenizer on a SQL corpus and compare its vocabulary to GPT-4's tokenization of the same SQL.
- Explain four concrete ways that tokenization choices affect model behavior for code and SQL generation.
- Use HuggingFace `tokenizers` to load, inspect, and extend a pre-trained tokenizer.

---

## Concepts

### Why Tokenization Exists

Neural networks operate on fixed-size vectors. Text must be mapped to integers (token IDs) that index into an embedding table. The question is: what unit should a "token" represent?

**Character-level:** Every character is a token. Vocabulary is tiny (~100–300 for Unicode subsets). But sequences are very long — a 500-character SQL query becomes 500 tokens. Long sequences are expensive for transformers (attention is O(T²)). And the model must learn to spell words from scratch.

**Word-level:** Every word is a token. Vocabulary must be huge (~50K–500K for full English + SQL), and any word not in the vocabulary (OOV) is mapped to `<UNK>`. SQL column names, function names, and identifiers are especially problematic.

**Subword tokenization:** Split words into common subword units. "PostgreSQL" might become `['Post', 'gre', 'SQL']` or `['PostgreSQL']` depending on training data. Vocabulary is medium (32K–128K). No OOV problem. Works well for code, numbers, and multilingual text.

### Byte-Pair Encoding (BPE)

BPE (Sennrich et al., 2016) starts from a character-level vocabulary and iteratively merges the most frequent pair of adjacent tokens into a new token:

```
Algorithm:
1. Start: vocabulary = all individual bytes/characters
2. Count all adjacent pairs in the corpus
3. Find the most frequent pair (e.g., ('S', 'E') → 'SE')
4. Add 'SE' to vocabulary; replace all occurrences of 'S', 'E' with 'SE'
5. Repeat from step 2 until vocabulary reaches target size
```

The resulting vocabulary contains common subwords as single tokens. The number of merge operations determines final vocabulary size: 1000 merges = ~1256 token vocab (256 base bytes + 1000 merges).

**GPT-2/4 specifics:**
- Start from raw bytes (256 initial tokens) — handles any Unicode without OOV.
- Before BPE, apply a regex pattern to split the raw text into chunks that BPE does not merge across:
  ```python
  GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
  ```
  This pattern prevents merging across word boundaries, ensures numbers are at most 3 digits per token, and handles whitespace and punctuation appropriately.
- Vocabulary size: GPT-2 = 50,257; GPT-4 = 100,277.

### SQL Tokenization Quirks

This is the most directly relevant section for your final capstone. Understanding these quirks will affect how you design your fine-tuning data format in Phases 4–5.

**1. Number fragmentation:** GPT-4's tokenizer splits long numbers. `123456789` might become `['123', '456', '789']`. SQL queries with large timestamps, UNIX epochs, or long IDs (common in TimescaleDB) create very long token sequences. A query over a time range `WHERE ts > 1698765432` becomes inefficient.

**2. Identifier tokenization:** PostgreSQL column names like `avg_response_time_ms` might tokenize as `['av', 'g_', 'response', '_time', '_ms']` — fragmented in ways that the model must learn to reassemble. A SQL-specific tokenizer trained on SQL data would likely keep `avg_response_time_ms` as fewer tokens.

**3. Keyword casing:** GPT-4's tokenizer is case-sensitive. `SELECT` and `select` are different tokens. Your training data should normalize SQL keywords to a consistent case (uppercase is conventional).

**4. Whitespace:** SQL has meaningful whitespace (keyword separation). GPT-4's regex pattern handles leading spaces as part of tokens: ` SELECT` (with space) is a different token from `SELECT`. This matters for prompting format.

**5. Special tokens:** `[CLS]`, `[SEP]`, `<|endoftext|>`, `<|im_start|>` are special tokens added to the vocabulary. When fine-tuning, you add task-specific special tokens (e.g., `<|sql|>`, `<|end_sql|>`). Understanding how the base tokenizer handles these is essential before adding them.

### BPE Training vs. Inference

**Training a tokenizer:** Apply the BPE merge algorithm to a corpus to learn the merge table. This produces a vocabulary and an ordered list of merges.

**Tokenizing new text (inference):** Apply the learned merges in order to the input text. Start with byte-level tokens, then apply each merge rule in the order it was learned.

The key insight: the merge order matters. If `('h', 'e')` is the first merge and `('he', 'l')` is the second, then "hello" becomes `['he', 'l', 'l', 'o']` not `['h', 'e', 'l', 'l', 'o']`. The merges are applied sequentially, not in parallel.

### WordPiece and SentencePiece

For completeness:

**WordPiece** (BERT): Uses likelihood rather than frequency as the merge criterion — merge the pair that most increases the language model likelihood. Produces slightly different vocabularies than BPE. The `##` prefix marks non-initial subwords: "playing" → `['play', '##ing']`.

**SentencePiece** (T5, LLaMA): BPE or Unigram applied to raw Unicode characters (including spaces). Space is treated as a regular character (represented as `▁`): "hello world" → `['▁hello', '▁world']`. Handles multiple languages without language-specific preprocessing.

For SQL generation: both GPT-4 (tiktoken BPE) and LLaMA/Qwen (SentencePiece) are relevant. Qwen2.5-Coder uses a tiktoken-based BPE with a 150K vocabulary specifically trained to include code efficiently.

---

## Connections

**Builds on:** Week 4's SQL corpus. Week 2's understanding of vocabulary and embedding tables.

**Unlocks:** Week 7's HuggingFace tokenizer loading. Week 8's capstone uses tokenization. Phase 4's fine-tuning requires understanding exactly how the model's tokenizer handles your SQL training data — poorly tokenized data leads to poor fine-tuning. This week is disproportionately important relative to its position.

---

## Common Misconceptions and Pitfalls

- **"The tokenizer is just preprocessing — it doesn't affect model quality."** Wrong. Tokenization is part of the model's inductive bias. A model fine-tuned with a SQL-specialized tokenizer will need fewer tokens per query, reducing sequence length and memory requirements.
- **"Special tokens can be added freely."** Adding special tokens changes the embedding table. You must ensure the new embeddings are properly initialized (usually from random or from similar existing tokens).
- **"BPE merges characters."** In GPT-2/4, BPE merges bytes, not Unicode characters. This is the key difference from earlier implementations.
- **"Tiktoken is the same as HuggingFace tokenizers."** Different implementations with the same algorithm (BPE). Tiktoken is OpenAI's fast Rust-based implementation. HuggingFace's `tokenizers` library is also fast (Rust backend) and more flexible for custom tokenizers.

---

## Time Allocation (6–8 hours this week)

| Activity | Time |
|---|---|
| Watch Karpathy's GPT Tokenizer video — code along (2h13m) | 2.5 h |
| Read HuggingFace Tokenizers tutorial Chapters 1–3 of the LLM course section on tokenization | 1 h |
| Implement BPE from scratch (training loop) | 1.5 h |
| Train BPE on SQL corpus; compare to GPT-4 | 1 h |
| Journal + commit | 30 min |
