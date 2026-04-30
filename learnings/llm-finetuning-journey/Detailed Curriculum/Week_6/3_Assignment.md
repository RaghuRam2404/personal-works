# Week 6 — Assignment

## Setup Checklist

- [ ] Install `tiktoken`: `pip install tiktoken`.
- [ ] Install HuggingFace `tokenizers`: `pip install tokenizers`.
- [ ] Spider SQL corpus available: `week_04/sql_queries.txt` (raw SQL query strings). If not, re-extract from Spider.
- [ ] Karpathy minbpe repo cloned for reference (do NOT copy-paste): `git clone https://github.com/karpathy/minbpe`.
- [ ] W&B not needed this week.

---

## Task 1 — Implement BPE from Scratch

**Goal:** Build a working BPE tokenizer by coding along with Karpathy's video.

**Requirements:**
- Watch [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (2h13m) fully. Type every line yourself.
- Save your implementation in `week_06/bpe.py`.
- Your `BPETokenizer` class must implement:
  - `train(text, vocab_size)` — learn merges from a corpus.
  - `encode(text)` → list of int token IDs.
  - `decode(ids)` → str.
  - `save(prefix)` — save merges and vocabulary to `{prefix}.model` and `{prefix}.vocab` files.
  - `load(prefix)` — load saved tokenizer.
- Implement the GPT-4 regex pre-tokenization pattern (Karpathy shows this in the video).
- Verify round-trip: `decode(encode(text)) == text` for arbitrary text.

**Deliverable:** `week_06/bpe.py`.

---

## Task 2 — Train on SQL and Compare to GPT-4

**Goal:** See concretely how SQL-specific tokenization differs from a general-purpose tokenizer.

**Requirements:**
- Create `week_06/train_sql_tokenizer.py`:
  - Load `week_04/sql_queries.txt` as training corpus.
  - Train your BPE tokenizer with `vocab_size=1000` (256 base + 744 merges).
  - Save the tokenizer as `week_06/sql_bpe` (produces `sql_bpe.model` and `sql_bpe.vocab`).
- Create `week_06/compare_tokenizers.py`:
  - Take these 5 SQL queries as test inputs:
    ```python
    TEST_QUERIES = [
        "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
        "SELECT t1.id, t2.name FROM employees AS t1 JOIN departments AS t2 ON t1.dept_id = t2.id",
        "SELECT AVG(response_time_ms) FROM api_logs WHERE timestamp > 1698765432000",
        "SELECT DISTINCT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 5",
        "SELECT * FROM time_series WHERE time >= NOW() - INTERVAL '7 days' ORDER BY time DESC",
    ]
    ```
  - For each query, tokenize with: (a) your SQL BPE tokenizer, (b) GPT-4 (`tiktoken.get_encoding("cl100k_base")`).
  - Print: query, token count (SQL BPE), token count (GPT-4), and the actual token strings for each.
  - Save the comparison output to `week_06/tokenizer_comparison.txt`.

- Write `week_06/tokenizer_analysis.md` with 400–600 words addressing:
  1. Which queries are tokenized more efficiently by your SQL BPE vs GPT-4? Why?
  2. How does GPT-4 tokenize PostgreSQL-specific identifiers like `response_time_ms` or `time_series`?
  3. What would happen to a model fine-tuned with GPT-4 tokenization that sees `1698765432000` (a Unix millisecond timestamp) in a WHERE clause?
  4. If you were to build a production text-to-SQL system, what tokenizer vocabulary size would you choose and why?

**Deliverable:** `week_06/train_sql_tokenizer.py`, `week_06/compare_tokenizers.py`, `week_06/tokenizer_comparison.txt`, `week_06/tokenizer_analysis.md`. Commit message: `week-06-bpe`.

---

## Task 3 — HuggingFace Tokenizer Inspection

**Goal:** Become fluent with the HuggingFace `tokenizers` and `transformers` tokenizer API.

**Requirements:**
- Create `week_06/hf_tokenizer_inspection.py`:
  1. Load the `Qwen/Qwen2.5-Coder-7B` tokenizer: `from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B")`.
  2. Print: vocab size, model max length, special tokens (`tok.special_tokens_map`), and the token IDs for `SELECT`, `FROM`, `WHERE`, `GROUP BY`.
  3. Tokenize the first 5 TEST_QUERIES above and print token count and token strings.
  4. Show what happens when you encode a query with `add_special_tokens=True` vs `add_special_tokens=False`.
  5. Demonstrate `tok.decode([id])` for each of the SQL keyword token IDs you found.

**Deliverable:** `week_06/hf_tokenizer_inspection.py` with output printed to console. Copy the output into `journal.md`.

---

## Stretch Goals

- Add special SQL tokens (`<|sql_start|>`, `<|sql_end|>`, `<|schema|>`) to your SQL BPE tokenizer. Verify that `encode("<|sql_start|> SELECT id FROM users <|sql_end|>")` returns them as single tokens.
- Implement the Unigram tokenization algorithm (an alternative to BPE used in SentencePiece) and compare its vocabulary to your BPE vocabulary on the SQL corpus.
- Profile how long it takes to tokenize 10,000 SQL queries with your Python BPE vs tiktoken (Rust-based). Measure the speedup. This motivates why production systems use compiled tokenizers.
