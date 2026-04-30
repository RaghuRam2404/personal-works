# Week 6 — Quiz

---

**Q1.** GPT-4's tokenizer splits the number `1698765432000` into multiple tokens because of a pattern in its pre-tokenization regex. Why does this design choice exist, and what is the consequence for a model generating SQL `WHERE` clauses with Unix millisecond timestamps?

A) Large numbers are split because they don't appear in the training data, so they get OOV tokens.
B) The regex limits numbers to at most 3 digits per token to prevent the model from treating all long numbers as single semantic units. The consequence is that a 13-digit timestamp becomes 5 separate tokens, each with their own positional embedding, making arithmetic and exact reproduction harder.
C) The regex splits numbers arbitrarily at spaces. This has no consequence for SQL.
D) Splitting numbers increases vocabulary diversity. The consequence is a shorter context for SQL queries.

---

**Q2.** In BPE training, why must merges be applied in the order they were learned, and not all simultaneously?

A) Simultaneous application would exceed memory limits.
B) Applying merge i produces new pairs that may trigger merge i+1. Later merges are defined in terms of tokens created by earlier merges — the vocabulary is built incrementally. Applying all merges simultaneously would skip these compound tokens.
C) The BPE algorithm only considers pairs, not triples, so simultaneous application is impossible.
D) Simultaneous application gives the same result; ordering is just a convention.

---

**Q3.** You add a new special token `<|sql_end|>` to a pre-trained tokenizer and resize the model's embedding table. How should the new token's embedding be initialized, and why?

A) Initialize to zeros — the model will learn the correct representation during fine-tuning.
B) Initialize to the mean of all existing embeddings — this provides a neutral starting point that is close to the distribution of existing embeddings. Random initialization far from this distribution can destabilize early fine-tuning steps where the special token appears frequently.
C) Copy the `<|endoftext|>` embedding exactly — special tokens should all be identical.
D) Initialize to ones — the model needs a strong signal for special tokens.

---

**Q4.** Your SQL BPE tokenizer trained on Spider has a vocabulary size of 1000 (256 bytes + 744 merges). The test SQL contains `TimescaleDB`, which never appeared in Spider training data. How does your tokenizer handle this?

A) It raises an OOV error and fails.
B) It maps it to `<UNK>`.
C) Because BPE starts from individual bytes, `TimescaleDB` is always encodable — it falls back to byte-level encoding for the unknown substrings. The token sequence will be longer (one token per byte for unknown parts) but there is no OOV failure.
D) It uses the closest token in the vocabulary based on edit distance.

---

**Q5.** WordPiece uses `##` to mark non-initial subword pieces (e.g., `playing` → `['play', '##ing']`). GPT-2/4's BPE instead encodes the leading space as part of the token (` playing` → `[' play', 'ing']`). What practical difference does this create for a model trained with each tokenizer?

A) No practical difference — both approaches encode the same information.
B) WordPiece's approach makes it harder to detokenize — you must strip `##` prefixes. GPT-2's approach makes it harder to split tokens at word boundaries. The main practical difference is that GPT-2's tokenizer needs the leading space in prompts — `"SELECT"` and `" SELECT"` may be different tokens.
C) WordPiece cannot handle SQL; GPT-2 tokenization is universal.
D) The `##` prefix is only used for BERT, not for any SQL-capable model.

---

**Q6 (short answer).** You are building a text-to-SQL system using Qwen2.5-Coder-7B. You notice that the model sometimes generates `GROUP  BY` (with two spaces) instead of `GROUP BY` (one space). Explain why tokenization might be the root cause of this error, and suggest a fix.

---

**Q7 (short answer).** Explain the difference between BPE training and BPE inference (encoding new text). Why does the order of the merge table matter at inference time?

---

**Q8 (short answer).** Why do character-level language models (like your Week 4 char-LSTM) have a structural advantage over BPE-based models for generating very specific SQL strings like `"WHERE time >= '2024-01-15 09:23:47'"` but a structural disadvantage for modeling long-range SQL syntax?
