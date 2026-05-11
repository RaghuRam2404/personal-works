# Week 6 — Answers

---

**Q1. Answer: B**

GPT-4's regex pattern `|\p{N}{1,3}|` matches at most 3 consecutive digits. This design prevents numbers of arbitrary length from being treated as single tokens, which would be impractical for a general vocabulary. The consequence for TimescaleDB SQL is significant: `1698765432000` (13 digits) becomes 5 tokens: `['169', '876', '543', '200', '0']`. Each token has an independent positional embedding and no inherent numeric relationship to adjacent tokens. The model must learn to handle exact numeric reproduction across multiple tokens — a known weakness of LLMs for exact arithmetic and timestamp manipulation.

---

**Q2. Answer: B**

Consider merging `('S', 'E') → 'SE'` (merge 1) and `('SE', 'L') → 'SEL'` (merge 2). After applying merge 1, the pair `('SE', 'L')` now exists in the corpus. If you apply both merges simultaneously (before any are applied), `('SE', 'L')` does not exist yet because `SE` hasn't been created. You would miss merge 2. The sequential nature of BPE means the vocabulary is built bottom-up: each merge creates new tokens that can participate in future merges. This is why the merge table is an ordered list, not a set.

---

**Q3. Answer: B**

Initializing to the mean of all existing embeddings places the new token near the centroid of the embedding space. This is the standard practice in HuggingFace `resize_token_embeddings()` (the default behavior). The alternative — random initialization with a large scale — puts the new token far from all existing embeddings, causing large activations early in fine-tuning that can destabilize the entire network. Zero initialization is also problematic: the token would produce zero contribution to the residual stream in the first layer, and gradients through it would be zero (or near-zero) for the first several steps.

---

**Q4. Answer: C**

BPE's byte-level foundation is its most important property for vocabulary coverage. Since every possible byte value (0–255) is in the base vocabulary, any UTF-8 text — including completely novel identifiers like `TimescaleDB` — can always be encoded. The bytes of `TimescaleDB` are `T=84, i=105, m=109, e=101, ...`. If none of these byte sequences match learned merges, each byte becomes its own token. The token sequence is longer, but there is no failure. This is why GPT-4 has no OOV tokens — the byte-level fallback guarantees full coverage.

---

**Q5. Answer: B**

The practical difference is subtle but matters for prompting. In GPT-2 tokenization: `" SELECT"` (with leading space) is often a single token, while `"SELECT"` (without leading space, at the start of a line) may be a different token or tokenized differently. When you write a prompt like `"Write SQL: SELECT id FROM"`, the tokenizer sees `SELECT` without a leading space (after the colon). At inference, if the model predicts ` SELECT` (with space), it may produce a leading space in the output. This is the root cause of many subtle formatting bugs in LLM outputs. Always log the decoded token strings (not just IDs) for your first few batches to verify the format is what you expect.

---

**Q6 (short answer — model answer):**

The likely root cause: `GROUP BY` may tokenize as `['GROUP', ' BY']` — two tokens with a single space. But if the model's attention pattern slightly misaligns, it may learn to associate `GROUP` with a space-containing token and then independently generate another space before `BY`. The surface-level cause is that whitespace handling in the tokenizer is irregular: sometimes the space is "attached" to the following keyword token, sometimes it is separate. The fix: normalize all SQL whitespace in your training data to a single consistent format before tokenization (collapse all internal whitespace to single spaces, strip leading/trailing whitespace from each SQL clause). Also verify that your evaluation decodes the raw token sequence properly and does not double-decode spaces.

---

**Q7 (short answer — model answer):**

BPE training: process a large corpus, count adjacent token pairs iteratively, and greedily merge the most frequent pair at each step. This produces an ordered list of merge rules (e.g., `('S','E')→256`, `('256','L')→257`, etc.) and a final vocabulary.

BPE inference (encoding new text): start with the byte-level representation of the text. Apply the merge rules one by one, in the exact order they were learned during training. For each rule in order, scan the current token sequence and replace all occurrences of that pair. The order matters because merges are defined in terms of tokens created by prior merges. If you apply rule 5 before rule 1, rule 5's left-hand side token may not exist yet (it was created by rule 1). The final encoding is entirely determined by the learned merge table and its order.

---

**Q8 (short answer — model answer):**

Character-level advantage: the model operates at the finest granularity. Every character in `'2024-01-15 09:23:47'` is an independent token. The model can generate this string exactly, character by character, without any tokenization artifacts. There is no risk of a boundary falling inside a date string. For SQL, this means exact numeric values, column names with underscores, and operator sequences (`>=`, `!=`) are all handled cleanly.

Character-level disadvantage: a SQL statement like `SELECT customer_id, SUM(amount) FROM orders WHERE status = 'completed' GROUP BY customer_id` has ~80 characters. At the character level, the model must attend across 80 positions to understand the full clause structure. Transformer attention is O(T²), so 80 characters costs 6,400 attention operations. A BPE tokenizer with a 32K vocabulary might encode the same statement in ~20 tokens — 400 attention operations. More importantly, with character-level tokens, the positional distance between `SELECT` (position 0–5) and its corresponding `GROUP BY` (position 75–82) is 75 steps. The model has less "budget" (in terms of attention heads and effective range) to maintain this long-range SQL syntactic dependency.
