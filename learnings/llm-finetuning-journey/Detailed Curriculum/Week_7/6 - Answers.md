# Week 7 — Answers

---

**Q1. Answer: B**

`logits[0, -1, :]` is the output of the model's language model head applied to the hidden state of the last token position in example 0. It has shape `(vocab_size,)`. When you apply softmax to it, you get `P(next_token | all input tokens)` — the probability distribution over every vocabulary token as the next token. This is exactly the quantity that text generation samples from. The "last position" is key: in a causal (decoder-only) LM, position `t` predicts position `t+1`, so the last input position predicts the first new token.

---

**Q2. Answer: B**

The `attention_mask` controls which tokens a position can attend to — it prevents padding tokens from contributing to other positions' hidden states via attention. However, it does not control the loss computation. If `labels[i, t]` is the padding token ID (not -100), `nn.CrossEntropyLoss` will compute a loss for predicting that padding token. The model receives a gradient signal to predict `<|endoftext|>` after legitimate SQL tokens, teaching it that SQL can end randomly — which causes premature EOS generation at inference time. Always mask padding positions in labels to -100.

---

**Q3. Answer: B**

Greedy decoding makes locally optimal decisions at each step: argmax over the vocabulary. But the optimal continuation from position t depends on all future tokens, not just the immediate next token. A sequence that looks locally best at step 5 may be globally suboptimal by step 10. Beam search maintains the top-k partial sequences simultaneously and scores them based on their complete log-probability — it is more expensive (O(k×T) forward passes) but finds better global sequences. For SQL, beam search is commonly used at inference to get the most likely complete SQL clause.

---

**Q4. Answer: B**

The HuggingFace `tokenizers` library is backed by a Rust implementation. When called on a list (batch) of strings, it can process them with minimal Python overhead — the Rust code loops internally. With `batched=True`, your map function receives a dict of lists (e.g., `{'text': ['sql1', 'sql2', ...]}`) and calls the tokenizer once on the entire list. Without `batched=True`, your function is called once per example (Python loop), adding thousands of Python function call overheads for a 7K-example dataset. For Spider, `batched=True` is typically 5–20× faster for tokenization.

---

**Q5. Answer: B**

Truncation at 512 tokens cuts off the end of long SQL queries. For Spider, the most complex queries involve multi-level subqueries, `INTERSECT`/`EXCEPT` operations, and long `WHERE` clause chains. If the SQL is truncated mid-clause (e.g., the `WHERE` condition is cut off), the training label is incomplete SQL. The model learns to generate SQL that ends abruptly. Additionally, if you mask the question/schema as prompt (-100) and the SQL starts near position 400, there may be only 112 tokens of SQL visible — not enough for the model to learn the full structure. Fix: filter out examples longer than your context window, or use `max_length=1024` at the cost of GPU memory.

---

**Q6. Answer: B**

HuggingFace private datasets are accessible via API token. Your colleague needs a read-access token from your account (or you grant them collaborator access on the dataset repo). They authenticate with `huggingface-cli login --token <YOUR_TOKEN>` or `os.environ["HF_TOKEN"] = "..."`, then `load_dataset("user/spider-tokenized")` works. HuggingFace Hub is a git-backed system (it uses git-lfs for large files), so cloning is also possible — but `load_dataset` is the standard access pattern.

---

**Q7 (short answer — model answer):**

**Top-k sampling:** At each step, restrict the sampling pool to only the k most probable tokens (e.g., k=40). Set all other token probabilities to 0, renormalize, then sample. This always selects from the top-40, regardless of how concentrated or spread out the distribution is.

**Top-p (nucleus) sampling:** At each step, find the smallest set of tokens whose cumulative probability exceeds p (e.g., p=0.9). Sample only from this set. The size of the nucleus adapts dynamically: when the model is confident (one token has probability 0.95), the nucleus is size 1. When the model is uncertain, the nucleus may include 50+ tokens.

For SQL generation, top-p (p=0.9–0.95) is generally preferable. SQL has predictable structure: at many positions, the model should be very confident (e.g., `GROUP` is almost always followed by `BY` — probability ~0.99). Top-p with a small nucleus will correctly force `BY` in this case. Top-k=40 would inappropriately allow 39 other tokens even when the model is very confident. However, for column names and literal values (where uncertainty is high), top-p appropriately expands the nucleus to allow diverse options.

---

**Q8 (short answer — model answer):**

Chat-format models like Qwen2.5-Coder-7B-Instruct add BOS (beginning of sequence) and chat-format special tokens when `add_special_tokens=True`. For the base (non-instruct) Qwen2.5-Coder-7B, typically only the BOS token `<|im_start|>` or `<|endoftext|>` is prepended. These signal to the model that a new sequence is beginning and help the model initialize its context correctly.

You would use `add_special_tokens=False` when: (1) you are tokenizing the middle segment of a multi-part prompt (the BOS should only appear at the very start, not in the middle); (2) you are concatenating multiple fields — question + schema + SQL — and only want BOS at the beginning of the question; (3) you are comparing raw token sequences between different models and don't want the comparison contaminated by model-specific special tokens. For fine-tuning, always use `add_special_tokens=True` for the complete input, but tokenize each field separately with `add_special_tokens=False` when building complex multi-field prompts.
