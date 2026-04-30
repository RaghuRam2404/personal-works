# Week 7 — Quiz

---

**Q1.** You call `model(**inputs)` and inspect `logits[0, -1, :]`. What exactly does this tensor represent?

A) The probability distribution over all possible completions of the full sequence.
B) The model's predicted logit scores for each vocabulary token as the next token after the last token in `inputs.input_ids[0]`.
C) The hidden state of the last token in the last transformer layer.
D) The embedding of the last input token.

---

**Q2.** You train a causal LM with `labels = input_ids` (no -100 masking). The batch includes padded sequences. What is the consequence?

A) No consequence — the `attention_mask` prevents the model from attending to padding, so loss at padding positions is naturally zero.
B) The model will try to predict the padding token `<|endoftext|>` at padded positions, adding spurious loss that degrades training. The model learns that `<|endoftext|>` is a valid continuation of SQL, which may cause premature EOS generation.
C) PyTorch's CrossEntropyLoss automatically ignores positions where `labels == pad_token_id`.
D) The loss will be NaN due to padding token IDs being out of range.

---

**Q3.** `model.generate()` with `do_sample=False` is called "greedy decoding." What is its limitation compared to beam search?

A) Greedy decoding is slower than beam search.
B) Greedy decoding always selects the token with the highest probability at each step. This can get "stuck" in a locally optimal sequence that is globally suboptimal — e.g., committing to `SELECT COUNT` when `SELECT AVG` would lead to a better overall continuation.
C) Greedy decoding does not support `temperature` or `top_k`.
D) Greedy decoding cannot generate SQL with special characters.

---

**Q4.** The `datasets` library uses `batched=True` in `.map()`. What does this change, and why is it faster for tokenization?

A) It processes examples in parallel across multiple CPU cores.
B) It passes a batch (dict of lists) to the map function instead of a single example, allowing the tokenizer to process many examples at once (using its internal batching). This avoids Python overhead per example and lets the fast Rust tokenizer handle the bulk.
C) It loads the dataset into GPU memory for faster processing.
D) It enables automatic caching of the map result.

---

**Q5.** You tokenize Spider with `max_length=512, truncation=True`. 3% of examples are longer than 512 tokens and get truncated. What is the risk for training?

A) Truncation causes a memory error in the DataLoader.
B) Truncated examples have incomplete SQL at the end, so the model may learn to generate truncated or cut-off SQL. More critically, if the last tokens of the SQL clause are the `FROM` or `WHERE` clause, the model never sees the complete structure, degrading fine-tuning quality for complex queries.
C) The `attention_mask` is incorrect for truncated examples.
D) Truncation always produces incorrect gradients.

---

**Q6.** You push a dataset with `push_to_hub("user/spider-tokenized", private=True)`. A colleague wants to use it from a Colab notebook. What must they do?

A) The dataset is private — they cannot access it.
B) They must have your HuggingFace access token and either log in via `huggingface-cli login` or set `HF_TOKEN` as an environment variable. Then `load_dataset("user/spider-tokenized", token=os.environ["HF_TOKEN"])` works.
C) Private datasets can only be downloaded directly — the Hub API doesn't support them.
D) They must clone the git repository behind the dataset.

---

**Q7 (short answer).** Explain the difference between `top_k` sampling and `top_p` (nucleus) sampling. For SQL generation, which would you prefer and why?

---

**Q8 (short answer).** You load Qwen2.5-Coder-7B and run `tok.encode("SELECT id FROM users", add_special_tokens=True)`. The output includes extra token IDs before `SELECT`. Explain what these are, why they are added, and when you would want to suppress them (use `add_special_tokens=False`).
