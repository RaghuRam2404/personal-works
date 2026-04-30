# Week 8 — Answers

---

**Q1. Answer: B**

`zero_grad → forward → loss → backward → optimizer.step`. The canonical 5-step loop. `zero_grad` must come first to clear accumulated gradients from the previous step. `forward` runs the model. `loss` computes the scalar objective. `backward` populates `.grad` on all parameters. `optimizer.step` updates parameters using those gradients. Any other order causes incorrect gradient accumulation (A), wasted computation (C), or a crash — `backward` requires a computed loss (D).

---

**Q2. Answer: B — ~800K**

For a transformer decoder: each layer has roughly `12 * n_embd²` parameters (4 attention projection matrices + 2 FFN matrices with 4× expansion). For `n_embd=128`: `12 * 128² = 196,608` per layer. 4 layers ≈ 786K. Plus embedding table `vocab_size * n_embd = 80 * 128 = 10,240` and positional embedding `block_size * n_embd = 128 * 128 = 16,384`. Total ≈ 812K — order of magnitude 800K. This is correct for a tiny model appropriate for Mac CPU or a short Colab run.

---

**Q3. Answer: B**

The train/val divergence starting at step 2000 is a textbook overfitting signature: the model has memorized the training set and is not generalizing to the validation set. The validation loss increasing while train loss decreases confirms this. Best interventions in order of expected impact: (1) dropout (add 0.1–0.2 in the transformer residual stream and attention), (2) weight decay increase (try 0.1 from 0.01), (3) reduce model size, (4) collect more training data. Do not increase LR — this would accelerate the problem.

---

**Q4. Answer: B**

The core AdamW insight: in standard Adam with L2, the weight decay is added to the gradient `g_t' = g_t + λθ`. This term is then divided by `sqrt(v̂_t)` (the adaptive scaling). Parameters with large historical gradient magnitudes (large `v̂`) receive less effective regularization — they are under-regularized. AdamW applies weight decay directly to the parameter value after the Adam step: `θ ← θ - lr * (Adam update) - lr * λ * θ`. This is independent of the gradient history and provides uniform regularization across all parameters.

---

**Q5. Answer: B**

The most likely cause is that the generated sequence has reached the model's `block_size` (128 characters in the recommended config). After exactly 128 characters, the generation loop terminates. The sequence `SELECT t1.name FROM t2 WHERE t1.id = t2.id GROUP` is approximately 50 characters, so this is not the immediate issue — however, if your generation code stops after `block_size - len(seed)` new characters and the seed was long, this truncation is the cause. Fix: increase `max_new_tokens` in your generation loop, or use a shorter seed. A secondary possibility (D) is that the model genuinely hasn't learned `GROUP BY` as a unit — but at 5000 steps and a SQL-heavy corpus, this is unlikely.

---

**Q6. Answer: A**

HuggingFace's `AutoModelForCausalLM.forward()` handles the label shift internally. If you pass `labels = input_ids`, the model computes loss between `logits[:, :-1, :]` (predictions at all positions) and `labels[:, 1:]` (the ground-truth next tokens). The shift is done internally in the model's loss computation. If you manually shift `labels` before passing them, you double-shift and the model tries to predict two positions ahead — a subtle bug that produces higher loss and slower convergence. Trust the HuggingFace API on this: pass `labels = input_ids` (with -100 for padding) and let the model handle the shift.

---

**Q7 (short answer — model answer):**

A byte-level BPE tokenizer starts with a vocabulary of all 256 byte values (0–255). Japanese text is encoded in UTF-8: each Japanese character is 3 bytes. The comment `-- ユーザー名` would be encoded as bytes first: `-`, `-`, ` `, then the 3-byte UTF-8 sequences for ユ, ー, ザ, ー, 名. If the BPE training corpus contained Japanese text, some of these byte sequences may have been merged into multi-byte tokens. If not, they fall back to individual byte tokens.

There is no error. The byte-level foundation guarantees that any Unicode text — including Japanese, Arabic, emoji, or SQL special characters — can always be encoded. The token sequence may be longer (more bytes per character for non-ASCII), but the tokenizer never raises OOV exceptions. This is the key advantage over word-level or Unicode-character-level tokenizers for a system that might encounter multilingual SQL comments from users.

---

**Q8 (short answer — model answer):**

The code has three bugs:

1. **Missing `optimizer.zero_grad()` before `loss.backward()`:** Gradients accumulate across steps. By step 2, all gradients are 2× the correct value; by step 100, they are 100× larger. The optimizer applies massively incorrect updates, causing either NaN loss or completely random weight movements. Fix: add `optimizer.zero_grad()` at the start of each loop iteration.

2. **`loss = model(x, y)` — the model is called without capturing its output correctly.** If `model(x, y)` returns a tuple `(logits, loss)` (as in nanoGPT), then `loss` here is the tuple itself, not the scalar loss. Calling `loss.backward()` on a tuple will raise a `RuntimeError`. Fix: `logits, loss = model(x, y)` (or `output = model(x, y); loss = output.loss` for HuggingFace models).

3. **No gradient clipping.** Without `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, gradient explosions (especially early in training) can cause NaN loss and corrupt model weights permanently. Fix: add the clipping call between `loss.backward()` and `optimizer.step()`.

---

**Q9 (short answer — model answer):**

No, this is not a sign of failure — it is the expected behavior for a char-level or token-level language model trained without schema information.

The model has learned SQL syntax: keyword order (SELECT before FROM before WHERE), clause structure (GROUP BY, ORDER BY), and the general pattern of SQL. This is exactly what Phase 1 training is supposed to produce. The gibberish table and column names (`t1_x3`, `col_g2`) are correct: the model was trained only on raw SQL queries without any schema context. It cannot know that `t1` refers to the `orders` table in a specific database — it only knows that identifiers often look like `t1`, `t2`, `col1`, etc., in Spider's query patterns.

To fix this in a real system: the model needs schema-grounded fine-tuning. The input must include the database schema (table names, column names, data types), and the model must learn to generate SQL that uses only the provided schema. This is exactly what Phase 4 will teach: fine-tuning with schema-conditioned prompts. The capstone model demonstrated that the generative training objective works — now the task is to condition it on the right context.
