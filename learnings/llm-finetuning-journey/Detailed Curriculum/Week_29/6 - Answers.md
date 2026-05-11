# Week 29 Quiz Answers

## Q1 — Answer: C

**Answer:** C. Chat template not applied; SFTTrainer cannot detect prompt/response boundary.

**Why:** `SFTTrainer` uses role markers in the chat template to determine which tokens are prompt (masked from loss) and which are response (included in loss). If you provide raw string data without the `<|im_start|>role<|im_end|>` structure, `SFTTrainer` has no way to identify the boundary and may include all tokens in the loss — or may apply a naive heuristic. Always apply `tokenizer.apply_chat_template` and verify one formatted example visually before training.

**Why others are wrong:**
- A: High LR causes loss spikes or divergence, not incorrect masking behavior.
- B: `add_generation_prompt=True` appends `<|im_start|>assistant` without a closing response — used for inference, not training.
- D: Gradient checkpointing affects memory, not loss masking.

---

## Q2 — Answer: B

**Answer:** B. Overfitting — stop at the eval loss minimum.

**Why:** When train loss drops to 0.05 on 1K examples but eval loss rises from 1.8 to 2.5, the model has memorized the training set. It has not learned to generalize to new (schema, question) pairs. The correct approach: monitor eval loss throughout training and stop (or roll back the checkpoint) when eval loss bottoms out — here, around step 200. For fine-tuning on small datasets, early stopping is essential.

**Why others are wrong:**
- A: Low train loss indicates memorization, not good generalization.
- C: Eval loss diverging from train loss in this pattern is the canonical overfitting signal; the eval set is not corrupted.
- D: LR schedule affects convergence speed but does not cause eval loss to rise while train loss falls; that is overfitting.

---

## Q3 — Answer: C

**Answer:** C. ~6GB.

**Why:** In fp16/bf16 mixed precision: model weights = 2 bytes × 500M = 1GB. Gradients (same dtype) = 1GB. AdamW optimizer stores first and second moment estimates in fp32: 4 bytes × 500M × 2 = 4GB. Total ≈ 6GB before activations. Activations depend on batch size and sequence length but add 0.5–2GB typically. On a 16GB GPU this is comfortable.

---

## Q4 — Answer: B

**Answer:** B. Concatenates multiple short examples into one sequence up to `max_seq_length`.

**Why:** Most SQL (schema + question + answer) pairs are short — often under 200 tokens. Without packing, each batch position uses only a fraction of the 512-token `max_seq_length`, wasting GPU compute on padding. With `packing=True`, `SFTTrainer` bins examples together using EOS-token separators so each position is full. This can increase throughput by 2–3x on short-sequence datasets like SQL.

---

## Q5 — Answer: B

**Answer:** B. Input not formatted with chat template.

**Why:** `Qwen2.5-0.5B` after SFT fine-tuning is calibrated to respond to the `<|im_start|>user ... <|im_end|><|im_start|>assistant` format. If you tokenize a raw question string without the template, the model sees an unfamiliar input distribution and generates garbage. The correct inference call: format the message with `tokenizer.apply_chat_template([{"role":"user","content":"..."}], add_generation_prompt=True, return_tensors="pt")`.

**Why others are wrong:**
- A: If training was done correctly, the model is fine — the issue is inference formatting.
- C: A missing tokenizer would cause an error when loading, not garbage output.
- D: `model.generate` works fine with fine-tuned causal LMs.

---

## Q6 — Short Answer

Qwen2.5-7B has 7 billion parameters. In fp16 mixed precision, model weights alone = 14GB, which already exceeds a 16GB GPU. Adding gradients (another 14GB in fp16) makes full SFT impossible without multi-GPU or offloading. AdamW optimizer states add another 28GB in fp32. Total for full SFT on a 16GB GPU: ~56GB — infeasible. For 0.5B: ~6GB total, easily fitting in 16GB. LoRA solves the 7B problem by training only rank-r adapter matrices (typically <1% of parameters), whose gradients and optimizer states are tiny.

---

## Q7 — Short Answer

**Hypothesis 1: Insufficient data coverage.** The 1K training set may not contain examples with the specific table names in the test set. SFT teaches format, not schema memorization — if `customers` never appeared in training, the model may default to a seen table name like `orders`. Fix: increase dataset diversity.

**Hypothesis 2: Schema conditioning is failing.** If the schema in the user message is not being attended to properly (due to formatting errors or very long schemas truncated at max_seq_length), the model may generate SQL based on question semantics alone rather than the provided schema. Fix: verify full schema appears in model input, reduce prompt length, increase `max_seq_length`.

---

## Q8 — Short Answer

Train on all 10K, but weight the 1K PostgreSQL examples more heavily. SFT on 9K generic SQL examples teaches the model correct SQL structure, JOINs, aggregations, and filtering — all of which transfer directly to PostgreSQL. The 1K PostgreSQL-specific examples then teach PostgreSQL-specific constructs on top of that foundation. Training on 10K is better than 1K because more diverse data reduces overfitting. However, if your held-out test set is PostgreSQL-specific, consider upsampling the 1K PostgreSQL examples (repeat them 3–5x in the training set) so the PostgreSQL distribution is not overwhelmed by generic SQL.

---

## Q9 — Scenario Answer

**Ranked hypotheses with fixes:**

1. **Gradient explosion from a malformed batch (most likely).** One training example contains an extremely long sequence that, when processed with a high LR, produces a very large gradient. Fix: add `max_grad_norm=1.0` to `SFTConfig` (gradient clipping). This is the single most common cause of loss spikes.

2. **Learning rate too high for this model size.** `2e-4` is appropriate for LoRA; for full SFT on a small model, `2e-5` to `5e-5` is safer. Fix: reduce LR to `1e-5` and add a warmup period of 5–10% of total steps.

3. **Mixed-precision numerical instability.** Some operations in bf16 or fp16 can produce NaN if the input range is too large. Fix: add `bf16=True` (bf16 is more numerically stable than fp16 for training) or add `fp16_full_eval=False`.

4. **A corrupted training example with very large label IDs.** If one example has token IDs that map to a vocabulary position causing embedding lookup issues. Fix: add a data validation step that filters examples with any token ID >= `tokenizer.vocab_size`.
