# Week 20 Quiz Answers

## Q1 — Answer: B

**Answer:** B — 7.1M parameters (768² × 12).

**Why:** Each transformer block contributes: 4 attention projection matrices (Q, K, V, O) each of size d_model² = 768² = 589,824, giving 4 × 589,824 = 2.36M from attention. The FFN has two matrices: (768 → 3072) and (3072 → 768), each with 768 × 3072 = 2.36M parameters, giving 4.72M from FFN. Total ≈ 7.07M per layer. With 8 layers: ~56.6M parameters.

**Why others are wrong:**
- A (4× d_model²): only counts attention or only counts half the FFN
- C (16× d_model²): would be true for a larger FFN expansion factor
- D (6× d_model²): incomplete accounting

---

## Q2 — Answer: C

**Answer:** C — 10.4.

**Why:** A randomly initialized model assigns uniform probability across all 32,000 vocabulary tokens. The cross-entropy of a uniform distribution over V classes is log(V). log(32000) = ln(32000) / ln(e) ... in natural log: ln(32000) ≈ 10.37. This is the theoretical baseline; observed initial loss should be within 0.5 of this value.

**Why others are wrong:**
- A (1.0): this would imply the model is already predicting with high confidence — impossible at random init
- B (3.5): log2(32000) ≈ 14.97 in bits... this doesn't correspond to any natural baseline
- D (32.0): confusing vocab size with log(vocab size)

---

## Q3 — Answer: B

**Answer:** B — Before the attention computation, to the input of each sub-layer.

**Why:** Pre-LN applies LayerNorm to the residual stream before branching into attention or FFN: `x = x + attn(LN(x))` and `x = x + ffn(LN(x))`. The original transformer used post-LN (LN after residual). Pre-LN was found to train more stably, especially at lower learning rates, because gradients flow directly through the residual without passing through a normalization layer.

**Why others are wrong:**
- A describes post-LN (the original formulation)
- C describes the final layer norm before the LM head (which exists in addition to block-level LN)
- D describes post-FFN LN, which is not pre-LN

---

## Q4 — Answer: B

**Answer:** B — Reduces parameter count and often improves perplexity.

**Why:** The input embedding is a matrix of shape (vocab_size, d_model). The LM head is (d_model, vocab_size) — the same shape transposed. Weight tying makes them share parameters, eliminating one copy. For vocab_size=32000, d_model=768, this saves 32000 × 768 × 4 bytes ≈ 96MB. Empirically, weight tying also improves perplexity because the model is forced to learn consistent representations at input and output.

**Why others are wrong:**
- A: it reduces, not increases, parameter count
- C: flash attention has no dependency on weight tying
- D: shapes are (V, d) and (d, V) — no requirement that V == d

---

## Q5 — Answer: B

**Answer:** B — 65,535.

**Why:** `uint16` is a 16-bit unsigned integer. Maximum value = 2^16 - 1 = 65,535. This is why a vocab_size of 32,000 or 50,257 (GPT-2) can be stored as `uint16`, but anything above 65,535 (e.g., some multilingual tokenizers) would require `uint32`.

---

## Q6 — Short Answer

`context_len` is the maximum sequence length baked into the model's positional embedding table — the model has no learned representation for positions beyond this. `block_size` in the data pipeline is the chunk size fed to the model during training (typically set equal to `context_len`). At inference, if you want to generate text longer than `context_len` tokens, you must use a sliding window (truncate the left side of the KV cache or recompute), since the positional embeddings beyond `context_len` do not exist. Absolute positional embeddings do not extrapolate.

---

## Q7 — Short Answer (initial loss = 6.2, expected ~10.4)

1. **Most likely — data leakage or incorrect targets:** The training targets might be the same as the inputs (predicting the current token instead of the next token), accidentally giving the model "hints." Check: `y = x[:, 1:]` not `y = x`.

2. **Second likely — tokenizer vocab mismatch:** If you loaded the wrong tokenizer (e.g., GPT-2's 50K vocab instead of your 32K vocab), the model was initialized for 32K but tokens above 32K are out-of-range → embedding lookup crash, or you are using GPT-2's tokenizer with your model's 50K embedding, giving log(50257) ≈ 10.8, not 6.2.

3. **Third likely — model parameter initialization:** If weights are initialized with abnormally large values (e.g., using `nn.init.normal_(std=1.0)` instead of `std=0.02`), the logit distribution becomes peaky, driving the loss lower artifically. The model appears confident but is not — it will fail to learn meaningful patterns.

---

## Q8 — Short Answer

GPT-2's tokenizer has vocab_size=50,257 vs. your 32,000. The embedding matrix grows from 32000 × 768 = 24.6M parameters to 50257 × 768 = 38.6M parameters — an increase of 14M parameters. For a ~56M model, this is a 25% increase in total parameters. The trade-offs: GPT-2's tokenizer is better at covering diverse English text (trained on WebText vs. your FineWeb sample), so tokenization quality will be marginally better, especially for unusual words or proper nouns. However, training your own tokenizer is the learning objective — use GPT-2's tokenizer only if you want to maximize performance at the cost of the educational experience.

---

## Q9 — Scenario Model Answer

**1. Epochs needed:** 2B / 80M = 25 epochs. This is problematic — training 25 epochs on the same data causes significant overfitting and memorization. The model will learn to reproduce the training documents rather than generalize.

**2. Is 2B tokens over-training?** Slightly: Chinchilla optimal for 56M is 56M × 20 = 1.12B tokens. Training on 2B tokens is about 1.8× over-trained by Chinchilla — this is fine for an inference-optimal small model. The issue is not the token target but the repeated data.

**3. Training time:** 2B tokens / 50K tokens/sec = 40,000 seconds ≈ 11 hours. This fits within a 24-hour Colab Pro A100 session.

**4. Recommendation:** Yes, re-generate a larger `train.bin` before starting Week 21. Aim for 500M–1B tokens to avoid excessive repetition. Use more FineWeb-Edu shards. You have the Colab Pro session available — use the first hour of Week 21 to run `prepare.py` with `max_tokens=1_000_000_000` and then start training. Do not start a 24-hour run on 80M unique tokens.
