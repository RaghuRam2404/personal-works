# Week 9 Quiz Answers

---

**Q1. Answer: B**

**Why:** The encoder in a vanilla seq2seq model produces one hidden state after processing all input tokens. That single vector must encode the entire meaning of the source sentence. For long sentences (30+ tokens), the information is bottlenecked — the earlier tokens' information is almost entirely lost due to vanishing gradients in the RNN. Attention fixes this by letting the decoder access all encoder hidden states directly.

**Why others are wrong:**
- A: Embedding dimension affects representational capacity uniformly, not length-dependent degradation.
- C: Slow backprop is a training concern, not an inference bottleneck. The degradation on long sequences occurs even at inference time.
- D: Dropout affects generalization, not the fundamental length problem.

---

**Q2. Answer: B**

**Why:** The score function `v_a^T * tanh(W_a * s + U_a * h)` is a single-hidden-layer MLP applied to the (projected) decoder state and encoder annotation. The tanh gives this MLP a nonlinear activation, allowing it to learn non-trivial alignment patterns. Without it, the score function collapses to a linear function — it could only represent linear relationships between decoder state and encoder state, which would significantly limit what alignments the model can learn.

**Why others are wrong:**
- A: The tanh output range is [-1, 1], but normalization to [0,1] happens later via softmax on the scores.
- C: The dot product growth problem is what motivates the `1/sqrt(d_k)` scaling in transformers. Bahdanau attention uses tanh to bound scores, not for this reason.
- D: This is not the motivation. Correctness of the computation matters, not hardware speed.

---

**Q3. Answer: B**

**Why:** A uniform attention distribution (all weights equal) means the model is not discriminating between source positions — it is effectively computing an average of all encoder hidden states as the context vector. This usually happens when (1) softmax is applied along the wrong dimension so the mathematical constraint doesn't hold, or (2) the attention module is not in the gradient path — perhaps you detached the context vector before passing it to the decoder, or the attention parameters have zero gradients. Check `attn_weights.sum(dim=-1)` and verify gradients flow back through `context`.

---

**Q4. Answer: B**

**Why:** A unidirectional encoder at position `j` has only seen tokens `1, ..., j`. A bidirectional encoder's annotation `h_j = [h_j_fwd; h_j_bwd]` incorporates context from tokens to the right as well. This makes `h_j` a much richer representation of the word at position `j` — it knows what comes before and after. This is especially important for alignment: a word's correct alignment often depends on surrounding context (e.g., distinguishing "bank" (financial) from "bank" (riverbank) requires context from both sides).

---

**Q5. Answer: B**

**Why:** `torch.bmm` performs batched matrix multiplication. `alpha.unsqueeze(1)` reshapes to `[batch, 1, src_len]`. Multiplied with `encoder_outputs` of shape `[batch, src_len, hidden_dim]`, this gives `[batch, 1, hidden_dim]`. The `.squeeze(1)` collapses to `[batch, hidden_dim]`. This is the weighted sum over encoder positions.

**Why others are wrong:**
- A: `torch.matmul` without unsqueeze would fail or produce wrong shapes for batched inputs.
- C: Summing over `dim=0` would sum across the batch dimension.
- D: Taking the argmax and indexing is hard attention — it picks one position and is not differentiable.

---

**Q6 (short answer).**

**Additive (Bahdanau):** Score computed by an MLP: `score(s, h) = v^T tanh(W_1 s + W_2 h)`. Has three learnable parameter matrices. More expressive in theory (nonlinear), but slower — requires a forward pass through a small network for every (decoder_step, source_position) pair.

**Multiplicative (Luong/dot-product):** Score computed by dot product: `score(s, h) = s^T h` or `s^T W h`. Cheaper — just a matrix multiply. Scales well when hidden dimensions are large. The Transformer uses the scaled dot-product variant with `1/sqrt(d_k)` scaling to control variance.

**Trade-offs:** Additive is slightly more expressive; multiplicative is faster and easier to parallelize. In practice, multiplicative is dominant in modern architectures because the expressiveness gap closes with sufficient model scale.

---

**Q7 (short answer).**

Create a boolean mask: `src_mask = (src_tokens == PAD_IDX)` — True where the token is padding. Before softmax, apply: `scores = scores.masked_fill(src_mask, -1e9)`. The softmax of `-1e9` is effectively 0, so those positions get zero attention weight.

If you skip masking: the model attends to padding tokens during training. Padding embeddings are typically zero or random, so the model learns spurious patterns from meaningless positions. This rarely causes catastrophic failure on short sequences but will hurt performance on variable-length inputs where padding varies significantly across examples.

---

**Q8 (scenario).**

This is called **exposure bias**. When teacher forcing ratio = 1.0, the model always receives ground-truth previous tokens as input during training. It never learns to handle its own (potentially incorrect) predictions as input. At inference time, a single early mistake causes a cascade — the model receives its own wrong token and has never been trained on this distribution.

**Fix:** Scheduled sampling — start with a high teacher forcing ratio (e.g., 0.9) and decay it over training (e.g., to 0.0 by epoch 30). This gradually exposes the model to its own outputs. Alternatively, use a fixed ratio of 0.5 throughout training. The tradeoff: lower teacher forcing ratio makes training slower and noisier but produces a model more robust to its own errors.
