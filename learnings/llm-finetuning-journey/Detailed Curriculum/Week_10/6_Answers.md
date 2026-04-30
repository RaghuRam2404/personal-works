# Week 10 Quiz Answers

---

**Q1. Answer: B**

**Why:** Dividing by `d_k` instead of `sqrt(d_k)` makes the scaling too aggressive for small `d_k` but correct for `d_k=1`. For `d_k=64`, you'd be dividing by 64 instead of 8 — scores will be 8x smaller than intended. This doesn't cause NaN, but it severely compresses the score range, making softmax outputs near-uniform (very flat distribution). The model still trains but behaves as if all positions are equally relevant, which heavily limits what the attention can learn.

**Why others are wrong:**
- A: Division produces small values, not NaN. NaN comes from division by zero, exp overflow of large values, or log of zero.
- C: Smaller scores → flatter softmax → less concentration, not more.
- D: Scores can be any real value; softmax output is always non-negative.

---

**Q2. Answer: B**

**Why:** The FFN applies an independent, position-wise transformation: `FFN(x) = max(0, xW_1+b_1)W_2+b_2`. The inner dimension is 4x larger than d_model (2048 vs 512), giving the network significant expressiveness. Attention is a weighted averaging operation — it cannot compute arbitrary nonlinear functions of the attended values. The FFN provides that nonlinearity and has been interpreted as a key-value memory: the first layer encodes "keys" and the second layer projects back, effectively memorizing factual associations.

---

**Q3. Answer: B**

**Why:** The decoder generates tokens autoregressively. At each step, it first applies masked self-attention to its own previously generated tokens — this allows the decoder to understand what it has generated so far. Only then does it query the encoder via cross-attention, asking: "given what I've generated so far, what encoder information is most relevant?" The FFN then applies a nonlinear transformation. This ordering ensures the cross-attention query reflects the full context of the generated prefix.

---

**Q4. Answer: B**

**Why:** Sinusoidal encodings satisfy: `PE(pos+k) = M_k * PE(pos)` where `M_k` is a fixed rotation matrix (linear in `PE(pos)`). This means the model can learn attention patterns based on relative offsets (e.g., "attend to the token 3 positions before") and this learning generalizes to positions beyond the training set. Learned positional embeddings have no such structure — positions beyond the training length have no meaningful embedding.

---

**Q5. Answer: B**

**Why:** In single-head: one `W_Q` matrix of shape `[512, 512]` = 262,144 parameters. In 8-head: each `W_Q` is `[512, 64]` = 32,768 parameters × 8 heads = 262,144 parameters — identical. The same holds for `W_K`, `W_V`. The output projection `W_O` is `[512, 512]` in both cases. Total parameter count is the same; the heads simply partition the parameter budget.

---

**Q6. Answer: C**

**Why:** Residual connections are the primary gradient highway in deep networks. Without `x + Sublayer(x)`, gradients must flow through every sublayer in series — through attention, through layer norm, through FFN. In a 6-layer network, this is 12+ sequential operations. Gradients will shrink at each step (or explode, then get clipped). The network is effectively a 12-layer non-residual network, which is very difficult to train. You would see near-zero gradients for the first few layers in W&B's gradient histograms.

---

**Q7 (short answer).**

Causal masking prevents position `i` in the decoder from attending to positions `j > i` (future tokens). Implementation: create a boolean upper-triangular matrix of shape `[seq_len, seq_len]` where `True` indicates "mask this position". Before softmax in the self-attention of the decoder, apply `scores.masked_fill(causal_mask, -1e9)`. After softmax, the `-1e9` positions become ≈ 0.0, so the weighted sum excludes future tokens.

In PyTorch: `causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)`.

If you forget the causal mask during training: the model can attend to future ground-truth tokens — it sees the answer it's supposed to predict. Training loss will be artificially low, but at inference time the model has no future tokens to attend to and will produce incoherent output. This is the autoregressive equivalent of cheating on an exam.

---

**Q8 (short answer).**

Loss at 2.3 ≈ -log(1/10) means the model is outputting a near-uniform distribution over 10 tokens — it hasn't learned anything.

**Hypothesis 1 (most likely): Causal mask missing or wrong.** Without the mask, the decoder attends to future tokens during training but gets nonsense at inference, leading to a degenerate uniform output. Diagnose: print `attn_weights[0, 0, 0, 1:]` — should be all zeros; if nonzero, mask is broken.

**Hypothesis 2 (second): Learning rate too low.** The warmup schedule may have warmup_steps set too high or the base LR is too small. Diagnose: print the LR at each step in W&B. It should reach a peak around step 400 and then decay.

**Hypothesis 3 (third): Positional encoding not being added to embeddings.** Without positional information, every position looks identical to the model; it can only output the marginal distribution. Diagnose: print `model.src_embed(src_tokens)` vs. `model.src_embed(src_tokens) + positional_encoding` — they should differ.
