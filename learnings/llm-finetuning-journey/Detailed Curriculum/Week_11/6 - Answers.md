# Week 11 Quiz Answers

---

**Q1. Answer: B**

**Why:** Causal language modeling trains the model to predict the next token at each position. For a sequence `x_1, ..., x_T`, the target at position `t` is `x_{t+1}`. In code this is implemented as shifted targets: `loss = cross_entropy(logits[:, :-1, :], x[:, 1:])`. The causal mask ensures that the logits at position `t` are only computed from tokens `x_1, ..., x_t`, enforcing the autoregressive property.

---

**Q2. Answer: B**

**Why:** `nn.Parameter` tensors are included in `model.parameters()` and are updated by the optimizer with gradients. The causal mask is a fixed lower-triangular binary tensor — it should never change. If it is a parameter, the optimizer will update it each step, gradually destroying the zero/one structure. After a few gradient updates, the mask becomes real-valued and the causal attention constraint breaks. Always register fixed buffers (like masks, positional encoding tables) with `register_buffer`.

---

**Q3. Answer: B**

**Why:** Position embeddings are indexed by position. For a sequence of length T, the positions are 0, 1, ..., T-1. The correct call is `pos_emb = self.wpe(torch.arange(T, device=device))` which selects the first T rows of the embedding table. This works for any T ≤ block_size. The model was designed to handle sequences up to block_size, not exactly block_size.

---

**Q4. Answer: B**

**Why:** In Post-LN, the residual connection is inside the normalization: `LN(x + Sublayer(x))`. At initialization, `Sublayer(x) ≈ 0`, so `LN(x + 0) = LN(x)` — this is fine. But as training proceeds, gradients must flow through `LN` before reaching the residual path in each layer, which can cause instability for deep networks. In Pre-LN, the residual stream `x` is untouched: `x + Sublayer(LN(x))`. Gradients can flow directly through `x` at every layer, providing a clean highway. This is why GPT-2 and all subsequent large models adopted Pre-LN.

---

**Q5. Answer: B**

**Why:** Decreasing training loss with degrading generation quality is a classic overfitting symptom. The model has memorized training examples and is losing generalization. Check val loss in W&B — if it's increasing after step 4000 while train loss falls, you have overfitting. Mitigations: add dropout (try 0.2), reduce n_embd, or use more training data. Weight decay in AdamW also helps.

---

**Q6 (short answer).**

Weight tying shares the same parameter tensor for the input embedding matrix and the output projection:

```python
self.lm_head.weight = self.transformer.wte.weight
```

Beyond parameter savings, it improves coherence because the same matrix governs both how tokens are represented as inputs AND how the model distributes probability over output tokens. If two tokens are similar in embedding space (close vectors), they will also have similar logit profiles, meaning the model treats them as interchangeable in similar contexts. Without weight tying, the embedding space and the logit space can drift apart during training, causing inconsistencies where semantically similar tokens produce very different outputs.

---

**Q7 (short answer).**

For text generation with no separate source text, a decoder-only transformer is more appropriate. An encoder-decoder model always encodes an input sequence first and then cross-attends to it — for pure generation, the "encoder input" would have to be an artificial prompt, adding unnecessary computation. The decoder-only model avoids: (1) the entire encoder stack (6 layers of bidirectional self-attention), and (2) cross-attention in every decoder layer. For pretraining on language, this is both more computationally efficient and conceptually cleaner — the same mechanism (causal self-attention) handles both understanding and generation.

---

**Q8 (scenario).**

Temperature controls how peaked or flat the softmax distribution is over the vocabulary. At temperature 1.0, logits are used as-is. At temperature 0.3, logits are divided by 0.3 before softmax, making the distribution much more peaked — the highest-probability token dominates. SQL is a highly constrained language: after `SELECT`, the next tokens should almost always be column names or `*`, never another keyword like `FROM`. At temp 1.0, low-probability tokens (like a second `SELECT`) are sampled more often, producing invalid SQL. At temp 0.3, the model stays close to the distribution peak, consistently picking the highest-probability (syntactically correct) token.

The risk of very low temperature: the model becomes deterministic and greedy. For SQL generation in text-to-SQL systems, this is often desirable — you want the most likely correct SQL. But for creative generation, low temperature produces repetitive, boring output. For SQL, temperatures in [0.2, 0.5] tend to work well in practice.
