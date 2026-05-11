# Week 2 — Answers

---

**Q1. Answer: B — Underfitting**

When train loss and val loss are nearly equal and both are high, the model is not fitting the training data well — this is underfitting, not overfitting. Overfitting shows up as train loss >> val loss (train loss much lower). Possible fixes: increase model capacity (more hidden units, more layers), increase LR, train for more steps, or reduce regularization (less dropout, less weight decay).

**Why others are wrong:**
- A: Dropout would make underfitting worse by adding more regularization.
- C: A batch norm bug would typically show as unstable loss or a train/eval gap, not both high.
- D: Dying ReLU would show as loss stopping at the log-uniform baseline (~3.0 for 27 classes) and never improving at all.

---

**Q2. Answer: C — Kaiming (He) initialization**

ReLU sets negative inputs to zero, which effectively halves the variance of the output compared to the input. Xavier init was derived assuming activations have near-linear behavior around zero (valid for Tanh/Sigmoid). For ReLU, you need to multiply the variance by 2 to compensate: `std = sqrt(2 / fan_in)`. This is the He/Kaiming formula.

**Why others are wrong:**
- A: Xavier was derived for symmetric activations. ReLU is not symmetric and Xavier underestimates the needed scale.
- B: All-ones initialization is a catastrophic mistake — all neurons compute identical outputs and identical gradients. No learning occurs (symmetry is never broken).
- D: Small weights cause vanishing pre-activations through deep networks. The gradients through ReLU become near-zero, and the network cannot learn.

---

**Q3. Answer: B — `(batch_size, 30)`**

Each of the 3 context tokens is looked up in the embedding table to get a vector of size 10. After lookup, the result is shape `(batch_size, 3, 10)`. Flattening the last two dimensions: `3 * 10 = 30`. So the input to the first linear layer is `(batch_size, 30)`.

---

**Q4. Answer: B**

`nn.BatchNorm1d` has two modes: training mode (uses batch mean/var) and eval mode (uses running mean/var accumulated during training via exponential moving average). If you forget `model.eval()`, batch norm at inference time uses the mean and variance of whatever the current batch is — which is a single test batch, statistically different from the training distribution. This produces shifted normalizations and poor accuracy.

---

**Q5. Answer: B — 4.0**

PyTorch uses inverted dropout: at training time, the surviving neurons are scaled by `1/(1-p)`. If `p=0.5`, surviving neurons are multiplied by 2. A neuron with activation `h=4.0` has 50% chance of being kept (→ value = `4.0 * 2 = 8.0`) and 50% of being dropped (→ value = 0). Expected value: `0.5 * 8.0 + 0.5 * 0 = 4.0`. At inference time with dropout disabled, the raw activation 4.0 is used directly — the expected values match without needing any scaling. This is why inverted dropout is the standard: it makes inference-time code simpler.

---

**Q6. Answer: B**

The bigram model only conditions on the single previous token. It learns `P(next | prev)` — a 27×27 (or ~V×V) table of probabilities. It cannot model that `GROUP BY` or `ORDER BY` almost always appear together as a unit because that requires seeing two tokens back. The MLP with context 4 conditions on `P(next | prev_1, prev_2, prev_3, prev_4)` and can learn these multi-token patterns. This is the fundamental motivation for longer-context models.

---

**Q7 (short answer — model answer):**

Batch normalization normalizes over the batch dimension: for each feature, it computes the mean and variance across all examples in the mini-batch `(N, D)` → statistics have shape `(D,)`. Layer normalization normalizes over the feature dimension: for each example, it computes mean and variance across all features `(N, D)` → statistics have shape `(N,)`.

Layer norm is preferred in transformers for two reasons: First, the batch size in transformer training can be 1 (or even sub-1 with gradient accumulation), and batch norm becomes unstable with very small batches (the batch statistics are noisy). Layer norm has no dependence on batch size. Second, transformer sequence modeling involves variable-length sequences — layer norm applies independently per token position, while batch norm would require fixed sequence lengths to compute statistics consistently.

---

**Q8 (short answer — model answer):**

Repeated tokens (e.g., `select select select`) indicate that the model has learned that `SELECT` frequently follows `SELECT` (possibly because `SELECT` is the most common token in SQL, so the model defaults to it). The context is not being used effectively, or the temperature during sampling is too high. Fixes: lower the sampling temperature to sharpen the distribution (or use top-k sampling), increase context length so the model can see that it just generated `SELECT` and should not do so again, train longer so the model learns more nuanced patterns, or add a repetition penalty during sampling.

---

**Q9 (short answer — model answer):**

All-zero initialization is catastrophic due to symmetry. In the first forward pass, every neuron in a layer receives identical inputs (since `0 @ anything = 0`), produces an identical pre-activation (0.0), and after any activation function, an identical output. During backprop, the gradients flowing back through each neuron are identical because all neurons are indistinguishable. After `optimizer.step()`, all weights are updated by exactly the same amount. The network remains symmetric — every neuron in a layer is still identical after the update. This symmetry is never broken. The network effectively acts as a single neuron regardless of width. This is why random initialization is essential: it breaks symmetry so each neuron can specialize.
