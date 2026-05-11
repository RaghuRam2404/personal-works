# Week 2 — Quiz

Calibration: junior ML engineer interview level.

---

**Q1.** You train an MLP with ReLU activations on a names dataset. After 2000 steps, train loss is 2.5 and val loss is 2.5 — they are nearly identical and both high. The loss curve is smooth and decreasing slowly. This is most likely:

A) Overfitting — add dropout.
B) Underfitting — the model is too small or LR is too low.
C) A batch norm bug — batch norm is suppressing the activations.
D) A dying ReLU problem — all neurons are outputting 0.

---

**Q2.** You switch your activation from Tanh to ReLU and training becomes unstable (loss spikes). You suspect an initialization problem. Which initialization should you use for ReLU and why?

A) Xavier (Glorot) — designed for all activations.
B) All weights set to 1.0 — symmetry breaking is not needed with ReLU.
C) Kaiming (He) — accounts for ReLU zeroing out half the inputs, preserving variance across layers.
D) Uniform random in [-0.01, 0.01] — small weights are always safe.

---

**Q3.** In makemore's MLP, the embedding layer `C` has shape `(27, 10)`. The context length is 3. What is the shape of the input to the first linear layer after embedding lookup and flattening?

A) `(batch_size, 27)`
B) `(batch_size, 30)`
C) `(batch_size, 10)`
D) `(batch_size, 3, 10)` — no flattening needed.

---

**Q4.** Your model uses `nn.BatchNorm1d`. During training, loss is great. At eval time, accuracy drops by 15 percentage points. The most likely cause is:

A) The learning rate is too high for inference.
B) You forgot to call `model.eval()` before evaluation, so batch norm still uses batch statistics instead of the running mean/variance accumulated during training.
C) Batch norm is incompatible with small batch sizes during inference.
D) The `gamma` and `beta` parameters need to be re-trained without batch norm.

---

**Q5.** Dropout with `p=0.5` is applied during training. You are computing the expected output of a neuron with pre-dropout activation `h = 4.0`. What is the expected value of the post-dropout activation, and why?

A) 2.0 — half of `h`, because half the neurons are zeroed.
B) 4.0 — dropout uses inverted dropout scaling (`h / (1-p)`), so surviving neurons are scaled up to maintain the expected value of 4.0.
C) 8.0 — the surviving neuron is doubled to compensate for the dropped neurons.
D) 0.0 — the neuron is dropped with probability 0.5 and the expected value is 0.

---

**Q6.** You are training a bigram model (lookup table) on SQL keywords. The training loss is 2.3 nats. Your MLP with context length 4 achieves 1.8 nats. Which statement best explains the improvement?

A) The MLP has more parameters, so it memorizes training data better.
B) The MLP can condition on longer context (4 previous tokens vs. 1), allowing it to learn that `GROUP` is almost always followed by `BY`, and `ORDER` by `BY`, which the bigram cannot capture because it only sees one prior token.
C) The MLP uses better optimization (Adam vs. the bigram's count-based approach).
D) The embedding layer in the MLP compresses the input more efficiently.

---

**Q7 (short answer).** Explain the difference between batch normalization and layer normalization in terms of what dimension is normalized and why layer norm is preferred in transformers.

---

**Q8 (short answer).** You have an MLP trained for SQL keyword prediction. The model generates samples like `select select select from from where`. What does this tell you about the model and what would you change to fix it?

---

**Q9 (short answer).** Your colleague proposes initializing all weights in an MLP to exactly 0.0. Explain concretely what happens during the first forward and backward pass.
