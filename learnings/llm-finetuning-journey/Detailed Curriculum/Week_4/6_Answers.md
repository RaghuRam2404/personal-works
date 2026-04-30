# Week 4 — Answers

---

**Q1. Answer: B**

The gradient of the loss at step T with respect to the hidden state at step 1 is `∂L/∂h_1 = (∂h_T/∂h_1) * ∂L/∂h_T`. By the chain rule, `∂h_T/∂h_1 = ∏_{t=2}^{T} ∂h_t/∂h_{t-1}`. Each factor is `diag(1 - h_t²) * W_hh` (the Jacobian of tanh-RNN). If the spectral radius of `W_hh` (modified by tanh) is < 1, this product shrinks exponentially with T. For T=499, even a factor of 0.99 per step gives `0.99^499 ≈ 0.007`. The gradient is effectively zero. The model cannot adjust early weights based on late-sequence errors.

---

**Q2. Answer: C**

Without `detach()`, the computation graph for the current chunk includes all graph nodes from the previous chunk, which was built during the previous iteration (where `optimizer.step()` already modified the parameters). This has two effects: memory grows linearly with the number of chunks processed, and the gradients flowing back into prior chunks are meaningless (those parameters were already updated). In practice this often causes out-of-memory errors or very slow training. Detaching explicitly severs the graph at the chunk boundary.

---

**Q3. Answer: C — `(2, 32, 256)`**

`h_n` has shape `(num_layers * num_directions, batch_size, hidden_size)`. For 2 layers, unidirectional: `(2, 32, 256)`. A common bug is treating `h_n` as shape `(batch, hidden)` and getting shape errors when passing it to the next chunk's LSTM call.

---

**Q4. Answer: B**

When `f_t ≈ 0`, the cell update is `c_t = 0 * c_{t-1} + i_t * g_t = i_t * g_t`. The previous cell state is completely erased. At every step, the network starts fresh. This eliminates the LSTM's ability to carry long-range information — it degrades to a network that only uses the current and very recent inputs, similar to a vanilla RNN. The forget gate value near 1 is the "no-op" state (preserve memory); near 0 is the "hard reset" state.

---

**Q5. Answer: B — Temperature too high**

When temperature `T` approaches infinity, `softmax(logits/T)` approaches a uniform distribution — all tokens are equally likely. When it is very high (>1.5), the model samples nearly randomly, ignoring the learned probability peaks. The result is incoherent repetition of high-frequency tokens (in SQL, `SELECT` and `FROM` are very frequent, so they dominate even a near-uniform sample). Fix: lower temperature to 0.7–0.8 for SQL generation.

---

**Q6. Answer: B**

Backpropagation through time requires storing the hidden state and all intermediate activations at every time step to compute gradients. For a 2-layer LSTM with hidden size 256 and batch size 32 over 50,000 steps, the memory for hidden states alone is `50,000 * 2 * 32 * 256 * 4 bytes ≈ 3.3 GB` — likely exceeding GPU VRAM. The backward pass also takes O(T) sequential time. TBPTT reduces this to `O(chunk_size)` memory per backward pass.

---

**Q7 (short answer — model answer):**

During training with teacher forcing, at every step t the LSTM receives the ground-truth character (or token) as input, regardless of what it predicted at step t-1. At inference, it must receive its own predictions as input — there is no ground truth. For `SELECT t1.id, t2.name FROM table1`:

If the model mispredicts a character — say generating `t1.ix` instead of `t1.id` at position 6 — during training it would never encounter this error state. At inference, the erroneous `x` becomes the next input, and the model has no learned behavior for "how to continue after `t1.ix`." This causes cascading errors: each mistake shifts the input distribution further from the training distribution, compounding into nonsensical output. In SQL, this manifests as a single wrong character causing the model to abandon the current clause entirely and hallucinate column names or syntax.

---

**Q8 (short answer — model answer):**

**Reason 1 — Parallelization:** LSTMs must compute `h_t` before `h_{t+1}` — the computation is inherently sequential. On a GPU with thousands of cores, this means at any given moment only one sequential step is being processed, leaving most compute idle. Transformers compute all position outputs simultaneously via matrix multiplications over the full sequence, utilizing all GPU cores in parallel. For long sequences, this is a 10–100× throughput advantage.

**Reason 2 — Long-range dependencies:** In an LSTM, information from token 1 must pass through every intermediate hidden state to reach token 500. Each step multiplies by gate values that can attenuate or distort the signal. In a transformer with self-attention, token 1 and token 500 interact directly in O(1) computational steps — the attention score between them is computed in a single matrix multiply. There is no "path" through intermediate states. This architectural property allows transformers to learn dependencies across the full context length, which LSTMs cannot reliably do.
