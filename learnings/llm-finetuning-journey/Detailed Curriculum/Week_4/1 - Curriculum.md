# Week 4 — RNNs, LSTMs, and Why We Abandoned Them

## Learning Objectives

By the end of this week, you will be able to:

- Implement a vanilla RNN and an LSTM cell from scratch in PyTorch using only tensor operations.
- Explain the vanishing gradient problem in RNNs mathematically, and describe how LSTM gates address it.
- Train a character-level LSTM on a SQL corpus and generate plausible-looking SQL character sequences.
- Explain truncated backpropagation through time (TBPTT) and implement it with `detach()`.
- Articulate precisely why LSTMs were superseded by transformers: parallelization and long-range dependency handling.
- Use teacher forcing correctly during training and understand why it diverges from autoregressive inference.

---

## Concepts

### Recurrent Neural Networks (RNNs)

An RNN processes sequences one step at a time, maintaining a hidden state `h_t` that summarizes all prior inputs:

```
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

The same weight matrices (`W_hh`, `W_xh`, `W_hy`) are used at every timestep — parameter sharing over time, analogous to parameter sharing over space in CNNs.

**Backpropagation through time (BPTT):** To train, you unroll the RNN for T timesteps and backpropagate through the unrolled graph. The gradient of the loss at step T with respect to `h_0` involves a product of T Jacobians (`∂h_t/∂h_{t-1}`). If these Jacobians have spectral radius < 1, gradients vanish; if > 1, they explode.

**Vanishing gradient:** `||dL/dh_0|| ≈ ||W_hh||^T * ||dL/dh_T||`. For T=100 and `||W_hh|| < 1`, this is effectively 0. The RNN cannot learn long-range dependencies — the gradient signal from a token 100 steps back is invisible by the time it reaches the parameters.

**Practical consequence:** Plain RNNs work on sequences up to ~20 tokens. Anything longer requires LSTMs, GRUs, or attention.

### Long Short-Term Memory (LSTM)

The LSTM (Hochreiter & Schmidhuber, 1997) introduces a cell state `c_t` — a "conveyor belt" that runs through the sequence with only additive updates (no multiplication). The gates are:

```
f_t = σ(W_f * [h_{t-1}, x_t] + b_f)   # forget gate
i_t = σ(W_i * [h_{t-1}, x_t] + b_i)   # input gate
g_t = tanh(W_g * [h_{t-1}, x_t] + b_g) # new candidate values
o_t = σ(W_o * [h_{t-1}, x_t] + b_o)   # output gate

c_t = f_t * c_{t-1} + i_t * g_t        # update cell state
h_t = o_t * tanh(c_t)                  # output hidden state
```

**Why this helps:** The cell state update `c_t = f_t * c_{t-1} + i_t * g_t` is additive when `f_t ≈ 1` and `i_t ≈ 0`. The gradient flowing back through this addition has magnitude ≈ `f_t` per step — near 1, so gradients do not vanish as quickly. The forget gate can learn to keep long-range information by staying near 1.

**Intuition:** The forget gate decides what to erase from memory. The input gate decides what new information to write. The output gate controls what part of the cell state to expose as the hidden state. The cell state is the memory; the hidden state is what gets published to the outside world.

### GRU (Gated Recurrent Unit)

The GRU (Cho et al., 2014) simplifies the LSTM by merging the cell and hidden states, using two gates instead of three:

```
z_t = σ(W_z * [h_{t-1}, x_t])   # update gate
r_t = σ(W_r * [h_{t-1}, x_t])   # reset gate
h̃_t = tanh(W * [r_t * h_{t-1}, x_t])
h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
```

GRU has fewer parameters and trains faster than LSTM. On many tasks performance is similar. In practice: LSTM for longer sequences and tasks requiring nuanced memory; GRU for faster prototyping.

### Truncated BPTT and `detach()`

Unrolling for the full sequence length is expensive (O(T) memory). Truncated BPTT breaks the sequence into chunks of length `k` and detaches the hidden state between chunks:

```python
h = h.detach()  # stop gradients from flowing into previous chunk
for chunk in chunks:
    output, h = model(chunk, h)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    h = h.detach()  # MUST detach again for next chunk
```

The `.detach()` call is critical. Without it, `backward()` would try to traverse back through all previous chunks — which is both incorrect (those parameters were already updated) and memory-expensive.

### Teacher Forcing

During training, the RNN/LSTM receives the ground-truth previous token as input at each step, regardless of what it actually predicted. This is teacher forcing. It stabilizes training because early in training the model's predictions are terrible — feeding those bad predictions back would compound errors.

During inference, the model must use its own predictions (autoregressive decoding). This creates a **train/inference distribution mismatch** — the model was never trained to handle its own errors. Mitigations: scheduled sampling (gradually replace teacher tokens with model tokens as training progresses) or prefix-LM training.

SQL generation relevance: when you later fine-tune a transformer to generate SQL (Phase 4–5), teacher forcing is built into the cross-entropy training objective. Understanding the train/inference gap here explains why your fine-tuned model might hallucinate column names — it never learned to recover from its own naming errors.

### Why Transformers Won

| Property | LSTM | Transformer |
|---|---|---|
| Parallelizable across sequence | No — each step depends on previous | Yes — all positions computed simultaneously |
| Long-range dependency | Limited by gradient flow | O(1) distance via attention |
| Training speed | Sequential, slow | Massively parallelizable, fast |
| Memory during backprop | O(T) hidden states | O(T²) attention, but manageable |
| Inference | Fast (O(T)) | Slower (O(T²) per forward) |

The fundamental issue is that LSTMs are inherently sequential — you cannot compute `h_5` until you have `h_4`. This means you cannot parallelize across the sequence dimension during training, which severely limits throughput on modern matrix-multiply-optimized GPUs.

---

## Connections

**Builds on:** Week 2's language modeling objective. Week 3's concept of sequential processing.

**Unlocks:** Week 9's attention mechanism is motivated directly by LSTMs' failure to handle long-range dependencies. The teacher forcing concept reappears in Phase 5's RLHF discussion. The SQL generation project in this week is the first direct domain hook toward the final capstone.

---

## Common Misconceptions and Pitfalls

- **"LSTMs have no vanishing gradient problem."** Partial truth. LSTMs substantially mitigate it, but very long sequences (>500 steps) still degrade. Transformers handle arbitrary sequence lengths (within memory limits) via attention.
- **"Detach() = stop_gradient permanently."** `detach()` creates a new tensor that shares data but has no grad_fn. The original tensor is not modified. Only the new tensor is treated as a constant by autograd.
- **"Teacher forcing makes training easier, so use it 100% of the time."** The train/inference gap it creates is a real problem. For SQL generation at inference time, small divergences from ground truth can cause the model to spiral into garbage outputs.
- **"LSTM hidden state shape is (batch, hidden)."** In PyTorch, `nn.LSTM` returns `h_n` and `c_n` of shape `(num_layers * num_directions, batch, hidden_size)`. Forgetting the leading `num_layers` dimension is a very common shape error.

---

## Time Allocation (6–8 hours this week)

| Activity | Time |
|---|---|
| Read Colah's LSTM blog post | 1 h |
| Read Karpathy's "Unreasonable Effectiveness of RNNs" | 30 min |
| Watch StatQuest RNN video (16m) | 20 min |
| Watch StatQuest LSTM video (20m) | 25 min |
| Implement RNN cell from scratch | 1 h |
| Implement LSTM cell from scratch | 1 h |
| Train char-LSTM on SQL corpus, generate samples | 2 h |
| Journal + commit | 30 min |
