# Week 4 — TakeAway

**This week in 15 words:** LSTM gates solve vanishing gradients; parallelism and attention are why transformers won.

---

## Key Code Patterns

```python
# nn.LSTM with batch_first (preferred)
lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=2, batch_first=True)
# Input:  (B, T, input_size)
# Output: (B, T, hidden_size), (h_n, c_n)
# h_n:    (num_layers, B, hidden_size)
```

```python
# Truncated BPTT — the correct pattern
state = None
for X, Y in chunked_loader:
    if state is not None:
        state = (state[0].detach(), state[1].detach())  # CRITICAL
    logits, state = model(X, state)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
```

```python
# Temperature sampling from logits
def sample(logits, temperature=0.8):
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## LSTM Gate Reference

```
f_t = σ(W_f * [h_{t-1}, x_t] + b_f)   # forget: erase cell state
i_t = σ(W_i * [h_{t-1}, x_t] + b_i)   # input: gate new info
g_t = tanh(W_g * [h_{t-1}, x_t] + b_g) # candidate new values
o_t = σ(W_o * [h_{t-1}, x_t] + b_o)   # output: expose hidden state

c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

---

## Decision Rules

- **Forget gate ≈ 0:** Cell state erased every step. No long-term memory.
- **Forget gate ≈ 1, input gate ≈ 0:** Cell state preserved exactly. Good for stable long-term dependencies.
- **Temperature < 0.5:** Samples are near-deterministic (always peak token). Safe but repetitive.
- **Temperature > 1.2:** Near-uniform sampling. Incoherent output.
- **Temperature 0.7–0.9:** Best for SQL/code generation — diverse but structured.
- **Loss → NaN:** Add gradient clipping (`clip_grad_norm_`, max_norm 1–5). Always do this for RNNs.

---

## Numbers to Remember

- Typical LSTM hidden size: 256 (small), 512 (medium), 1024 (large)
- Gradient clip max_norm for RNNs: 5.0 (common default); 1.0 for very unstable models
- PyTorch `nn.LSTM` gate order in weights: i, f, g, o (not f, i, g, o — check the docs)
- TBPTT chunk length: 64–256 characters or tokens

---

## Red Flags During Training

- Loss goes to NaN after a few steps → missing gradient clipping; RNNs are prone to gradient explosion.
- Loss decreases then plateaus well above 1.0 → LR too low, or model too small.
- Generated samples are all the same token → temperature too low or model collapsed (check loss).
- Shape error on LSTM output → forgot `batch_first=True` or misread `h_n` shape (it has `num_layers` dim).
- Memory grows each step → not calling `.detach()` on hidden state between TBPTT chunks.
