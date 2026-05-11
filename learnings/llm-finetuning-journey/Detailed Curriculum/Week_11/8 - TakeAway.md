# Week 11 TakeAway — Decoder-Only Transformers / GPT

**This week in 15 words:** Drop the encoder; predict next tokens autoregressively; weight-tied embeddings; causal mask enables parallelism during training.

---

## Key Formula

```
# Causal LM loss (shifted targets)
loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), x[:, 1:].reshape(-1))

# Temperature sampling
logits_scaled = logits / temperature
probs = F.softmax(logits_scaled, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

---

## Key Code Patterns

```python
# Causal mask (register as buffer, not parameter)
self.register_buffer('bias',
    torch.tril(torch.ones(block_size, block_size)).view(1,1,block_size,block_size))

# Weight tying
self.lm_head.weight = self.transformer.wte.weight

# Pre-LN Block (modern convention)
def forward(self, x):
    x = x + self.attn(self.ln1(x))   # attention with pre-norm
    x = x + self.mlp(self.ln2(x))    # FFN with pre-norm
    return x

# Generate autoregressively
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx
```

---

## Numbers to Remember

| Config | Shakespeare | SQL (small) |
|---|---|---|
| n_layer | 6 | 4 |
| n_head | 6 | 4 |
| n_embd | 384 | 256 |
| block_size | 256 | 128 |
| Expected val loss | <1.65 | <1.5 |
| Training iters | 5000 | 3000 |

---

## Decision Rules

- Causal mask: register as buffer (`register_buffer`), never as parameter.
- Weight tying: always. One line: `self.lm_head.weight = self.transformer.wte.weight`.
- Temperature for SQL generation: 0.2–0.5 (low, for syntactic correctness).
- Temperature for creative text: 0.8–1.2 (higher, for variety).
- Pre-LN vs Post-LN: use Pre-LN (modern default, more stable).

---

## Red Flags During Training

- Loss stuck at log(vocab_size): causal mask not applied or positional embeddings missing.
- Generated text is all one character repeated: softmax is computing on the wrong dimension.
- Val loss rises while train loss falls after step ~3000: overfitting — add dropout or reduce model size.
- `RuntimeError: weight is on cpu but input is on cuda`: forgot to move model to device; use `model.to(device)`.
