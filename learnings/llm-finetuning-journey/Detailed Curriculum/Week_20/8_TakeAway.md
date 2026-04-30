# Week 20 TakeAway — Pretraining Setup

**One-liner:** 8 layers × d_model=768 × 12 heads = ~57M params; BPE tokenizer; memory-mapped .bin files; sanity-check loss ≈ log(vocab_size) before training.

---

## Key Formulas

```python
# Parameter count per transformer block (approximate)
params_per_block = 12 * d_model ** 2  # attn: 4, ffn: 8

# Total params (excluding embedding)
total_params = n_layers * params_per_block

# Expected initial cross-entropy loss
import math
initial_loss = math.log(vocab_size)   # ~10.37 for vocab=32000

# Recommended config for 50M model
config = dict(n_layers=8, d_model=768, n_heads=12,
              vocab_size=32000, context_len=1024)
```

---

## Key Code Patterns

```python
# Weight-tied GPT (critical detail)
self.tok_emb = nn.Embedding(vocab_size, d_model)
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.tok_emb.weight  # tying

# Flash attention (one line, ~20-30% speedup)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Memory-mapped dataset (avoids loading all tokens into RAM)
data = np.memmap(path, dtype=np.uint16, mode='r')

# Cosine LR schedule with warmup
def get_lr(step, warmup=100, max_steps=10000, max_lr=3e-4, min_lr=3e-5):
    if step < warmup: return max_lr * step / warmup
    t = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))
```

---

## Decision Rules

- vocab_size > 65535 → must store tokens as `uint32`, not `uint16`
- Initial loss ≠ log(vocab_size) ± 1.0 → something is wrong with initialization or data pipeline
- Loss not decreasing after 100 steps → check targets are x[:, 1:] not x[:, :]
- For 50M model, context_len=1024 → ~1GB activation memory per batch; safe on A100
- Always run 200-step sanity check before committing to 24-hour training run

---

## Numbers to Remember

| Config | Value |
|---|---|
| n_layers | 8 |
| d_model | 768 |
| n_heads | 12 |
| context_len | 1024 |
| vocab_size | 32,000 |
| Total params | ~56–57M |
| Initial loss | ~10.37 |
| Target val loss after full training | < 3.5 |
| Chinchilla-optimal tokens | ~1.1B (20 × 56M) |

---

## Red Flags

- Initial loss < 8.0 → data leak or wrong initialization scale
- Loss drops to 2.0 in 100 steps → model is memorizing a tiny dataset
- CUDA OOM → reduce batch_size before reducing model size
- `count_params` returns 0 → model has no trainable parameters (check `requires_grad`)
- BPE tokenizer does not include `<|endoftext|>` → documents concatenate without separator → garbage data
