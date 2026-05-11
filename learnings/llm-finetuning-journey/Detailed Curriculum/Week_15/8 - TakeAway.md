# Week 15 TakeAway — From-Scratch GPT-2 124M Reproduction

**This week in 15 words:** Reproduce GPT-2 124M with mixed precision, Flash Attention, and gradient accumulation — for real.

---

## Key Production Training Patterns

```python
# Mixed precision (forward only)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(x, y)

# Gradient accumulation
optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    x, y = loader.next_batch()
    with torch.autocast(...):
        logits, loss = model(x, y)
    (loss / grad_accum_steps).backward()  # normalize!

# After all micro-steps:
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()

# Flash Attention (drop-in)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Cosine LR with warmup
def get_lr(step):
    if step < warmup: return max_lr * step / warmup
    ratio = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))

# AdamW: weight decay on 2D params only
decay = [p for n,p in model.named_parameters() if p.dim() >= 2]
nodecay = [p for n,p in model.named_parameters() if p.dim() < 2]
optimizer = AdamW([{'params': decay, 'weight_decay': 0.1},
                   {'params': nodecay, 'weight_decay': 0.0}], ...)
```

---

## GPT-2 124M Numbers to Remember

| Config | Value |
|---|---|
| n_layer | 12 |
| n_head | 12 |
| n_embd | 768 |
| vocab_size | 50257 |
| block_size | 1024 |
| total params | 124M |
| target val loss | 3.11 (OpenWebText) |
| target HellaSwag | 29.5% |
| max_lr | 6e-4 |
| min_lr | 6e-5 |
| warmup_steps | 715 |
| total_batch | 524288 tokens |
| training tokens | ~10B (full run) |

---

## Decision Rules

- Step 0 loss ≠ 10.82: model is not randomly initialized or your data pipeline is wrong.
- Grad norm consistently at 1.0 clip: LR too high; reduce max_lr by 20–30%.
- OOM on A100 at B=16, T=1024: reduce B to 8 and double grad_accum_steps.
- bfloat16 not available (older GPU): switch to FP16 with `torch.cuda.amp.GradScaler`.
- Always normalize loss before `.backward()` when using gradient accumulation.

---

## Red Flags During Training

- Loss goes to NaN at step 1 → forgot to divide loss by grad_accum_steps.
- Loss stuck at 10.82 after 1000 steps → LR is 0 or optimizer not called (forgot optimizer.step()).
- val_loss >> train_loss by a large margin → data leakage in training set (train data overlaps val).
- HellaSwag stays at 25% throughout → evaluation bug; model is not seeing the correct inputs.
- tokens/second < 100k on A100 → autocast not active or model is on CPU.
