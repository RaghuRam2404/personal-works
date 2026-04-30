# Week 5 — TakeAway

**This week in 15 words:** AdamW + warmup/cosine + grad clip + AMP is the canonical modern training recipe.

---

## The Modern Training Recipe (use this as default)

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, betas=(0.9, 0.999),
    eps=1e-8, weight_decay=0.01
)
scaler = torch.cuda.amp.GradScaler()

for step, (x, y) in enumerate(loader):
    lr = get_lr(step, warmup_steps, total_steps, max_lr=3e-4)
    for pg in optimizer.param_groups: pg['lr'] = lr

    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                              # unscale BEFORE clip
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

## Cosine Warmup Schedule

```python
import math
def get_lr(step, warmup_steps, total_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

---

## Key Formulas

Adam update:
```
m = β1*m + (1-β1)*g
v = β2*v + (1-β2)*g²
θ -= lr * (m/(1-β1^t)) / (sqrt(v/(1-β2^t)) + ε)
```
AdamW weight decay: `θ -= lr * wd * θ` (separate from Adam step)

---

## Decision Rules

- **Adam or AdamW?** Always AdamW when using weight decay. They differ — see Curriculum.
- **SGD or Adam?** Training from scratch on vision → SGD+momentum. NLP/fine-tuning → AdamW.
- **Warmup length:** 5–10% of total steps for fine-tuning; ~2% for large-scale pre-training.
- **Grad clip threshold:** 1.0 for transformers. 5.0 for RNNs. If loss is unstable, try 0.5.
- **AMP on MPS (Mac):** Use `torch.autocast(device_type='mps')`. Skip `GradScaler`.

---

## Numbers to Remember

- AdamW defaults: `β1=0.9`, `β2=0.999`, `ε=1e-8`, `weight_decay=0.01`
- AdamW LR for LLM fine-tuning with LoRA: `2e-4` (Phase 4)
- AdamW LR for transformer pre-training (nanoGPT-size): `3e-4`
- Gradient clip max_norm: `1.0` (transformers), `5.0` (RNNs)
- AMP speedup on T4: ~1.5–2.5× for model sizes >10M parameters

---

## Red Flags During Training

- Loss spike then partial recovery → missing gradient clipping.
- LR trace in W&B is flat at zero → forgot to call `scheduler.step()` or LR schedule bug.
- AMP `scaler.get_scale()` dropping repeatedly → numerical instability; check for log(0).
- Weight decay making train loss worse → too high; try 0x, 0.001, 0.01 in that order.
- Loss curve is completely flat for all steps → LR may be 0 (wrong schedule) or optimizer bug.
