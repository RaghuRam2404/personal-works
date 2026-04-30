# Week 8 — TakeAway

**This week in 15 words:** Phase 1 gate: write the training loop from memory, diagnose loss curves, advance honestly.

---

## The Complete Modern Training Loop (memorize this)

```python
# Everything you need for any Phase 1–2 training run
import torch, math
from torch.cuda.amp import autocast, GradScaler

model     = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
scaler    = GradScaler(enabled=device=='cuda')

def get_lr(step, warmup, total, max_lr, min_lr=0):
    if step < warmup: return max_lr * step / warmup
    t = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))

for step in range(total_steps):
    # 1. LR schedule
    lr = get_lr(step, warmup_steps, total_steps, max_lr)
    for pg in optimizer.param_groups: pg['lr'] = lr

    # 2. Training step
    model.train()
    x, y = get_batch('train')
    optimizer.zero_grad()
    with autocast(enabled=device=='cuda'):
        logits, loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # 3. Validation (every N steps)
    if step % val_interval == 0:
        model.eval()
        with torch.no_grad():
            _, val_loss = model(xv, yv)
        print(f"step {step}: train={loss:.4f}, val={val_loss:.4f}")
```

---

## Phase 1 Knowledge Checkpoints

| Topic | Week | Self-test |
|---|---|---|
| `zero_grad` position | 1 | Can you write the 5-step loop? |
| Kaiming init | 2 | Can you derive `sqrt(2/fan_in)` on paper? |
| Conv output formula | 3 | `(H + 2P - K)/S + 1` — can you apply it? |
| LSTM gates | 4 | Can you write `c_t = f_t*c_{t-1} + i_t*g_t`? |
| AdamW vs Adam | 5 | Can you explain the weight decay difference? |
| BPE merge loop | 6 | Can you write `get_stats` + `merge` from scratch? |
| Labels = -100 | 7 | Can you explain why padding must be masked? |

---

## Loss Curve Diagnosis (Phase 1 Summary)

| Curve pattern | Diagnosis | Fix |
|---|---|---|
| Train ↓, Val ↑ (diverging) | Overfitting | Dropout, weight decay, more data |
| Both high, both flat | Underfitting | Larger model, higher LR, train longer |
| Loss spikes then recovers | Gradient explosion | Add `clip_grad_norm_` |
| Loss oscillates | LR too high | Reduce by 5–10× |
| Loss very slow | LR too low | Increase by 3–5× |
| Loss → NaN step 1 | Numerical instability | Check log(0), normalization |

---

## Phase Gate Criteria (summary)

- [ ] Training loop from memory in < 10 minutes
- [ ] Loss curve diagnosis: can identify all 6 patterns above
- [ ] Backprop by hand: 2-layer network, chain rule, `dL/dW`
- [ ] GitHub: commits for all 7 prior weeks
- [ ] HuggingFace: 1+ uploaded artifact with accessible URL

**If any criterion fails: repeat the relevant week. Do not advance.**
