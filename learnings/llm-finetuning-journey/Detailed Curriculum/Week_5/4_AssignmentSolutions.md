# Week 5 — Assignment Solutions

## Task 1 — Key Snippets (LR Schedule)

```python
import math

def get_lr(step, warmup_steps, total_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# Integrate into training loop
for step in range(total_steps):
    lr = get_lr(step, warmup_steps=200, total_steps=2000, max_lr=3e-4)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # ... rest of training step
```

**Expected plot shape:** LR rises linearly from 0 to 3e-4 over the first 200 steps, then follows a smooth cosine curve down to 0 by step 2000. The transition at step 200 should be smooth (continuity, though not differentiability).

**Common gotchas:**
- At step=0 with the warmup formula, LR = `max_lr * 0 / 200 = 0`. Correct.
- At step=200 with cosine formula, `progress = 0`, `cos(0) = 1`, so LR = `min_lr + 0.5*(max_lr-min_lr)*2 = max_lr`. Continuity preserved.
- Do not call `scheduler.step()` if you are using a custom LR function — that would double-update the LR.

---

## Task 2 — Key Snippets (AMP Training)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for step, (x, y) in enumerate(train_loader):
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()

    with autocast():
        logits = model(x)
        loss   = criterion(logits, y)

    scaler.scale(loss).backward()

    # Unscale before computing grad norm
    scaler.unscale_(optimizer)
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    lr = get_lr(step, warmup_steps, total_steps, max_lr)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
```

**Expected comparison results:**

| Run | Final Val Acc | Convergence (epochs to 70%) |
|---|---|---|
| `baseline` (SGD+const LR) | ~74% | ~12 epochs |
| `adamw-cosine` | ~77% | ~8 epochs |
| `adamw-cosine-amp` | ~77% | ~8 epochs |

AMP typically gives 1.5–2.5× faster epoch time on T4 for this model size.

**Common gotchas:**
- `scaler.unscale_(optimizer)` must be called before `clip_grad_norm_` — otherwise you are clipping scaled gradients, not actual gradients.
- If `scaler.step()` skips (because gradients had Inf/NaN), the optimizer does not update. This is logged internally by `scaler`. The LR should still be updated.
- W&B report: use "Group by" → run name; then select "val/acc" on Y-axis and "epoch" on X-axis for the comparison.

---

## Task 3 — Key Snippets (Adam from Scratch)

```python
class AdamOptimizer:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr; self.b1, self.b2 = betas
        self.eps = eps; self.wd = weight_decay
        self.state = [{'m': torch.zeros_like(p.data),
                       'v': torch.zeros_like(p.data),
                       't': 0} for p in self.params]

    def step(self):
        for p, s in zip(self.params, self.state):
            if p.grad is None: continue
            g = p.grad.data
            s['t'] += 1
            s['m'] = self.b1 * s['m'] + (1 - self.b1) * g
            s['v'] = self.b2 * s['v'] + (1 - self.b2) * g * g
            m_hat = s['m'] / (1 - self.b1 ** s['t'])
            v_hat = s['v'] / (1 - self.b2 ** s['t'])
            # AdamW: decoupled weight decay
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
            p.data -= self.lr * self.wd * p.data    # weight decay applied separately

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
```

**Expected output:** Loss curves for `AdamOptimizer` and `torch.optim.AdamW` should be nearly identical on `make_moons` (within numerical precision of the bias correction implementation).

**Common gotchas:**
- The bias correction uses `1 - beta^t`, not `1 - beta^step`. `t` is per-parameter step count — track it in the state dict.
- Weight decay in the Adam update order matters: apply the Adam step first, then the weight decay. The two terms are independent in AdamW.

---

## How to Verify You Did It Right

1. **Task 1:** LR at step 0 = 0. At step 200 = 3e-4. At step 2000 = 0 (or `min_lr`). Plot matches a smooth bell-then-cosine shape.
2. **Task 2:** `adamw-cosine` run achieves higher val acc than `baseline` within 20 epochs. AMP run matches `adamw-cosine` accuracy and is visibly faster (check epoch time in W&B).
3. **Task 3:** Loss curves for your `AdamOptimizer` and `torch.optim.AdamW` match within 1% at every logging step.
