# Week 1 — TakeAway

**This week in 15 words:** PyTorch builds a dynamic compute graph; `backward()` chains gradients; the training loop has 5 steps.

---

## The Canonical Training Loop

```python
model.train()
for x, y in train_loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()          # 1. Zero gradients — NEVER skip
    y_hat = model(x)               # 2. Forward
    loss = criterion(y_hat, y)     # 3. Loss (must be scalar)
    loss.backward()                # 4. Backward
    optimizer.step()               # 5. Update weights
```

## Validation Loop

```python
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        val_loss += criterion(y_hat, y).item()
model.train()
```

## Micrograd Core Pattern

```python
# Every op stores a _backward that accumulates gradients
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    def _backward():
        self.grad  += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward
    return out
```

---

## Key Formulas

Chain rule: `dL/dx = dL/dy * dy/dx`

For `y = x @ W`:
- `dL/dW = x.T @ dL/dy`
- `dL/dx = dL/dy @ W.T`

---

## Decision Rules

- If loss is **flat from step 1**: check `zero_grad()` is present; check LR is not 0.
- If loss **spikes then recovers**: bad batch, or LR too high.
- If loss **goes to NaN**: numerical instability — check for log(0), use `_with_logits` loss variants.
- If GPU memory grows each step: you are storing tensors (not `.item()`) in a list.
- If `view` raises `RuntimeError`: call `.contiguous()` first, or use `reshape`.

---

## Numbers to Remember

- Default PyTorch dtype: `float32`
- `torch.randn` — mean 0, std 1
- Adam default LR: `1e-3`; safe starting LR for most small experiments.
- `loss.item()` cost: O(1), always use it for logging.

---

## Red Flags During Training

- Loss does not move at all for 10+ steps → missing `zero_grad` or LR = 0.
- Loss is exactly 0.0 → your labels and predictions are aligned by accident, or loss is computed on the wrong values.
- Loss goes to NaN instantly → exploding gradients or log(0); reduce LR first, then check data.
- Memory grows each epoch → graph references leaking into Python lists.
