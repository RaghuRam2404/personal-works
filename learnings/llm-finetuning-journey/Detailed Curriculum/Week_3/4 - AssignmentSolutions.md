# Week 3 — Assignment Solutions

## Task 1 — Key Snippets (Backprop Ninja)

The two trickiest backward passes:

**Cross-entropy + softmax combined backward:**
```python
# Forward: logits -> probs -> loss
probs  = F.softmax(logits, dim=1)
loss   = -probs[range(N), Yb].log().mean()

# Backward (combined formula)
dlogits = probs.clone()
dlogits[range(N), Yb] -= 1
dlogits /= N
# Verify:
assert (dlogits - logits.grad).abs().max() < 1e-4
```

**Batch norm backward (full):**
```python
# Forward
xhat = (x - xmean) / (xvar + 1e-5).sqrt()
y    = gamma * xhat + beta

# Backward
N, D = x.shape
dbeta  = dout.sum(0)
dgamma = (xhat * dout).sum(0)
dxhat  = dout * gamma
dvar   = (dxhat * (x - xmean) * -0.5 * (xvar + 1e-5)**(-1.5)).sum(0)
dmean  = (dxhat * -1 / (xvar + 1e-5).sqrt()).sum(0) + dvar * (-2 * (x - xmean)).mean(0)
dx     = dxhat / (xvar + 1e-5).sqrt() + dvar * 2 * (x - xmean) / N + dmean / N
```

**Expected output:** All `assert` statements pass with tolerance `1e-4`.

**Common gotchas:**
- The batch norm backward requires the mean term to account for the fact that `xmean` also depends on `x` — forgetting this term gives a wrong gradient.
- The cross-entropy backward only subtracts 1 from the index corresponding to the correct class — a common off-by-one error.
- Division by `N` in the loss means the gradients must also be divided by `N` in the backward pass.
- Embedding backward: use `dC.scatter_add_` or a loop over the batch — do not use indexing assignment (it does not accumulate for repeated indices).

---

## Task 2 — Key Snippets (CIFAR-10 CNN)

```python
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # -> (32, 16, 16)
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # -> (64, 8, 8)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
```

**Expected performance:**
- After 20 epochs: test accuracy **76–80%** with data augmentation.
- Without augmentation (just normalization): ~72–74%.
- Training time on Colab T4: ~15–20 minutes for 20 epochs.

**Common gotchas:**
- `nn.BatchNorm2d` for spatial feature maps; `nn.BatchNorm1d` for flat vectors. Using the wrong one raises a shape error.
- Forgetting to call `scheduler.step()` at the end of each epoch — LR never decays.
- CIFAR-10 normalization: the constants `(0.4914, 0.4822, 0.4465)` and `(0.2023, 0.1994, 0.2010)` are per-channel mean and std of the training set. Not using them causes slower convergence.
- The confusion matrix: expect cat/dog and automobile/truck to be the most confused pairs.

---

## Task 3 — Conv Shape Answers

| Layer | Input | H_out formula | Output shape |
|---|---|---|---|
| Conv2d(3,64,3,stride=1,pad=1) | (1,3,32,32) | (32+2-3)/1+1=32 | (1,64,32,32) |
| Conv2d(3,64,3,stride=2,pad=1) | (1,3,32,32) | (32+2-3)/2+1=16 | (1,64,16,16) |
| Conv2d(3,64,5,stride=1,pad=0) | (1,3,32,32) | (32+0-5)/1+1=28 | (1,64,28,28) |
| Conv2d(3,64,1,stride=1,pad=0) | (1,3,32,32) | (32+0-1)/1+1=32 | (1,64,32,32) |
| Stack → GAP | (1,64,32,32)→pool→(1,64,16,16)→conv+pool→(1,64,8,8)→GAP | — | (1,64) |

**Common gotchas:**
- 1×1 convolutions (`kernel_size=1`) preserve spatial dimensions but change channel count — useful for dimensionality reduction (used in ResNets and MobileNets).
- Global average pooling (`nn.AdaptiveAvgPool2d(1)` or `x.mean(dim=[2,3])`) outputs `(N, C, 1, 1)` — flatten to `(N, C)` before the linear layer.

---

## How to Verify You Did It Right

1. **Backprop Ninja:** Every assertion passes. No `AssertionError`.
2. **CIFAR-10:** W&B shows val acc crossing 75% within 20 epochs. Confusion matrix shows sensible patterns (cat/dog confused; airplane/ship less so).
3. **Conv shapes:** Each of your hand-computed answers matches the Python `output.shape` print.
