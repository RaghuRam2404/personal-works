# Week 3 — TakeAway

**This week in 15 words:** Conv = parameter sharing over space; backprop ninja; WaveNet = dilated causal hierarchy.

---

## Key Code Patterns

```python
# Standard conv block (for CIFAR-10 scale)
nn.Sequential(
    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
    nn.BatchNorm2d(c_out),
    nn.ReLU(inplace=True),
)

# Global average pooling (replaces Flatten for spatial inputs)
x = x.mean(dim=[2, 3])  # (N, C, H, W) -> (N, C)

# Cosine LR scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
# Call at end of each epoch:
scheduler.step()
```

```python
# Combined softmax + cross-entropy backward (backprop ninja)
dlogits = probs.clone()
dlogits[range(N), Yb] -= 1
dlogits /= N
```

```python
# Data augmentation for CIFAR-10
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
```

---

## Key Formulas

Conv output size:
```
H_out = floor((H_in + 2*P - K) / S + 1)
P=padding, K=kernel, S=stride
```

Dilated receptive field (geometric series):
```
RF = 2 * sum(d_i) + 1  for dilations d_i
RF = 2^(n_layers+1) - 1  for doublings (1,2,4,...,2^(n-1))
```

---

## Decision Rules

- **Kernel 1×1:** Changes channel count without touching spatial dims. Use for bottlenecks.
- **Stride 2 vs. MaxPool:** Strided conv is learnable; MaxPool is fixed. ResNets use strided conv.
- **BatchNorm2d vs BatchNorm1d:** 2D for conv feature maps; 1D for flat vectors.
- **Train acc >> Test acc (>10% gap):** Overfitting. Add augmentation + dropout.
- **Loss NaN on first batch:** Check for zero std after normalization (divisor = 0 + eps).

---

## Numbers to Remember

- CIFAR-10: 50K train, 10K test, 10 classes, 32×32×3
- 75% test accuracy = minimum acceptable for this week
- Conv2d(3, 64, 3, padding=1) on 32×32 → still 32×32 (same-size conv)
- WaveNet dilation: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 → RF = 1023 steps per block

---

## Red Flags During Training

- Test acc stalls at ~60% despite low train loss → overfitting, add augmentation.
- Loss NaN at step 0 → incorrect normalization or wrong loss function.
- Shape error in `nn.Flatten` → computed wrong spatial dims after pooling; check with print.
- Batch norm warning about small batch size → use `nn.LayerNorm` or increase batch.
- Confusion matrix all-same-column → model is predicting only one class; check class imbalance and loss.
