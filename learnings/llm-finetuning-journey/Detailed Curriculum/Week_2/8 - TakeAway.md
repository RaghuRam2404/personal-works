# Week 2 — TakeAway

**This week in 15 words:** MLP = Linear + activation; init and batch norm control whether gradients flow cleanly.

---

## Key Code Patterns

```python
# Standard MLP block (training-safe)
class MLPBlock(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.bn = nn.BatchNorm1d(d_out)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return F.relu(self.bn(self.fc(x)))  # Linear → BN → ReLU

# Dropout (training-only)
self.drop = nn.Dropout(p=0.3)
h = self.drop(F.relu(self.bn(self.fc(x))))
```

```python
# Embedding + flatten for language modeling
self.C = nn.Embedding(vocab_size, n_embd)
# In forward:
emb = self.C(x)                    # (B, context, n_embd)
emb = emb.view(emb.size(0), -1)   # (B, context * n_embd)
```

```python
# Activation selection
F.relu(x)                  # for CNNs, simple MLPs
F.gelu(x)                  # for transformers
torch.tanh(x)              # for RNNs, micrograd
F.sigmoid(x)               # for binary output only
```

---

## Key Formulas

Kaiming init std: `sqrt(2 / fan_in)` (for ReLU)

Xavier init std: `sqrt(2 / (fan_in + fan_out))` (for Tanh/Sigmoid)

Batch norm:
```
x_hat = (x - mean(x)) / sqrt(var(x) + 1e-5)
y = gamma * x_hat + beta
```

---

## Decision Rules

- **ReLU or GELU?** ReLU for CNNs/simple MLPs. GELU for transformer-style blocks.
- **Bad init symptom:** Pre-activation std far from 1.0 at layer 0. Use `kaiming_normal_` for ReLU.
- **Train loss ≈ Val loss, both high:** Underfitting. Increase capacity or LR.
- **Train loss << Val loss:** Overfitting. Add dropout, weight decay, or get more data.
- **Batch norm train/eval gap:** You forgot `model.eval()` before evaluation.
- **Batch norm + batch size 1:** Use `nn.LayerNorm` instead.

---

## Numbers to Remember

- Kaiming init std = `sqrt(2 / fan_in)` for ReLU
- Dropout rates: 0.1 (transformers), 0.3–0.5 (large MLP classifiers)
- Batch norm epsilon: `1e-5` (PyTorch default)
- Batch norm momentum: `0.1` (PyTorch default, means 10% of new batch stats each step)
- makemore MLP val loss target: < 2.2 nats on names

---

## Red Flags During Training

- Pre-activation means far from 0 and std far from 1 → initialization or missing batch norm.
- Loss suddenly stuck at `log(vocab_size)` → dying ReLUs or zero initialization.
- Train/eval accuracy gap > 10%: likely forgetting `model.eval()` with batch norm active.
- Embeddings not updating: check that `requires_grad=True` on `nn.Embedding` weight (it is by default — but check if you accidentally froze it).
