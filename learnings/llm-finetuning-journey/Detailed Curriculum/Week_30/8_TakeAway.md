# Week 30 TakeAway — LoRA Math and Implementation

**One-liner:** LoRA = frozen W + trainable low-rank delta_W = BA, using r×(d_in+d_out) params instead of d_in×d_out.

---

## The Core Formula

```
Forward pass:
  y = W x + (B A x) * (alpha / r)

Where:
  W ∈ R^(d_out × d_in)  — frozen pretrained weight
  A ∈ R^(r × d_in)      — initialized Kaiming uniform
  B ∈ R^(d_out × r)     — initialized ZERO
  alpha / r              — scaling factor

Parameter count (trainable): r × (d_in + d_out)
```

---

## Minimal LoraLinear

```python
class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.scaling = alpha / rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return x @ self.weight.T + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merge(self):
        with torch.no_grad():
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.lora_A.zero_(); self.lora_B.zero_()
```

---

## Freeze / Unfreeze Pattern

```python
for p in model.parameters():
    p.requires_grad = False  # freeze everything first
for name, p in model.named_parameters():
    if "lora_" in name:
        p.requires_grad = True  # then unfreeze LoRA only
```

---

## Numbers to Remember

- Rank 16 LoRA on all linear layers of Qwen2.5-7B ≈ 0.25–0.5% of total params
- Full SFT on 7B: ~84GB. LoRA SFT on 7B: ~18GB (A and B are tiny)
- `alpha = 2 × rank` is a common default (scaling = 2.0)
- `alpha = rank` sets scaling = 1.0, simpler reasoning about magnitudes

---

## Decision Rules

- If d_in = d_out = 4096, rank 8 → 65,536 adapter params per layer
- If task needs high expressiveness → increase rank (16, 32, 64)
- If dataset is small (< 5K) → lower rank (8) to prevent overfitting
- At inference → always merge LoRA into W to remove overhead

---

## Red Flags

- `lora_B` is not zero at step 0 → initialization bug
- `weight` gradient is non-zero → forgot `requires_grad=False`
- Loss starts at 8+ (should start near pretrained loss) → A or B initialized randomly together
- After merge, outputs differ → scaling factor applied incorrectly in merge
