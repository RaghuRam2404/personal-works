# Week 30 Assignment Solutions

## Task 1 — LoraLinear: Key Implementation

```python
import torch
import torch.nn as nn
import math

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        base_out = x @ self.weight.T
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling
    
    def merge(self):
        with torch.no_grad():
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.lora_A.zero_()
            self.lora_B.zero_()
```

**Test snippet:**
```python
layer = LoraLinear(64, 32, rank=4, alpha=8)
x = torch.randn(2, 64)
assert torch.allclose(layer(x), x @ layer.weight.T)  # B=0 so LoRA = 0
assert not layer.weight.requires_grad
assert layer.lora_A.requires_grad
print("All tests passed.")
```

---

## Task 2 — Parameter Count: Expected Answers

For Qwen2.5-7B with d_model = 4096, d_ffn = 11008, 28 layers, rank = 16:

```
q_proj: 16 × (4096 + 4096) = 131,072
k_proj: 16 × (4096 + 4096) = 131,072
v_proj: 16 × (4096 + 4096) = 131,072
gate_proj: 16 × (4096 + 11008) = 241,664

Per layer (4 modules): 634,880
Across 28 layers: 17,776,640 ≈ 17.8M parameters

Qwen2.5-7B total ≈ 7.24B parameters
LoRA ratio: 17.8M / 7,240M ≈ 0.25%
```

Note: if you also include `o_proj`, `up_proj`, `down_proj`, the count roughly doubles to ~0.5%. This is still less than 1% of total parameters.

---

## Task 3 — apply_lora: Key Snippet

```python
def apply_lora(model, rank=8, alpha=16, target_modules=("c_attn", "c_proj")):
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            parent = get_parent_module(model, name)
            attr = name.split(".")[-1]
            lora_layer = LoraLinear(
                module.in_features, module.out_features,
                rank=rank, alpha=alpha
            )
            lora_layer.weight.data = module.weight.data.clone()
            setattr(parent, attr, lora_layer)
    
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
    return model
```

**Expected output:**
```
Total params:     124,439,808
Trainable params:     294,912
Trainable %:          0.24%
Step 0 loss: ~2.9
Step 100 loss: ~2.1–2.5
```

---

## Common Gotchas

- **nanoGPT's `c_attn` combines Q, K, V**: `out_features = 3 * n_embd`. Your `LoraLinear` handles this automatically since it just stores W as (out_features, in_features) — no special handling needed.
- **Merge not zero-sum**: After `merge()`, the LoRA path should output zero. Add `assert torch.allclose(layer.lora_B @ layer.lora_A, torch.zeros_like(layer.lora_B @ layer.lora_A))` to verify.
- **`requires_grad` check order**: You must freeze all params THEN un-freeze LoRA — not the other way around. If you un-freeze LoRA first, then freeze all, LoRA gets frozen too.
- **Scaling factor**: `alpha/rank` is applied at inference. If you forget this, the LoRA update is unscaled — training still works but the hyperparameter semantics change.

---

## How to Verify You Did It Right

- `python test_lora.py` prints "All tests passed." with no assertions failing
- Trainable parameter % is between 0.1% and 1% for rank-8 LoRA on nanoGPT
- Loss decreases from step 0 to step 100
- After `merge()`, `layer(x_test)` and `merged_layer(x_test)` are numerically identical within `atol=1e-6`
- Your `week30_derivation.md` shows LoRA on Qwen2.5-7B at rank 16 = 0.2–0.5% of total params
