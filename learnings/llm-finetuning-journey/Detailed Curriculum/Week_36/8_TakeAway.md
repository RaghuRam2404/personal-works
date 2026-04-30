# Week 36 TakeAway — DoRA, RSLoRA, LoftQ

**One-liner:** DoRA = magnitude+direction decomposition (usually +3–5% quality); RSLoRA = stable scaling for high ranks; LoftQ = compensate quantization error at init.

---

## Quick Comparison

| Method | Config flag | When to use | Overhead |
|---|---|---|---|
| Standard LoRA | default | Always, baseline | None |
| DoRA | `use_dora=True` | SFT tasks, want best quality | +5–10% compute |
| RSLoRA | `use_rslora=True` | Rank ≥ 64 | None |
| LoftQ | `init_lora_weights="loftq"` | Large quantization error at init | +30s init SVD |

---

## peft Configurations

```python
# DoRA
LoraConfig(r=16, lora_alpha=32, use_dora=True, ...)

# RSLoRA
LoraConfig(r=64, lora_alpha=32, use_rslora=True, ...)
# scaling = 32 / sqrt(64) = 4.0

# LoftQ
from peft import LoftQConfig
loftq_config = LoftQConfig(loftq_bits=4, loftq_iter=1)
LoraConfig(r=16, init_lora_weights="loftq", loftq_config=loftq_config, ...)
```

---

## DoRA Forward Pass (Conceptual)

```
W = m × (V / ||V||)    # decompose pretrained weight
ΔW = BA                # LoRA low-rank direction update
output = (m + Δm) × ((V + ΔV_from_BA) / ||V + ΔV||) × x
```
Both magnitude m and direction V are adapted.

---

## RSLoRA Scaling vs. Standard LoRA

| Rank | Standard scaling | RSLoRA scaling (alpha=16) |
|---|---|---|
| 8 | 16/8 = 2.0 | 16/√8 = 5.66 |
| 16 | 16/16 = 1.0 | 16/√16 = 4.0 |
| 64 | 16/64 = 0.25 | 16/√64 = 2.0 |

RSLoRA prevents the adapter from becoming ineffective at large ranks.

---

## Decision Rules

- For rank 16 SFT: DoRA first, then compare to standard LoRA empirically
- For rank ≥ 64: always use RSLoRA (standard LoRA scaling collapses)
- For QLoRA with large quantization error (starting loss >> base model loss): try LoftQ
- If DoRA adds 0.02 eval loss improvement: use it (overhead is minimal on A100)

---

## Red Flags

- Forgetting to upgrade peft before using DoRA → "unexpected keyword argument `use_dora`"
- RSLoRA with high alpha at high rank → effective scaling too large → loss spike → reduce LR
- LoftQ SVD on huge model → takes minutes → run once, cache the initialized weights
