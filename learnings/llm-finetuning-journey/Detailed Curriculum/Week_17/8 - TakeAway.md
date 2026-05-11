# Week 17 TakeAway — Scaling Laws

**One-liner:** Chinchilla says scale tokens and parameters equally; inference needs justify over-training smaller models.

---

## Key Formulas

```python
# Compute cost approximation
C_flops = 6 * N_params * D_tokens

# Chinchilla optimal (Approach 3 constants)
N_opt = 6.8e-2 * (C ** 0.5)   # params
D_opt = 1.96   * (C ** 0.5)   # tokens

# Rough heuristic
D_optimal_tokens = 20 * N_params

# A100 effective throughput
C_available = MFU * 312e12 * hours * 3600  # FLOPs
# MFU ≈ 0.35 for well-tuned single-GPU training
```

---

## Decision Rules

- If you have a fixed compute budget → use Chinchilla to find N_opt and D_opt before touching a GPU
- If inference will be high-volume → train a smaller model on more tokens (inference-optimal)
- If your data is limited to D_max tokens → max useful model size is D_max / 20
- If training from scratch → never spend more than half your budget on parameters
- If fine-tuning a pretrained model → Chinchilla does not apply; focus on SFT dataset quality

---

## Numbers to Remember

| Fact | Value |
|---|---|
| Chinchilla heuristic | ~20 tokens per parameter |
| C ≈ 6ND approximation | holds for typical sequence lengths |
| A100 80GB peak FP16 | 312 TFLOP/s |
| Realistic MFU (single GPU) | 30–40% |
| Chinchilla-70B tokens | 1.4T (20 × 70B) |
| GPT-3 tokens | 300B (under-trained by 11×) |
| Llama-3-8B tokens | 15T (over-trained for inference) |

---

## Key Code Pattern

```python
def recommend_training_config(budget_usd, price_per_hour=1.50, mfu=0.35):
    a100_peak_flops = 312e12
    hours = budget_usd / price_per_hour
    C = mfu * a100_peak_flops * hours * 3600
    N_opt = 6.8e-2 * (C ** 0.5)
    D_opt = 1.96   * (C ** 0.5)
    return N_opt, D_opt, C

N, D, C = recommend_training_config(50)
print(f"N={N/1e6:.0f}M params, D={D/1e9:.1f}B tokens, C={C:.2e} FLOPs")
```

---

## Red Flags

- Computing Chinchilla for fine-tuning (it does not apply)
- Using peak FLOPs without MFU discount → overestimates available compute by 2.5×
- Applying the 20× heuristic to context length instead of token count
- Concluding that "bigger always wins" without checking data availability
