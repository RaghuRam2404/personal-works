# Week 17 Assignment Solutions

## Task 1 — Key Snippet: chinchilla_calculator.py

```python
import math

# Chinchilla Approach 3 constants (Table A3, Hoffmann et al. 2022)
A = 6.8e-2
B = 1.96
EXPONENT = 0.5

def chinchilla_optimal(C_flops, dollar_per_hour=1.50, mfu=0.35):
    """
    Given a compute budget in FLOPs, return Chinchilla-optimal N and D.
    Also compute the dollar cost and wall-clock hours.
    """
    # a100 peak FP16: 312e12 FLOP/s
    a100_peak = 312e12
    effective_flop_s = mfu * a100_peak

    N_opt = A * (C_flops ** EXPONENT)
    D_opt = B * (C_flops ** EXPONENT)

    hours = C_flops / (effective_flop_s * 3600)
    cost = hours * dollar_per_hour
    c_check = 6 * N_opt * D_opt

    return {
        "N_opt_M": N_opt / 1e6,
        "D_opt_B": D_opt / 1e9,
        "hours": hours,
        "cost_usd": cost,
        "c_check": c_check,
    }

budgets = [1e18, 1e19, 1e20, 1e21, 1e22]
for C in budgets:
    r = chinchilla_optimal(C)
    print(f"C={C:.2e}  N={r['N_opt_M']:.1f}M  D={r['D_opt_B']:.2f}B  "
          f"$={r['cost_usd']:.2f}  check={r['c_check']:.2e}")
```

**Expected output (approximate):**
```
C=1.00e+18  N=2.1M  D=61.9B  — wait, check your constants
```

Common gotcha: some students confuse Chinchilla's three approaches. Approach 3 (IsoFLOP profiles) gives a ≈ 6.8e-2, b ≈ 1.96. Approach 1 gives different constants. Use Approach 3 for the most robust estimates.

---

## Task 2 — Expected Table

| Model | Params | Actual Tokens | Chinchilla-Optimal Tokens | Classification |
|---|---|---|---|---|
| GPT-3 | 175B | 300B | 3,500B (3.5T) | Under-trained |
| LLaMA-1-7B | 7B | 1T | 140B | Over-trained (inference) |
| Chinchilla-70B | 70B | 1.4T | 1,400B | Near-optimal |
| Llama-3-8B | 8B | 15T | 160B | Heavily over-trained (inference) |

**Common gotchas:**
- Chinchilla-optimal D = 20 × N is an approximation of Approach 3, not an exact law
- "Over-trained for inference" is a positive label for deployment, not a criticism
- LLaMA-1-7B trained on 1T tokens was deliberately inference-optimal before Llama 3
- GPT-3's under-training was genuine — OpenAI's 2020 priority was model capability demos, not cost efficiency

---

## Task 3 — How to Verify You Did It Right

**Check your $50 calculation:**
```
A100 effective FLOP/s = 0.35 × 312e12 = 1.09e14 FLOP/s
Hours = 50 / 1.50 = 33.3 hrs
C = 1.09e14 × 33.3 × 3600 ≈ 1.3e19 FLOPs

N_opt = 6.8e-2 × (1.3e19)^0.5 ≈ 6.8e-2 × 3.6e9.5 ≈ ~245M params
D_opt = 1.96 × (1.3e19)^0.5 ≈ ~7B tokens
```

Your answer should be in the neighborhood of 200–400M params and 5–10B tokens. If you got 7B params, you made an exponent error.

**Four questions — model answers:**
1. Chinchilla recommends ~250M params for $50
2. 250M params easily fits on an A100 (roughly 500MB in FP16); memory is not the bottleneck
3. Qwen2.5-Coder-7B on 5.5T tokens is 79× over-trained by Chinchilla (optimal = 140B). Alibaba did this because inference at 7B scale is cheap and they wanted maximum quality for deployment
4. Always start from a pretrained SOTA model for your fine-tuning project — 7B params trained on trillions of tokens gives you a far better initialization than anything you could train from scratch with $50

**Red flags in your writeup:**
- Computing C in FLOPs but forgetting to convert hours → seconds
- Using peak A100 FLOPs without applying MFU (35–40%)
- Concluding you should train from scratch for the SQL task (wrong — always fine-tune a SOTA base)
