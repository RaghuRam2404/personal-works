# Week 36 — DoRA, RSLoRA, LoftQ, and LoRA Variants

## Learning Objectives

By the end of this week, you will be able to:

- Explain the key differences between DoRA, RSLoRA, and LoftQ and the problem each solves
- Configure DoRA and RSLoRA via peft's `LoraConfig`
- Run a DoRA vs. LoRA comparison on your SQL dataset and interpret the results
- Decide when each variant is preferable over standard LoRA
- Understand the weight decomposition (magnitude + direction) in DoRA

---

## Concepts

### 1. Why LoRA Variants Exist

Standard LoRA (Week 30) is powerful but has known limitations:
- The rank determines capacity, but the scaling is naive (uniform `alpha/r` across all layers)
- At high ranks, LoRA may be numerically unstable or converge poorly
- When the base model is already quantized (QLoRA), the adapter may not optimally compensate for quantization artifacts

DoRA, RSLoRA, and LoftQ each address one of these limitations.

### 2. DoRA: Weight-Decomposed Low-Rank Adaptation

DoRA (Dora: Weight-Decomposed Low-Rank Adaptation, Liu et al. 2024, arxiv 2402.09353) decomposes the pretrained weight matrix into its magnitude and direction components:

```
W = m × (V / ||V||)   where m is magnitude (scalar), V is direction (unit vector)
```

DoRA then adapts:
- **Direction:** via a standard LoRA matrix BA (low-rank update)
- **Magnitude:** via a learned scalar m per output dimension

This decomposition is inspired by the observation that during full fine-tuning, models tend to update both the magnitude and direction of weight vectors, but LoRA primarily adapts direction (since the LoRA update BA is in the column space of A). DoRA explicitly separates and trains both, which empirically allows it to behave more like full fine-tuning.

**In practice:** DoRA uses slightly more parameters than LoRA (adds a magnitude vector of size d_out per adapted layer) but consistently outperforms LoRA of the same rank on several benchmarks. The improvement is most notable on tasks requiring fine-grained knowledge transfer.

**peft configuration:**
```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[...],
    use_dora=True,  # Enable DoRA
    ...
)
```

Unsloth also supports DoRA: `FastLanguageModel.get_peft_model(..., use_dora=True)`.

### 3. RSLoRA: Rank-Stabilized LoRA

RSLoRA (Kalajdzic et al. 2023, arxiv 2312.03732) addresses a numerical stability issue in LoRA scaling. The original scaling factor is `alpha / r`. RSLoRA proposes using `alpha / sqrt(r)` instead.

**Why:** At high ranks (r = 64, 128, 256), the original `alpha/r` scaling shrinks the LoRA contribution rapidly. With `alpha=64` and `r=64`, scaling = 1.0 — the LoRA update has the same scale as the pretrained output. With `alpha=64` and `r=256`, scaling = 0.25 — the LoRA contribution is heavily attenuated even if the adapter has learned something useful.

RSLoRA's `alpha / sqrt(r)` scaling is rank-stabilized: doubling the rank no longer halves the effective contribution. This makes RSLoRA particularly useful when sweeping large ranks (r ≥ 64).

**peft configuration:**
```python
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    use_rslora=True,  # scaling = alpha / sqrt(r) instead of alpha / r
    ...
)
```

When should you use RSLoRA? When you need high rank (≥ 64) and find that standard LoRA underperforms at those ranks — often because the `alpha/r` scaling makes the adapter contribution negligible.

### 4. LoftQ: LoRA-Fine-Tuning-Aware Quantization

LoftQ (Li et al. 2023, arxiv 2310.08659) addresses a specific problem in QLoRA: the quantization step introduces errors into the base model weights, and the LoRA adapter must learn to compensate for those errors. Standard QLoRA initializes B=0 (adapter starts as no-op), meaning the model starts training from the quantized (lossy) base without any initial correction.

LoftQ instead initializes the LoRA adapter to approximately compensate for the quantization error from the start:

```
W_original ≈ W_quantized + B_init × A_init

Solved as: B_init × A_init = W_original - W_quantized
```

This is solved via SVD (truncated to rank r) applied to the quantization error matrix. The result: at step 0, the LoftQ model starts closer to the original BF16 performance than standard QLoRA does.

**When to use:** When you observe that your QLoRA model starts with very high loss (significantly worse than the BF16 base) and you suspect quantization error is the bottleneck. For most practical setups, standard QLoRA initialization (B=0) works fine, and LoftQ's advantage diminishes after the first few hundred training steps.

**peft configuration:** LoftQ initialization is applied as a pre-processing step:
```python
from peft import LoftQConfig

loftq_config = LoftQConfig(loftq_bits=4, loftq_iter=1)
lora_config = LoraConfig(
    r=16,
    init_lora_weights="loftq",
    loftq_config=loftq_config,
    ...
)
```

### 5. Comparison: LoRA vs. DoRA vs. RSLoRA vs. LoftQ

| Method | Key change | When to use | Overhead |
|---|---|---|---|
| LoRA | Baseline adapter | Always applicable | None |
| DoRA | Magnitude + direction decomposition | Dense tasks, better convergence | Slight memory + compute |
| RSLoRA | alpha/sqrt(r) scaling | High-rank training (r ≥ 64) | None |
| LoftQ | Compensate for quantization error at init | QLoRA with significant quality loss | Slight init overhead |

For your SQL task: **DoRA is worth trying** if your LoRA eval loss plateaus and you suspect the standard LoRA is not learning fine-grained patterns. **RSLoRA** is worth enabling if you use rank ≥ 64. **LoftQ** is only worth the effort if standard QLoRA is degraded on your specific base model.

---

## Connections

**Builds on:** Week 30–31 LoRA fundamentals. This week extends those concepts with three algorithmic improvements.

**Needed for:** Week 38 (choose your adapter type for the 15K sprint based on this week's experiments).

---

## Common Misconceptions / Pitfalls

- **"DoRA always outperforms LoRA."** Not always. On simple tasks or small datasets, the advantage may not materialize. Empirically compare on your specific task.
- **"RSLoRA changes the optimal learning rate."** Yes — because scaling changes from `alpha/r` to `alpha/sqrt(r)`, the effective update magnitude changes. You may need to re-tune LR slightly when switching.
- **"LoftQ eliminates quantization error."** No — it reduces the initialization gap, but quantization error remains throughout training. The adapter must still learn despite the noisy base.
- **"I should always use DoRA for best performance."** DoRA adds a marginal compute overhead (~5–10%). For large-scale production training where compute cost is a concern, the overhead matters. For Colab-scale experiments, it is negligible.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read DoRA paper (arxiv 2402.09353) — sections 1–4 | 1.5h |
| Read RSLoRA paper (arxiv 2312.03732) — sections 1–3 | 1h |
| Skim LoftQ paper (arxiv 2310.08659) — sections 1–3 | 1h |
| Run DoRA vs. LoRA experiment | 2h |
| Write comparison report | 1h |
| Commit to GitHub | 30m |
