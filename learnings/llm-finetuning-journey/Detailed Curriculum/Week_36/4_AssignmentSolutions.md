# Week 36 Assignment Solutions

## Task 1 — DoRA Configuration: Key Snippet

```python
# Standard LoRA (Run A)
lora_config_standard = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
    use_dora=False,
    task_type="CAUSAL_LM",
)

# DoRA (Run B)
lora_config_dora = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
    use_dora=True,   # the only change
    task_type="CAUSAL_LM",
)

# In Unsloth:
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=[...],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    use_dora=True,   # DoRA enabled
)
```

---

## Expected Results

| Variant | Trainable Params | Train Loss | Eval Loss | Held-out EM |
|---|---|---|---|---|
| LoRA baseline | ~42M | 0.7–1.0 | 1.1–1.4 | 35–50% |
| DoRA | ~43M (+magnitude) | 0.65–0.95 | 1.05–1.35 | 37–53% |
| RSLoRA rank 64 | ~160M | 0.6–0.9 | 1.0–1.3 | 38–55% |
| Std LoRA rank 64 | ~160M | 0.5–0.8 | 1.1–1.5 (may overfit) | 33–48% |

Typical finding: DoRA provides a small but consistent improvement (0.05–0.1 eval loss) on SQL tasks. RSLoRA at rank 64 with proper scaling is more stable than standard LoRA at rank 64.

---

## Task 2 — RSLoRA Scaling Math

With rank 64, alpha 32, `use_rslora=True`:
- Scaling = `alpha / sqrt(rank)` = `32 / sqrt(64)` = `32 / 8` = 4.0

With rank 64, alpha 128, standard LoRA:
- Scaling = `alpha / rank` = `128 / 64` = 2.0

RSLoRA's scaling is higher (4.0 vs. 2.0), which effectively doubles the LoRA contribution at the same training step. If you observe that standard LoRA rank 64 has lower eval loss than RSLoRA rank 64 despite RSLoRA's larger contribution, reduce the LR for RSLoRA slightly (try 1e-4 instead of 2e-4).

---

## Task 3 — LoftQ Key Points

The quantization error matrix E = W_fp16 - W_nf4 has some rank (often large). Truncating E's SVD to rank r gives the best rank-r approximation of the quantization error, which initializes (B_init × A_init ≈ E_r). After initialization, the model starts at a state much closer to the FP16 base than standard QLoRA (where B=0 means the entire quantization error is uncompensated).

For your SQL task: standard QLoRA on Qwen2.5-Coder-7B with NF4 4-bit shows minimal quality degradation from the BF16 base (our Week 32 measurements showed <5% quality drop). LoftQ's benefit is most significant when quantization error is large — typically for smaller models (1.5B) where the relative quantization error per parameter is higher. For 7B models with double quantization (which reduces grouping artifacts), LoftQ is usually not worth the additional complexity.

---

## Common Gotchas

- **DoRA requires peft >= 0.9.0**: `pip install --upgrade peft` before testing DoRA.
- **Trainable param count with DoRA**: slightly higher than standard LoRA (add d_out per adapted layer). `model.print_trainable_parameters()` will show the exact count.
- **RSLoRA and LR interaction**: RSLoRA's higher effective scaling with high ranks means the existing LR of 2e-4 may be too large → loss spike. Try 1e-4 if you see instability.
- **LoftQ svd overhead**: LoftQ initialization adds a one-time SVD computation on the quantization error — takes 30–60 seconds for a 7B model.

---

## How to Verify You Did It Right

- DoRA run shows slightly lower eval loss than standard LoRA (within 0.05–0.15 loss units)
- RSLoRA at rank 64 is more stable than standard LoRA at rank 64 (no early loss spike)
- `week36_loftq_analysis.md` correctly explains the SVD-based initialization
- Your Week 38 recommendation is backed by empirical evidence from this week's runs
