# Week 35 TakeAway — Hyperparameter Tuning for SFT/LoRA

**One-liner:** LR is king. Fix alpha=2r, target all linear layers, sweep LR first, use early stopping always.

---

## HP Priority Ranking

| Priority | Hyperparameter | Typical Range | Default |
|---|---|---|---|
| 1 | Learning rate | 1e-5 – 5e-4 | 2e-4 |
| 2 | LoRA rank | 8–64 | 16 |
| 3 | Epochs | 1–3 | 2 |
| 4 | Effective batch size | 8–128 | 16 |
| 5 | alpha | rank or 2×rank | 2×rank |

---

## Fixed Settings (Don't Tune)

```python
target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
optimizer = "paged_adamw_8bit"
scheduler = "cosine"
warmup_ratio = 0.05  # increase to 0.1 if loss spikes early
lora_dropout = 0  # with Unsloth; 0.05 with vanilla peft
```

---

## Early Stopping Pattern

```python
SFTConfig(
    eval_steps=50,
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
)
```

---

## LR vs. Batch Size Interaction

```
If doubling batch size, multiply LR by sqrt(2) = 1.41
Example: batch 16 → batch 32: LR 2e-4 → 2.8e-4
```

---

## Overfitting Signal

```
train loss: 2.1 → 0.4  (still falling)
eval  loss: 2.1 → 0.8 → 1.2  (rose in epoch 3)
                              ↑ stop here
```

---

## Numbers to Remember

- LR 2e-4 = standard LoRA LR; full SFT LR 1e-5 to 5e-5
- alpha = 2 × rank → scaling factor = 2.0 (don't tune independently)
- 1K examples: max 2 epochs. 5K examples: 2 epochs. 15K: 2 epochs with early stopping
- warmup_ratio = 0.05–0.10 (5–10% of total steps)

---

## Decision Rules

- If loss spike at step 10–100 → LR too high, reduce by 5x
- If train and eval loss both plateau high → LR too low
- If eval loss rises while train loss falls → overfitting, enable early stopping
- If sweep shows rank 16 == rank 32 → use rank 16 (fewer params, less risk)

---

## Red Flags

- LR > 5e-4 with LoRA → almost always unstable
- alpha >> 2×rank (e.g., alpha=128 with rank=8) → effective update too large
- No eval set → cannot detect overfitting; always keep 5–10% of data for eval
- 5+ epochs on 1K examples → guaranteed overfitting
