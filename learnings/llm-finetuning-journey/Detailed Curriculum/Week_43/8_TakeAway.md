# Week 43 TakeAway — DPO

**One-liner:** DPO replaces PPO's training loop by reparameterizing the reward as a log-ratio of policy to reference model.

---

## Key Derivation (3-step summary)

```
1. Optimal KL-constrained RL policy:
   π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)

2. Invert to get implicit reward:
   r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)

3. Z(x) cancels in preference difference → DPO loss:
   L_DPO = -log σ(β·log_ratio_w - β·log_ratio_l)
   where log_ratio_y = log π_θ(y|x) - log π_ref(y|x)
```

---

## Key Code Pattern

```python
from trl import DPOConfig, DPOTrainer

training_args = DPOConfig(
    beta=0.1,           # KL coefficient — lower = less regularization
    learning_rate=5e-7, # Much lower than SFT
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_prompt_length=512,
    max_completion_length=256,
)

trainer = DPOTrainer(
    model=model,        # Training model (with LoRA)
    ref_model=ref_model,  # Frozen SFT model
    args=training_args,
    train_dataset=dataset,  # Must have 'prompt', 'chosen', 'rejected' columns
    processing_class=tokenizer,
)
trainer.train()
```

---

## DPO Health Metrics

| Metric | Healthy value at convergence |
|---|---|
| `reward_margin` | > 1.0 (positive and growing) |
| `rewards/chosen` | Positive and increasing |
| `rewards/rejected` | Negative and decreasing |
| Loss | Starting ~0.69, converging toward 0.3–0.5 |

---

## Decision Rules

- If `reward_margin` is flat at 0: reference model is not frozen, or LR is too small
- If DPO model increases refusal rate: rejected samples contain refusals — audit the dataset
- If hard examples do not improve: dataset is underrepresenting hard cases — oversample them
- DPO β ↑ → stays closer to SFT model (more conservative)
- DPO β ↓ → more freedom to diverge from SFT (higher risk of collapse)
- For verifiable rewards: use GRPO instead of DPO

---

## Numbers to Remember

- DPO LR: 5e-7 to 5e-6 (10–100x lower than SFT LR)
- DPO β: 0.01–0.2 (start at 0.1)
- Reference model: frozen, NOT updated
- 2 forward passes per step (train model + ref model) — 2x cheaper than PPO
- Dataset format: `prompt`, `chosen`, `rejected` columns

---

## Red Flags

- `rewards/chosen` and `rewards/rejected` both increasing: reference model is being updated
- Loss at exactly 0.693 after many steps: model learned nothing (log(0.5) = random binary)
- `reward_margin` < 0.2 after 500 steps: pairs too hard or β too high
- Model OOM: reduce `max_completion_length` or use `model_adapter_name` (Unsloth DPO path)
