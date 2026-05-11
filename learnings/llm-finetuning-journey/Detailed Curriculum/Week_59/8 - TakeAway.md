# Week 59 TakeAway — DPO on Refreshed Preference Data

**One-liner:** Hard on-policy pairs with execution-based labels; beta=0.2; stop before loss goes deeply negative.

---

## Preference Pair Quality Hierarchy

```
Hardest (most informative): chosen executes + correct rows
                            rejected executes + wrong rows
Medium: chosen executes + correct
        rejected has syntax error (but is plausible SQL attempt)
Easiest (least informative): chosen is gold teacher SQL
        rejected is obvious gibberish
```

## DPO Config

```python
DPOConfig(
    beta=0.2,              # lower = more aggressive (safe for verifiable SQL)
    learning_rate=5e-5,    # 4× lower than SFT
    num_train_epochs=1,    # DPO overfits fast on 5K pairs
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_prompt_length=1024,   # MUST SET — prevents silent skipping
    max_length=2048,
)
```

## Monitor These W&B Metrics

```
rewards/chosen        → must be positive and increasing
rewards/rejected      → must be negative (decreasing)
rewards/margin        → chosen - rejected, target > 0.3 by step 200
loss                  → target: -0.5 < loss < 0.1 (not deeply negative)
domain_exec_accuracy  → the ground truth — must improve vs SFT-v3
```

---

## Decision Rules

- If reward margin < 0.1 at step 100: pairs too similar OR reference model wrong — stop and diagnose
- If loss < -1.0 by step 200: early stopping; evaluate the step-200 checkpoint, it may be better
- Beta tuning: start at 0.2 for execution-verified SQL; increase to 0.4 if you see collapse
- Use LoRA reference trick (ref_model=None with LoRA) to save 14GB VRAM
- Stratify pairs: 60% on-policy student pairs, 30% hard semantic pairs, 10% teacher pairs

---

## Numbers to Remember

- DPO learning rate: 5e-5 (not 2e-4 like SFT)
- Beta range: 0.1–0.5; for SQL with clean labels, 0.2 is a good start
- 5K pairs × 1 epoch at A100: ~2 hours
- Reward margin target by end of training: > 0.5
- Expected improvement from DPO: +2–7pp on custom benchmark

---

## Red Flags

- Reward margin stays at 0 after 200 steps: reference model mismatch — check π_ref loading
- Loss = -5.0 at step 200: overfit to training pairs; eval may be worse than SFT baseline
- `max_prompt_length` not set: long prompts silently dropped, training on a biased subset
- DPO checkpoint forgot general instruction-following: test on 5 general questions before proceeding to GRPO
