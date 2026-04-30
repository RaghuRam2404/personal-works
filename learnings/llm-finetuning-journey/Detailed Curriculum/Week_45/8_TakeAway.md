# Week 45 TakeAway — DPO on Your Domain Model

**One-liner:** DPO reduces syntax errors and improves easy queries; complex queries need GRPO's online reward.

---

## Training Config Reference

```python
DPOConfig(
    beta=0.1,                        # Start here; try 0.05 if reward_margin < 0.3
    learning_rate=5e-7,              # Much lower than SFT; try 1e-7 if oscillating
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # Effective batch = 16
    max_prompt_length=512,
    max_completion_length=256,
    num_train_epochs=1,              # Rarely need more than 1 epoch
    bf16=True,                       # A100 supports bfloat16
    warmup_ratio=0.1,
)
```

---

## Eval Metrics (Run on 200 held-out prompts)

| Metric | How to compute |
|---|---|
| Execution accuracy | % where execute_sql() returns success=True |
| Semantic accuracy | % where rows == reference_rows |
| Syntax error rate | % where error contains "syntax" |
| Empty result rate | % where success=True but row_count=0 |

---

## Decision Rules

- If reward_margin < 0.2 after training: check for mislabeled data first, then lower β
- If v2 execution accuracy < v1: data quality problem — audit 50 pairs before hyperparameter tuning
- If v2 is better on easy but not complex: normal — GRPO will handle complex in Weeks 47–48
- If refusal rate increased: rejected examples contained non-SQL text — filter preference data
- If loss oscillates: learning rate too high or noisy data — reduce LR by 5×

---

## Reference Model Integrity Check

```python
# Run this before training to verify ref model is frozen
n_frozen = sum(1 for p in ref_model.parameters() if not p.requires_grad)
n_total = sum(1 for p in ref_model.parameters())
assert n_frozen == n_total, f"Found {n_total - n_frozen} non-frozen params in ref model!"
```

---

## Numbers to Remember

- DPO β: 0.1 default; 0.05–0.2 range
- DPO LR: 5e-7 (5× lower than typical SFT)
- Healthy reward_margin at convergence: 0.5–2.0
- 7B model in bf16: ~14GB VRAM
- 7B model in 4-bit: ~5GB VRAM (use for ref model if memory constrained)
- Expected eval improvement (DPO): +10–15pp on simple queries, minimal on complex

---

## Red Flags

- v2 worse than v1 on test set: data quality issue, not hyperparameter issue
- `rewards/chosen` < 0 throughout training: chosen SQL is less probable than π_ref's expectation (very wrong pairs)
- reward_margin negative: mislabeled pairs — stop training, audit data
- Loss perfectly flat at 0.693: β is too high, model is stuck at π_ref
