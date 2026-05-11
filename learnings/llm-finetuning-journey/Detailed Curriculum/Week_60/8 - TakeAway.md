# Week 60 TakeAway — Final GRPO Training

**One-liner:** GRPO needs reward variance within each group — select prompts where the model sometimes fails, always use partial credit rewards.

---

## Reward Function Skeleton

```python
def reward(completion, reference_sql, schema_ddl, conn) -> float:
    sql = extract_sql(completion)
    if sql is None: return -1.0
    try:
        rows = execute(sql, schema_ddl, conn)
        exec_r = 1.0
    except: return -0.5
    
    ref_rows = execute(reference_sql, schema_ddl, conn)
    if rows == ref_rows:           acc_r = 1.0
    elif len(rows) == len(ref_rows): acc_r = 0.2  # partial
    else:                            acc_r = 0.0
    
    fmt_r = 0.1 if is_valid_sql_statement(sql) else 0.0
    return exec_r + acc_r + fmt_r   # range: -1 to 2.2
```

## GRPO Config

```python
GRPOConfig(
    num_generations=8,   # K — reduce to 4 if OOM
    temperature=0.9,     # MUST be > 0
    learning_rate=5e-6,  # very low for final alignment step
    kl_coef=0.05,        # increase to 0.1 if KL > 10 bits
    max_new_tokens=512,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
)
```

## Prompt Selection Logic

```python
# Select prompts where DPO-v3 fails (reward variance > 0)
# Skip prompts where all 8 samples would be identical (wrong OR right)
# Oversample TimescaleDB prompts (40% target)
```

---

## Decision Rules

- If all K rewards are equal: this prompt gives zero gradient — replace it with a harder one
- If KL divergence > 10 bits: stop, increase kl_coef, resume from earlier checkpoint
- If GRPO regresses on eval: DPO-v3 is your final model — GRPO failure is a valid result
- Use partial credit reward (not binary) when K ≤ 8 — sparse binary signal causes too many zero-gradient steps
- Merge adapters to bf16 for the final push to HuggingFace

---

## Numbers to Remember

- GRPO learning rate: 5e-6 (10× lower than DPO's 5e-5)
- GRPO kl_coef: 0.05 starting point
- Group size K=8: 8 inference passes per prompt
- 1,500 prompts × K=8 × 1 epoch ≈ 3.5 hours on H100 ≈ $10
- Mean reward target: increase from ~0.8 to ~1.5 over training
- KL divergence safety limit: < 10 bits from reference

---

## Red Flags

- Mean reward flat at ~0.8 after 100 steps: all prompts too easy OR reward function broken
- KL > 10 bits: policy drifted too far, stop and increase kl_coef
- GRPO model regresses on eval vs DPO: valid result — use DPO as final model
- RunPod instance left running: check console after training
