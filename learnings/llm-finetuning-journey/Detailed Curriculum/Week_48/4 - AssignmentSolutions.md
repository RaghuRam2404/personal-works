# Week 48 Assignment Solutions

## Task 1 — One-Step Verification

The most common issues in the first step:

```
Issue: CUDA out of memory
Fix: Add to GRPOConfig:
  gradient_checkpointing=True,
  load_in_4bit=True (for ref model)
  K=4 instead of K=8

Issue: reward_fn returns wrong type
Fix: Ensure reward_fn returns list[float], not list[int] or tensor
  rewards = [float(r) for r in rewards]  # explicit cast

Issue: "completions and prompts have different lengths"
Fix: GRPO passes flat lists (batch_size * K completions, batch_size * K prompts)
  Ensure your reward_fn handles this: len(completions) == batch_size * K

Issue: reward_fn called with unexpected kwargs
Fix: Add **kwargs to the signature:
  def sql_reward_fn(completions, prompts, **kwargs) -> list[float]:
```

Expected step 1 output:
```
Step 1/1000 | loss: 0.234 | mean_reward: 0.182 | kl: 0.002 | 
grad_norm: 0.43 | time: 87s
```

Step time of 60–120s per step is normal for GRPO with K=8 on a 7B model.

---

## Task 2 — GRPO Training Run Reference Config

```python
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from reward_fn import sql_reward_fn
from datasets import load_dataset
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    "<your-handle>/postgres-sqlcoder-7b-v2-dpo",
    max_seq_length=1024,
    dtype=None,           # Auto (bf16 on A100)
    load_in_4bit=False,   # bf16 on A100 80GB
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32,
    target_modules="all-linear", lora_dropout=0.05)

ref_model, _ = FastLanguageModel.from_pretrained(
    "<your-handle>/postgres-sqlcoder-7b-v2-dpo",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,    # 4-bit for frozen ref model to save VRAM
)
ref_model.eval()

dataset = load_dataset("json", data_files="train_prompts.jsonl")["train"]

config = GRPOConfig(
    num_generations=8,
    max_completion_length=256,
    learning_rate=5e-7,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=1000,
    save_steps=50,
    save_total_limit=5,
    output_dir="./grpo_checkpoints",
    temperature=0.7,
    kl_coef=0.05,
    max_grad_norm=0.5,       # Gradient clipping for stability
    report_to="wandb",
    run_name="grpo-v3-run1",
    bf16=True,
    warmup_steps=10,
)

trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    config=config,
    train_dataset=dataset,
    reward_fn=sql_reward_fn,
    processing_class=tokenizer,
)
trainer.train()
model.push_to_hub("<your-handle>/postgres-sqlcoder-7b-v3-grpo")
```

**Expected training progression:**
- Steps 1–100: mean_reward slowly increasing (0.18 → 0.25)
- Steps 100–400: faster improvement as the model finds reward patterns (0.25 → 0.40)
- Steps 400–1000: gradual improvement with plateau (0.40 → 0.50)
- Final KL: 0.5–3.0 nats (healthy range)

**Common gotchas:**
- If step time exceeds 200s: K is too large or generation is too slow — reduce K or max_completion_length
- If mean_reward is exactly your diagnostic baseline at step 100: check that the model is generating NEW completions, not returning cached ones
- If grad_norm spikes to > 10 at any step: stop, restart with lower LR (1e-7) — one bad update can corrupt the LoRA adapters
- RunPod sessions can disconnect; use `screen` or `tmux` + `nohup python grpo_train.py &` to keep training running

---

## Task 3 — Expected Eval Results

| Metric | v1 (SFT) | v2 (DPO) | v3 (GRPO) |
|---|---|---|---|
| Execution accuracy | ~70% | ~80% | ~88%+ |
| Semantic accuracy | ~50% | ~60% | ~65%+ |
| Syntax error rate | ~20% | ~8% | ~5% |
| Complex query exec. acc. | ~45% | ~48% | ~60%+ |
| Mean generation length | 120 | 115 | 145 |

If v3 is better than v2 overall but not on complex queries: the GRPO training dataset lacked complex examples. Add them in Week 50.

---

## How to Verify You Did It Right

1. `mean_reward` at step 1000 is higher than at step 1 (even 0.03 higher counts as training).
2. v3 execution accuracy > v2 execution accuracy by at least 5pp.
3. The HF Hub model `v3-grpo` is accessible and loads without error.
4. W&B run shows 1000 steps logged (not just 50 or 100 before stopping).
5. Complex query tier: v3 > v1 (even if v3 ≈ v2 on complex queries, this shows GRPO stabilized quality).
