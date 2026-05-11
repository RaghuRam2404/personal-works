# Week 59 Assignment Solutions

## Task 1 — Candidate Generation and Pair Labeling

```python
import hashlib

def result_hash(cur, sql):
    """Execute SQL and return a hash of the result set for comparison."""
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        return hashlib.md5(str(sorted(rows)).encode()).hexdigest()
    except Exception as e:
        return f"error:{str(e)[:40]}"

def build_preference_pair(prompt, candidates, reference_sql, conn):
    """Returns (chosen, rejected) or None if no valid pair."""
    cur = conn.cursor()
    
    # Get reference result hash
    ref_hash = result_hash(cur, reference_sql)
    
    correct = []   # executes AND matches reference
    wrong = []     # executes but wrong result
    failed = []    # syntax error
    
    for sql in candidates:
        h = result_hash(cur, sql)
        if h == ref_hash:
            correct.append(sql)
        elif h.startswith("error:"):
            failed.append(sql)
        else:
            wrong.append(sql)  # executes but wrong
    
    if not correct:
        return None  # no chosen candidate
    
    chosen = min(correct, key=lambda s: len(s))  # prefer concise
    
    # Rejected preference: wrong > failed (harder negative)
    if wrong:
        rejected = wrong[0]
    elif failed:
        rejected = failed[0]
    else:
        return None  # all candidates are correct; no usable rejected
    
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected,
            "chosen_executes": True, "rejected_executes": len(wrong) > 0}
```

## Task 2 — Reference Model Loading

```python
from trl import DPOTrainer, DPOConfig
from peft import PeftModel

# Training model: SFT-v3 with new LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    "<your-handle>/qwen2.5-coder-7b-postgres-sft-v3",
    load_in_4bit=False
)
model = FastLanguageModel.get_peft_model(model, r=16, ...)

# Reference model: frozen SFT-v3 (no new LoRA)
# TRL DPOTrainer accepts ref_model=None when using LoRA
# in that case it uses the SFT base weights as reference
# This only works if model has LoRA adapters (the diff is the DPO update)
```

---

## Common Gotchas

- **DPO loss goes to -5.0 by step 100.** Beta too low (0.1) with very easy preference pairs. Increase beta to 0.3 or use harder pairs only.
- **Reward margin stays at 0.** Usually means chosen and rejected are too similar in token probability — your pairs are too hard. Or: the reference model is wrong (you're computing π_ref against the wrong checkpoint).
- **Colab runtime disconnects during training.** DPO on 4K pairs takes ~2 hours on A100. Enable WandB checkpointing every 100 steps so you can resume.
- **`chosen` and `rejected` sequences exceed max_length.** Set `max_prompt_length=1024` and `max_length=2048` in DPOConfig. DPO processes prompt + response together; if total > max_length, the example is silently skipped.

---

## How to Verify You Did It Right

1. `dpo_pairs_v3.jsonl` has ≥ 4,000 lines; each line has `prompt`, `chosen`, `rejected` keys
2. At least 2,000 pairs where both chosen and rejected execute (the "hard" pairs)
3. W&B shows `rewards/chosen` consistently > `rewards/rejected` by step 200
4. DPO loss stays between -0.5 and 0.5 (not going deeply negative)
5. `dpo_eval_results.md` shows ≥ 2pp improvement on custom benchmark over SFT-v3
6. DPO checkpoint pushed to HuggingFace
