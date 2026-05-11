# Week 45 Assignment Solutions

## Task 1 — Unsloth DPO Training Script

```python
from unsloth import FastLanguageModel
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset
import torch

# Load model with Unsloth (4-bit for memory efficiency, or bf16 if A100)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="<your-handle>/postgres-sqlcoder-7b-v1",
    max_seq_length=1024,
    dtype=torch.bfloat16,
    load_in_4bit=False,  # Use bf16 on A100 for full precision
)

# Apply LoRA to the training model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)

# Load reference model separately (frozen, no LoRA)
from transformers import AutoModelForCausalLM
ref_model = AutoModelForCausalLM.from_pretrained(
    "<your-handle>/postgres-sqlcoder-7b-v1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
for p in ref_model.parameters():
    p.requires_grad_(False)

# Load preference dataset
dataset = load_dataset("<your-handle>/postgres-sql-preferences-v1")

training_args = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_prompt_length=512,
    max_completion_length=256,
    num_train_epochs=1,
    logging_steps=25,
    save_strategy="epoch",
    report_to="wandb",
    run_name="week-45-dpo-sql",
    bf16=True,
    warmup_ratio=0.1,
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

trainer.train()
model.push_to_hub("<your-handle>/postgres-sqlcoder-7b-v2-dpo")
```

**Expected training metrics (1 epoch on 1800 pairs):**
- Duration: ~1–2 hours on Colab Pro A100
- Final reward_margin: 0.4–1.2 (depends on dataset quality)
- Final loss: 0.35–0.55 (down from 0.69 at start)

**Common gotchas:**
- Colab Pro A100 has 40GB — a 7B model in bf16 uses ~14GB, leaving enough for the ref model. If OOM, use 4-bit for the ref model only.
- If `ref_model` is on the same device as the training model and memory is tight: use `model_adapter_name` trick in Unsloth's DPO to avoid loading a second 7B model.
- The DPOConfig `max_prompt_length` and `max_completion_length` are separate from the model's `max_seq_length`; ensure they sum to less than 1024 (or whatever your max_seq_length is).
- If reward_margin is negative after 100 steps: stop training. Check 5 pairs manually — they may be mislabeled.

---

## Task 2 — Expected Eval Results

Typical results comparing SFT-only (v1) vs. SFT+DPO (v2):

| Metric | v1 (SFT only) | v2 (SFT+DPO) |
|---|---|---|
| Execution accuracy | 65–75% | 75–85% |
| Semantic accuracy | 45–60% | 50–65% |
| Syntax error rate | 15–25% | 5–10% |
| Empty result rate | 8–12% | 5–8% |

Breakdown by complexity:
- Simple queries: v2 > v1 by 10–15pp (DPO directly fixes syntax errors that appear in preference data)
- Medium queries: v2 > v1 by 5–8pp
- Complex queries: v2 ≈ v1 (or v2 slightly worse — DPO has no execution-time signal for novel hard queries)

---

## How to Verify You Did It Right

1. v2 execution accuracy > v1 on the full test set. If not, re-examine the preference data quality.
2. W&B shows `reward_margin` growing monotonically during training.
3. Spot-check 5 generated SQL from v2 for prompts that v1 got wrong: does v2 get them right?
4. `rewards/chosen` is positive and `rewards/rejected` is negative at the end of training.
5. Model size on Hub: v2-dpo should have the same architecture as v1, just different weights (the LoRA is merged, so file size is similar).
