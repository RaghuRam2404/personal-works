# Week 38 Assignment Solutions

## Task 1 — Production Training Script: Complete Version

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch, os, json

os.environ["WANDB_PROJECT"] = "week-38-qlora-15k"

MODEL_ID = "Qwen/Qwen2.5-Coder-7B"

# Load model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID, max_seq_length=512,
    dtype=None, load_in_4bit=True,
)

# Apply DoRA adapter (change use_dora=False for standard LoRA)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    use_dora=True,   # From Week 36 decision
    random_state=42,
)
model.print_trainable_parameters()

# Load dataset
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl("train_15k.jsonl")
val_data = load_jsonl("val_500.jsonl")

# Convert to HuggingFace dataset
from datasets import Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Train
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="./postgres-sqlcoder-7b-v1",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        max_seq_length=512,
        packing=True,
        logging_steps=10,
        eval_steps=100,
        evaluation_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to="wandb",
        run_name="postgres-sqlcoder-7b-v1",
        dataset_text_field="text",
    ),
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
model.save_pretrained("./postgres-sqlcoder-7b-v1")
tokenizer.save_pretrained("./postgres-sqlcoder-7b-v1")
model.push_to_hub("<your-handle>/postgres-sqlcoder-7b-v1")
```

---

## Expected Training Results (A100 40GB)

| Metric | Expected Value |
|---|---|
| Trainable params | ~43M (DoRA, rank 16) |
| Peak VRAM | 18–26 GB |
| Training time | 15–30 minutes |
| Final train loss | 0.4–0.8 |
| Final eval loss | 0.9–1.3 |

---

## Task 3 — Expected Evaluation Results

| Model | Training Data | Exact Match | Valid SQL |
|---|---|---|---|
| Qwen2.5-Coder-7B base | None | 5–15% | 60–80% |
| Week 33 model | 5K examples | 30–50% | 85–95% |
| Week 38 model (v1) | 15K examples | 50–70% | 90–97% |

The 15K model should outperform the 5K model by 15–25 percentage points on exact match, primarily because:
1. More schema diversity → better generalization to unseen table names/structures
2. More complex query types covered → model can handle JOINs and GROUP BY reliably
3. PostgreSQL-specific examples reduce MySQL-syntax generation errors

---

## Common Gotchas

- **Validation set has overlapping schemas with test set**: Verify with dedup check before training. If overlap found, regenerate validation split from non-test-overlapping examples.
- **Load_best_model_at_end fails if only 1 checkpoint saved**: Set `save_steps=100` and ensure eval happens at same interval. The best checkpoint must exist on disk.
- **HuggingFace push quota**: Large adapters (>100MB) may hit free tier limits. Verify storage quota before pushing.
- **ChatML formatting mismatch**: If you changed system prompt between Week 29 and Week 38, the model card must document the exact prompt format for inference users.

---

## How to Verify You Did It Right

- Week 38 model achieves at least 15% higher exact match than Week 33 model on the same 100 examples
- W&B run shows 2 epochs with decreasing then stabilizing eval loss
- `postgres-sqlcoder-7b-v1` is publicly visible on HuggingFace with a complete model card
- Error analysis shows a distribution of failure types — no single category exceeds 50% of errors (if it does, that's a dataset gap to fix in v2)
- Training completed without loss spikes (all steps visible in W&B, no NaN)
