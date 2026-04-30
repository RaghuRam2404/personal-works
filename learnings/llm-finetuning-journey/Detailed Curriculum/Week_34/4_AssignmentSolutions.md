# Week 34 Assignment Solutions

## Task 2 — Unsloth Script: Complete Key Snippet

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import torch, os

os.environ["WANDB_PROJECT"] = "week-34-unsloth"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-Coder-7B",
    max_seq_length=512,
    dtype=None,        # Auto: BF16 on Ampere+, float16 on older
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,    # 0 dropout = maximum speed
    bias="none",
    use_gradient_checkpointing="unsloth",  # NOT True — use "unsloth"
    random_state=42,
)

# Tokenizer setup (Unsloth may handle this automatically)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="./qwen-coder-7b-unsloth",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=512,
        packing=True,
        logging_steps=10,
        eval_steps=100,
        evaluation_strategy="steps",
        report_to="wandb",
        run_name="qwen-coder-7b-unsloth-sql-5k",
        dataset_text_field="text",
        # NOTE: Do NOT set gradient_checkpointing=True here when using
        # use_gradient_checkpointing="unsloth" in get_peft_model
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

---

## Expected Results

| Metric | Vanilla QLoRA | Unsloth | Typical Speedup |
|---|---|---|---|
| Steps/second | 0.8–1.2 | 1.8–3.5 | 2–3x |
| Peak VRAM (GB) | 22–30 | 12–18 | 40–60% less |
| Training time (5K, 2 ep) | 40–50 min | 15–25 min | 2–3x |
| Final eval loss | 1.0–1.5 | 1.0–1.5 | Equal |

Results vary with A100 vs. V100 and batch size. On A100 40GB, the VRAM savings allow increasing batch size, which further improves throughput.

---

## Common Gotchas

- **`use_gradient_checkpointing="unsloth"` must be in `get_peft_model`, not in `SFTConfig`.** If you also set `gradient_checkpointing=True` in SFTConfig, there will be a conflict. Use one or the other — Unsloth's version is better.
- **`dtype=None` vs `dtype=torch.bfloat16`:** `None` lets Unsloth auto-detect; on A100 it picks BF16. Explicitly setting `dtype=torch.bfloat16` is equivalent but explicit.
- **Unsloth installation issues:** Run `pip install "unsloth[colab-new] @ git+..."` exactly as in the Unsloth README for the current Colab version. The pip release may lag behind the GitHub version.
- **SFTTrainer compatibility:** Unsloth models work with the standard TRL `SFTTrainer`. If you see errors, check that your `trl` version matches Unsloth's requirements (listed in their pyproject.toml).
- **VRAM measurement**: `torch.cuda.max_memory_allocated()` must be called after the first training step — it returns 0 before any computation.

---

## How to Verify You Did It Right

- Steps per second with Unsloth is at least 1.5x higher than vanilla QLoRA on the same GPU
- Peak VRAM with Unsloth is at least 20% lower than vanilla QLoRA
- Final eval loss of Unsloth model equals vanilla model within 0.05 loss units (same quality)
- `week34_comparison.md` has the complete table with all 4 metrics filled in from both runs
