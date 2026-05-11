# Week 38 TakeAway — Production QLoRA 15K Fine-Tune

**One-liner:** The main event: Unsloth + DoRA + 15K SQL on A100 → postgres-sqlcoder-7b-v1 in ~20 minutes.

---

## Production Training Script Summary

```python
# Unsloth model load
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B", max_seq_length=512, dtype=None, load_in_4bit=True)

# DoRA adapter
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth",
    use_dora=True, random_state=42)

# SFTConfig
SFTConfig(
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
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    eval_steps=100,
)
```

---

## Expected Outcomes

| Stage | Model | Exact Match |
|---|---|---|
| Baseline | Qwen2.5-Coder-7B base | 5–15% |
| Week 33 | 5K QLoRA | 30–50% |
| Week 38 | 15K QLoRA/DoRA | 50–70% |

---

## Training Cost (15K, A100, 2 epochs)

- Training time: 15–30 minutes
- Colab Pro cost: ~$0.30–0.60
- VRAM: 18–26 GB

---

## Quick Decisions During Training

| W&B Signal | Action |
|---|---|
| Loss spike >3 at any step | Stop, add `max_grad_norm=1.0`, restart |
| Eval loss rising from step 100 | Wait until step 300 to confirm overfitting |
| Steps/sec drops below 1 | Reduce batch size to 2 |
| Eval loss still high at step 500 | Dataset quality issue — check formatting |

---

## Post-Training Checklist

- [ ] `postgres-sqlcoder-7b-v1` pushed to HuggingFace
- [ ] Model card complete with training details and eval results
- [ ] 100-example comparison table written: base vs. 5K vs. 15K
- [ ] Error analysis categorized by failure type
- [ ] `week38_results.md` and `week38_error_analysis.md` committed
- [ ] GitHub commit: `week-38-qlora-15k`
