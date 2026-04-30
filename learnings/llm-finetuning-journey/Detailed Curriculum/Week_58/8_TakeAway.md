# Week 58 TakeAway — Full SFT on v3

**One-liner:** Start from CPT checkpoint, LoRA rank 32, 2 epochs, completion-only loss, monitor execution accuracy not just val loss.

---

## Key Config

```python
peft_config = LoraConfig(r=32, lora_alpha=64,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"])

training_args = SFTConfig(
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # effective = 32
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    bf16=True,
    load_best_model_at_end=True,
)

# CRITICAL: loss only on assistant tokens
collator = DataCollatorForCompletionOnlyLM(
    response_template="<|im_start|>assistant\n",
    tokenizer=tokenizer,
)
```

## Chat Template Application

```python
text = tokenizer.apply_chat_template(
    example["messages"],
    tokenize=False,
    add_generation_prompt=False  # False during training
)
```

---

## Decision Rules

- Start SFT from CPT checkpoint, NOT base model (saves ~100 steps of convergence)
- Use completion-only collator — full-token loss degrades performance by 2–4pp
- Stop at epoch where domain execution accuracy peaks (not when val loss floors)
- If val loss increases and exec accuracy still rising: trust exec accuracy, stop at next eval
- LoRA rank 32: safe default. Rank 64: 1–2pp better, 4× more LoRA params
- Warmup steps: 100 (shorter than CPT's 1000 — model is already initialized)

---

## Numbers to Remember

- Starting loss from CPT: ~1.5–2.0 (vs. ~2.4 from base model)
- Target final training loss: ~0.8–1.2
- Expected training time at H100: ~6 hours for 25K × 2 epochs at effective batch 32
- RunPod cost estimate: ~$17
- Expected improvement over Phase 5 baseline: +5–15 pp on custom benchmark

---

## Red Flags

- Starting loss < 1.0: data contamination or CPT overfitting
- Training loss not decreasing after warmup: collator misconfigured (computing loss on user tokens)
- Training loss spikes > 0.5: bad data batch or LR too high
- domain_exec_accuracy flat at 0.0: eval script is broken or prompt format mismatch
- Instance not terminated after training: check RunPod console
