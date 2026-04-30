# Week 33 TakeAway — QLoRA: First 7B Fine-Tune

**One-liner:** QLoRA = 4-bit frozen base + BF16 LoRA adapters. Train 7B on 24GB. Gradients never touch the NF4 weights.

---

## Complete QLoRA Setup

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model.config.use_cache = False  # REQUIRED before gradient checkpointing

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
```

---

## SFTConfig Defaults for 7B QLoRA

```python
SFTConfig(
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    max_seq_length=512,
    packing=True,
)
```

---

## Memory Breakdown (7B, Rank 16)

| Component | Memory |
|---|---|
| NF4 model (7B) | ~4.5 GB |
| LoRA adapters (42M params, BF16) | ~0.1 GB |
| Optimizer states (8-bit AdamW) | ~0.1 GB |
| Activations (batch 4, seq 512) | ~4–8 GB |
| **Total** | **~10–14 GB** |

---

## Numbers to Remember

- 7B NF4 = ~4.5 GB (vs. 14 GB BF16)
- Rank 16 LoRA on 7B = ~42M trainable params = 0.58% of total
- A100 40GB: fits 7B QLoRA at batch 8+; T4 16GB: batch 1–2 with gradient checkpointing
- Training time on A100 for 5K examples, 2 epochs: 25–45 min

---

## Decision Rules

- OOM on A100 → enable `gradient_checkpointing=True`
- OOM on T4 → reduce batch size to 2, reduce max_seq_length to 256
- Loss plateau at step 200–500 → more data needed (not more epochs)
- eval loss == train loss at plateau → capacity limit, expand dataset

---

## Red Flags

- `model.config.use_cache = False` not set → error with gradient checkpointing
- Trainable % > 5% → LoRA not applied correctly or too many target_modules
- VRAM > 35GB on A100 → disable packing or reduce batch
- Loss starts at 10+ → BF16 compute dtype not set; using FP16 overflow
