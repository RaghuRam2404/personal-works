# Week 33 — QLoRA: Your First 7B Fine-Tune

## Learning Objectives

By the end of this week, you will be able to:

- Combine bitsandbytes 4-bit NF4 quantization with peft LoRA into a single QLoRA training pipeline
- Fine-tune `Qwen2.5-Coder-7B` on your domain dataset in under 30 minutes on a Colab Pro A100
- Explain why QLoRA uses bfloat16 for LoRA adapter compute despite an INT4 base model
- Evaluate a fine-tuned 7B model on your held-out 100-example SQL test set
- Push a 7B adapter to HuggingFace Hub

---

## Concepts

### 1. QLoRA = NF4 Base + LoRA Adapters

QLoRA (Dettmers et al., 2023) is elegantly simple in implementation despite its theoretical sophistication:

1. Load the base model in 4-bit NF4 (frozen — no gradients)
2. Apply LoRA adapters on top (trainable — gradients here only)
3. Train with standard cross-entropy loss via SFTTrainer
4. The 4-bit base uses NF4 storage but dequantizes to BF16 for computation

The mathematical insight that makes this work: the LoRA adapters are in BF16 precision, not 4-bit. The gradient signal flows through the BF16 LoRA matrices, not through the frozen 4-bit weights. The 4-bit weights are frozen — they never receive gradients. This means there is no "4-bit gradient" problem.

```
Forward pass:
  x → [NF4 frozen W, dequantized to BF16] → base_out
  x → [BF16 lora_A] → [BF16 lora_B] → lora_out
  output = base_out + lora_out * scaling

Backward pass:
  gradient flows only through lora_A and lora_B (BF16)
  W is not in the computation graph (frozen)
```

### 2. Memory Breakdown for 7B QLoRA on A100 24GB

| Component | Size |
|---|---|
| 7B model in NF4 (with grouping overhead) | ~4.5 GB |
| LoRA adapters (rank 16, all layers) | ~0.1 GB |
| Adapter gradients + optimizer states (AdamW) | ~0.5 GB |
| Activation memory (batch 4, seq 512) | ~4–8 GB |
| Total estimate | ~10–14 GB |

Fits comfortably in a Colab Pro A100 (40GB). Even fits in a 24GB consumer GPU (RTX 3090/4090) with batch size 2–4 and gradient checkpointing.

### 3. Paged Optimizers

QLoRA introduces paged optimizers: when GPU memory is pressured, optimizer states (AdamW first/second moments for LoRA params) are transparently paged to CPU RAM and fetched back when needed. This prevents OOM crashes during long training runs.

Enabled via `optim="paged_adamw_8bit"` (for 8-bit AdamW optimizer, which further compresses optimizer state memory) or `optim="paged_adamw_32bit"`.

For a 7B QLoRA run on Colab Pro A100, paged optimizers are optional (A100 has enough VRAM) but good practice:

```python
SFTConfig(
    ...
    optim="paged_adamw_8bit",
    ...
)
```

### 4. Compute dtype vs. Storage dtype

The most confusing aspect of QLoRA for beginners:

- **Storage dtype:** NF4 (4-bit). This is how weights are stored in GPU memory.
- **Compute dtype:** BF16. When a matmul happens, weights are dequantized to BF16 on the fly, the compute runs in BF16, and the result is in BF16.

This dequantize-on-the-fly approach is what bitsandbytes implements. It means: (1) memory usage is 4-bit (low), (2) compute precision is BF16 (acceptable), (3) there is a small overhead for the dequantization step.

The `bnb_4bit_compute_dtype=torch.bfloat16` parameter in `BitsAndBytesConfig` controls the compute dtype. **Always use BF16, not FP16**, to avoid overflow during computation.

### 5. Gradient Checkpointing

For 7B training with QLoRA, enable gradient checkpointing to trade compute for memory:

```python
model.gradient_checkpointing_enable()
# or in SFTConfig: gradient_checkpointing=True
```

This recomputes activations during the backward pass instead of storing them, reducing activation memory by 60–70% at the cost of ~20% more compute. On A100 with 7B, it is often not needed for batch size 1–4, but use it if you see OOM errors.

### 6. The Complete QLoRA Setup

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

# Step 1: Load in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B",
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False  # Required for gradient checkpointing

# Step 2: Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Step 3: Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="./qwen-coder-7b-qlora",
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
        report_to="wandb",
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

### 7. Evaluating Your 7B Model

After training, run your fine-tuned model on the 100-example held-out test set from Week 32. Evaluation steps:
1. Format each test example with the chat template (system + user, no assistant response)
2. Generate SQL with `do_sample=False` (greedy decoding for reproducibility)
3. Check: does the generated SQL match the expected SQL exactly (exact match)?
4. Optionally: check whether the SQL is syntactically valid Python using `sqlparse` library

At this stage, target at least 30% exact match on the 100 examples. This is a low bar — your model is trained on generic SQL, not specifically tuned to your test set. Weeks 37–39 will raise this substantially.

---

## Connections

**Builds on:** Week 31 (peft LoRA setup), Week 32 (bitsandbytes 4-bit quantization). QLoRA is directly the combination.

**Needed for:** Week 34 (Unsloth will speed up this exact pipeline). Week 35 (you sweep hyperparameters on this setup). Week 38 (the domain sprint uses this setup at 15K examples).

---

## Common Misconceptions / Pitfalls

- **"QLoRA trains the base model weights."** No — the NF4 weights are frozen. Only A and B matrices train.
- **"I need to dequantize before saving."** No — save with `model.save_pretrained()` to save only the LoRA adapter. The NF4 base is loaded separately.
- **"`model.config.use_cache = False` is optional."** No — you must set this before `gradient_checkpointing_enable()` or you will get a warning/error.
- **"paged optimizers reduce training quality."** No — they are numerically identical to standard AdamW; paging only affects when optimizer states live in memory.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read QLoRA paper (fully) | 2h |
| Set up Colab Pro A100 (request if needed) | 30m |
| Write and debug QLoRA training script | 2h |
| Run training (estimate 20–40 min on A100 with 5K examples) | 1h |
| Evaluate on held-out test set, push to HuggingFace | 1h |
| Commit and document results | 30m |
