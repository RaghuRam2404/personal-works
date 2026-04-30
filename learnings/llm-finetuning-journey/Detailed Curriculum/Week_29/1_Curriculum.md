# Week 29 — Full SFT on a Tiny Model

## Learning Objectives

By the end of this week, you will be able to:

- Set up a complete SFT training loop using HuggingFace `SFTTrainer` from `trl`
- Format a PostgreSQL text-to-SQL dataset into a chat-template-compatible format
- Configure and debug `Qwen2.5-0.5B`'s tokenizer, including the chat template
- Push a fine-tuned model to HuggingFace Hub and log training to Weights & Biases
- Interpret training loss curves and identify basic failure modes

---

## Concepts

### 1. The SFTTrainer Interface

`SFTTrainer` is built on top of HuggingFace `Trainer` and adds two key conveniences: (1) automatic input masking — it masks the prompt tokens from the loss so only response tokens contribute, and (2) packing — it can concatenate multiple short examples into a single sequence up to `max_seq_length`, improving GPU utilization.

The minimal SFTTrainer setup:

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./qwen-0.5b-sft",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
    ),
    train_dataset=dataset,
)
trainer.train()
```

The `dataset` must have a `text` field (if using `dataset_text_field`) or messages in a conversational format. For this week, you will use the conversational format with `apply_chat_template`.

### 2. Chat Templates and Qwen2.5

Every modern model has a chat template — a specific way of formatting (system prompt, user message, assistant response) into a flat token sequence. Qwen2.5 uses the ChatML format:

```
<|im_start|>system
You are a PostgreSQL expert. Given a schema and question, output valid SQL.<|im_end|>
<|im_start|>user
Schema: ...
Question: ...<|im_end|>
<|im_start|>assistant
SELECT ...<|im_end|>
```

The critical fact: **the chat template determines where the model learns to respond.** If you format data incorrectly, the model will learn to predict the wrong tokens. Always call `tokenizer.apply_chat_template(messages, tokenize=False)` and inspect the output string before training.

Qwen2.5's special tokens:
- `<|im_start|>` — beginning of a message block
- `<|im_end|>` — end of a message block
- `<|endoftext|>` — end of document

### 3. Dataset Format for SQL SFT

Your training examples should look like:

```python
{
    "messages": [
        {"role": "system", "content": "You are a PostgreSQL expert..."},
        {"role": "user", "content": f"Schema:\n{schema}\nQuestion:\n{question}"},
        {"role": "assistant", "content": sql_answer}
    ]
}
```

For this week, use 1K examples from a publicly available text-to-SQL dataset such as [Spider](https://yale-nlp.github.io/spider/) or [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) on HuggingFace. Both are small enough to download on Colab Free. Filter for PostgreSQL-compatible queries where possible.

### 4. Full SFT vs. LoRA SFT

This week you are doing full SFT: all model weights are updated. For `Qwen2.5-0.5B` (500M parameters), this is feasible on Colab Pro with batch size 4 and gradient accumulation. At 7B parameters, full SFT requires 80GB+ VRAM — that is why you will switch to LoRA in Weeks 30–31 and QLoRA in Week 33.

Memory usage in full SFT for a model with P parameters (mixed-precision fp16):
- Model weights: 2P bytes
- Gradients: 2P bytes
- Optimizer states (AdamW): 8P bytes
- Activations (depends on batch size, sequence length)

For Qwen2.5-0.5B (P ≈ 500M): ~6GB for model + gradients + optimizer. Fits comfortably on Colab Pro's 16GB A100.

### 5. Interpreting Training Loss

A healthy SFT training run for a small model on 1K examples typically shows:

- Initial loss: 2.0–3.0 (depends on how unfamiliar the output format is)
- Loss after 500 steps: 1.0–1.8 (the model has learned the output pattern)
- Converging at 0.3–0.8 after 2 epochs (possibly overfitting — check val loss)

Red flags:
- Loss stuck above 2.5 after 100 steps: likely formatting issue in dataset or wrong chat template
- Loss explodes (NaN or >10): learning rate too high or bad batches
- Loss reaches 0.01: memorization, not generalization — reduce epochs

### 6. Weights & Biases Integration

`SFTTrainer` integrates with W&B automatically if you set `WANDB_PROJECT` in your environment. Log at minimum: loss, learning rate schedule, and generated samples at end of training.

```python
import os
os.environ["WANDB_PROJECT"] = "week-29-sft-tiny"
```

Every run should have a meaningful name: set `run_name="qwen-0.5b-sql-1k-epoch2"` in `SFTConfig`.

---

## Connections

**Builds on:** Week 28's conceptual overview of SFT; Phase 2's PyTorch training loops; Phase 3's tokenizer knowledge.

**Needed for:** Week 30 (you will swap in LoRA adapters on top of this same training setup). Week 33 (QLoRA uses this same SFTTrainer framework). Weeks 37–38 (sprint uses this exact pattern at 15K examples).

---

## Common Misconceptions / Pitfalls

- **"The SFTTrainer handles chat templates automatically."** Partially — you must preprocess data into the right format first. Use `tokenizer.apply_chat_template` explicitly.
- **"More training steps always help."** Not for 1K examples. 2 epochs is usually enough; the model will memorize past that.
- **"I can use the same script on any model."** False — each model has a different chat template, padding token, and EOS token. Qwen2.5's tokenizer needs `pad_token = eos_token` if no explicit pad token is set.
- **"Full SFT is better than LoRA SFT."** Not necessarily, and LoRA is much more memory-efficient. Full SFT is done this week only to give you a baseline.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read SFTTrainer docs + HuggingFace fine-tuning tutorial | 1h |
| Set up Colab Pro notebook, install packages | 30m |
| Download and format 1K SQL dataset | 1h |
| Write and debug the SFT training script | 2h |
| Run training, monitor W&B | 1h |
| Push model to HuggingFace Hub, commit to GitHub | 30m |
