# Week 57 Assignment Solutions

## Task 2 — Document Packing Snippet

```python
from transformers import AutoTokenizer
from datasets import Dataset
import json

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
MAX_SEQ_LEN = 2048
EOS = tokenizer.eos_token_id

all_ids = []
with open("cpt_corpus.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        ids = tokenizer.encode(doc["text"], add_special_tokens=False)
        all_ids.extend(ids + [EOS])

# Pack into sequences of exactly MAX_SEQ_LEN
packed = []
for i in range(0, len(all_ids) - MAX_SEQ_LEN, MAX_SEQ_LEN):
    packed.append({"input_ids": all_ids[i:i+MAX_SEQ_LEN]})

dataset = Dataset.from_list(packed)
dataset.push_to_hub("<your-handle>/postgres-cpt-corpus-packed", private=True)
print(f"Packing efficiency: {len(all_ids) / (len(packed)*MAX_SEQ_LEN):.3f}")
```

## Task 4 — CPT Training Script Key Config

```python
from unsloth import FastLanguageModel
from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    max_seq_length=2048,
    dtype=None,  # bf16 on H100
    load_in_4bit=False,  # full precision for CPT
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

training_args = TrainingArguments(
    output_dir="./cpt_output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,  # effective batch = 64
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    num_train_epochs=1,  # EXACTLY ONE EPOCH
    bf16=True,
    logging_steps=10,
    save_steps=200,
    report_to="wandb",
    run_name="week-57-cpt-full",
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(model=model, args=training_args,
                  train_dataset=dataset, data_collator=collator)
trainer.train()
```

---

## Common Gotchas

- **Forgetting to add EOS between documents.** Without EOS, the model learns to predict across document boundaries — "the next token after the last SQL line in the PostgreSQL docs is 'TimescaleDB' (the first word of the next document)." This is wrong. Always separate documents with EOS.
- **Training loss plateaus at 2.2+.** Usually means your corpus has too much boilerplate (nav menus, repeated headers). Re-run quality filtering.
- **RunPod instance not terminated.** H100 at $2.79/hr × 24 hours = $67 wasted. Set a spending cap AND manually verify termination.
- **CPT checkpoint is too large to push to HuggingFace (LFS issues).** Use `model.save_pretrained()` with `safe_serialization=True` to split into shards. Or use HuggingFace Hub's large file support.
- **Wikipedia perplexity spikes at step 200.** Learning rate too high. Reduce to 2e-5 and resume from last checkpoint.

---

## How to Verify You Did It Right

1. `cpt_corpus.jsonl` verified token count: at least 95M tokens
2. Packing efficiency > 95% (< 5% padding in packed sequences)
3. W&B shows smooth loss decrease: step 1 (~2.6) → step 763 (~1.8–2.0)
4. PostgreSQL held-out perplexity decreases at least 10% vs. base model
5. Wikipedia held-out perplexity change < 0.5 bits (no catastrophic forgetting)
6. HuggingFace model page shows CPT checkpoint with README explaining training
7. RunPod instance is TERMINATED (check your RunPod console)
