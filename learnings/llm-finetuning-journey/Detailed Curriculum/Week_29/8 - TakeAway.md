# Week 29 TakeAway — Full SFT on a Tiny Model

**One-liner:** SFTTrainer + chat template + 1K SQL pairs = your first working SQL fine-tune in under 15 minutes.

---

## Minimal SFT Setup

```python
from trl import SFTTrainer, SFTConfig
import torch, os

os.environ["WANDB_PROJECT"] = "week-29-sft-tiny"

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    args=SFTConfig(output_dir="./out", num_train_epochs=2,
                   per_device_train_batch_size=4, learning_rate=2e-5,
                   dataset_text_field="text", max_seq_length=512),
    train_dataset=train_ds, eval_dataset=eval_ds,
)
trainer.train()
```

---

## Chat Template Pattern (Qwen2.5)

```python
messages = [
    {"role": "system", "content": "You are a PostgreSQL expert..."},
    {"role": "user", "content": f"Schema:\n{schema}\nQuestion:\n{question}"},
    {"role": "assistant", "content": sql},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

---

## Inference Pattern (after SFT)

```python
messages = [{"role": "user", "content": f"Schema:\n{schema}\nQuestion:\n{question}"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=True, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=200, do_sample=False)
```

---

## Memory at a Glance (full SFT, AdamW)

| Model | Weights (fp16) | Grads | AdamW | Total |
|---|---|---|---|---|
| 0.5B | 1 GB | 1 GB | 4 GB | ~6 GB |
| 7B | 14 GB | 14 GB | 56 GB | ~84 GB |

---

## Numbers to Remember

- 1K examples, 2 epochs: ~8–15 min on T4
- Expected final train loss: 0.5–1.0 on 1K SQL pairs
- Eval loss should be checked every 100 steps; rising eval loss = overfitting
- `max_grad_norm=1.0` — always set this to prevent loss spikes

---

## Decision Rules

- If `tokenizer.pad_token is None` → set it to `eos_token` before training
- If train loss < 0.3 but eval loss > 1.5 → overfitting, reduce epochs
- If loss spikes to NaN → add `max_grad_norm=1.0`, reduce LR
- If throughput is low on short sequences → add `packing=True`

---

## Red Flags During Training

- Loss spike then NaN → gradient explosion, add gradient clipping
- Loss stuck at 2.5+ → chat template wrong, check formatted example
- Eval loss rising while train loss falling → overfitting, use early stopping
- Model outputs prompt text at inference → forgot `add_generation_prompt=True`
