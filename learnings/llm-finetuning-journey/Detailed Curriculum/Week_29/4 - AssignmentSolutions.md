# Week 29 Assignment Solutions

## Task 1 — Dataset Formatting: Key Snippet

```python
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("b-mc2/sql-create-context", split="train")
dataset = dataset.shuffle(seed=42).select(range(1100))  # 1K train + 100 eval

def format_example(example):
    messages = [
        {"role": "system", "content": "You are a PostgreSQL expert. Given a table schema and a question, write a valid SQL query."},
        {"role": "user", "content": f"Schema:\n{example['context']}\nQuestion:\n{example['question']}"},
        {"role": "assistant", "content": example['answer']},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

dataset = dataset.map(format_example)
train_dataset = dataset.select(range(1000))
eval_dataset = dataset.select(range(1000, 1100))
```

**Expected output of `tokenizer.apply_chat_template` call:**
```
<|im_start|>system
You are a PostgreSQL expert...<|im_end|>
<|im_start|>user
Schema:
CREATE TABLE employee (id INT, name VARCHAR, salary FLOAT);
Question:
What is the average salary?<|im_end|>
<|im_start|>assistant
SELECT AVG(salary) FROM employee<|im_end|>
```

---

## Task 2 — Training Script: Key Snippet

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM
import torch, os

os.environ["WANDB_PROJECT"] = "week-29-sft-tiny"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="./qwen-0.5b-sft",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        evaluation_strategy="steps",
        report_to="wandb",
        run_name="qwen-0.5b-sql-1k-epoch2",
        dataset_text_field="text",
        max_seq_length=512,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
trainer.push_to_hub("<your-handle>/qwen-0.5b-postgres-sft-v1")
```

---

## Expected Output

- **Initial train loss:** ~2.2–2.8 (depends on how SQL-heavy the base model's pretraining was)
- **Final train loss after 2 epochs on 1K:** ~0.5–1.0
- **Final eval loss:** ~1.2–1.8 (higher than train loss is normal; gap indicates some overfitting)
- **Training time:** 8–15 minutes on T4, 3–6 minutes on A100

---

## Common Gotchas

- **`pad_token` is `None`**: Qwen2.5-0.5B may not have a pad token. Always set `tokenizer.pad_token = tokenizer.eos_token` before training.
- **Loss stuck at 2.5+**: Your chat template is wrong. Print one formatted example and verify the `<|im_start|>assistant` token appears before the SQL.
- **CUDA out of memory**: Reduce `per_device_train_batch_size` to 2 or add `gradient_checkpointing=True` in `SFTConfig`.
- **W&B not logging**: Run `wandb login` in a Colab cell before starting training.
- **Model generates prompt instead of SQL**: Add `add_generation_prompt=True` during inference (not training). During inference: `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")`.

---

## How to Verify You Did It Right

- Final train loss is below 1.5 after 2 epochs
- W&B run shows a monotonically (or mostly) decreasing loss curve
- At least 3 of 5 held-out test examples produce syntactically valid SQL (e.g., starts with SELECT, has FROM clause)
- The base model comparison shows the base `Qwen2.5-0.5B` mostly produces rambling text or incorrect format, while your fine-tuned model produces SQL
- Model is accessible at `huggingface.co/<your-handle>/qwen-0.5b-postgres-sft-v1`
