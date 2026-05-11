# Week 33 Assignment Solutions

## Task 1 — Complete QLoRA Script: Key Snippet

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch, os

os.environ["WANDB_PROJECT"] = "week-33-qlora-7b"

MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: trainable params: 41,943,040 || all params: ~7.24B || trainable%: 0.58%

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
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
        logging_steps=10,
        eval_steps=100,
        evaluation_strategy="steps",
        report_to="wandb",
        run_name="qwen-coder-7b-sql-5k-r16",
        dataset_text_field="text",
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
model.save_pretrained("./qwen-coder-7b-sql-qlora-r16")
```

---

## Expected Training Results (A100 40GB, 5K examples)

- **Trainable parameters:** ~42M (0.58% of 7.24B)
- **VRAM peak:** 18–28 GB (depends on batch size, gradient checkpointing, packing)
- **Training time:** 25–45 minutes for 2 epochs on 5K examples
- **Final train loss:** 0.6–1.0
- **Final eval loss:** 1.0–1.5

---

## Task 3 — Evaluation Script: Key Snippet

```python
from peft import PeftModel
import json, re

def evaluate_sql(model, tokenizer, test_path, device="cuda"):
    with open(test_path) as f:
        test_data = json.load(f)
    
    exact_match, valid_sql = 0, 0
    for item in test_data:
        messages = [
            {"role": "system", "content": "You are a PostgreSQL expert..."},
            {"role": "user", "content": f"Schema:\n{item['schema']}\nQuestion:\n{item['question']}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output = model.generate(inputs, max_new_tokens=200, do_sample=False)
        generated = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        
        if generated.strip().lower() == item["expected_sql"].strip().lower():
            exact_match += 1
        if "select" in generated.lower() and "from" in generated.lower():
            valid_sql += 1
    
    n = len(test_data)
    print(f"Exact match: {exact_match/n:.1%} | Valid SQL: {valid_sql/n:.1%}")
```

**Expected results:**
- Base model: 5–15% exact match, 60–80% valid SQL
- QLoRA fine-tuned: 30–55% exact match, 85–95% valid SQL
- The gap confirms fine-tuning is working

---

## Common Gotchas

- **`model.config.use_cache = False` forgotten**: Get `ValueError` when enabling gradient checkpointing. Set it immediately after loading.
- **A100 not selected**: Request A100 runtime in Colab under "Runtime → Change runtime type → A100". T4 may OOM for 7B QLoRA training.
- **`device_map="auto"` conflicts with DataParallel**: Use `device_map="auto"` for single-GPU Colab; don't set CUDA_VISIBLE_DEVICES manually.
- **Evaluation generates prompt instead of SQL**: Ensure `add_generation_prompt=True` during inference, not during training.
- **HuggingFace push fails**: Verify HF_TOKEN is set: `from huggingface_hub import login; login(token="hf_...")`

---

## How to Verify You Did It Right

- `model.print_trainable_parameters()` shows ~0.58% trainable (for rank 16)
- Peak VRAM is under 35GB on A100 (usually 18–28GB)
- Fine-tuned model exact match is at least 2–3x higher than base model
- `qwen-coder-7b-postgres-v1` adapter appears on HuggingFace Hub
- Adapter file is under 200MB (not 14GB)
