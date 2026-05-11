# Week 31 Assignment Solutions

## Task 1 — Target Module Enumeration: Key Snippet

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B", torch_dtype=torch.float16
)

linear_layers = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        leaf_name = name.split(".")[-1]
        linear_layers.add(leaf_name)

print(sorted(linear_layers))
# Expected: ['down_proj', 'gate_proj', 'k_proj', 'lm_head', 'o_proj',
#            'q_proj', 'up_proj', 'v_proj']
```

Note: `lm_head` is the output projection; do NOT include it in LoRA target_modules for standard SFT. Target: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.

---

## Task 2 — LoRA Fine-Tune Script: Key Snippet

```python
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os

os.environ["WANDB_PROJECT"] = "week-31-lora-sweep"

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="./qwen-1.5b-sql-r16",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        eval_steps=50,
        evaluation_strategy="steps",
        report_to="wandb",
        run_name="qwen-1.5b-sql-r16",
        dataset_text_field="text",
        max_seq_length=512,
        packing=True,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
model.save_pretrained("./qwen-1.5b-sql-r16")
```

---

## Expected Results

| Rank | Trainable Params | Train Loss | Eval Loss |
|---|---|---|---|
| 8 | ~10M (0.65%) | 0.8–1.0 | 1.2–1.5 |
| 16 | ~20M (1.30%) | 0.7–0.9 | 1.1–1.4 |
| 32 | ~40M (2.60%) | 0.6–0.8 | 1.1–1.4 |
| 64 | ~80M (5.20%) | 0.4–0.6 | 1.2–1.6 (may overfit) |

Expect rank 16 or 32 to give the best eval loss on 5K examples. Rank 64 may show lower train loss but higher eval loss — classic overfitting sign.

**Adapter file size (rank 16, all 7 layer types):** ~30–60MB for Qwen2.5-Coder-1.5B.

---

## Common Gotchas

- **`lm_head` included in target_modules**: Do not add the output projection. It changes the logit distribution in ways that can be unstable.
- **Trainable % unexpectedly high (>5%)**: You may have forgotten to freeze with `get_peft_model` — it should handle freezing. If using manual setup, double-check freeze logic.
- **Eval loss not improving from baseline**: Ensure `packing=True` and `max_seq_length=512` — without packing on short SQL examples, you waste compute on padding.
- **W&B runs not in the same sweep**: Use the same `WANDB_PROJECT` name; create a sweep via W&B UI after runs complete for side-by-side comparison.

---

## How to Verify You Did It Right

- `model.print_trainable_parameters()` shows ~0.5–2% trainable for rank 8–32 on 1.5B model
- All 4 runs appear in your W&B project with clearly labeled run names
- Adapter file is under 100MB (much smaller than base model)
- Eval loss follows expected pattern: rank 8 slightly higher than rank 16, rank 64 potentially higher than rank 16 or 32 on 5K examples
- Loading adapter onto fresh base model produces identical outputs to original trained model
