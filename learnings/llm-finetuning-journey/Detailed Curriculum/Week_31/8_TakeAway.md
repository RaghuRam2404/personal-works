# Week 31 TakeAway — peft LoRA, Target Modules, Rank Sweeps

**One-liner:** peft wraps Week 30's LoRA math; enumerate target_modules explicitly, set alpha=2r, sweep rank 8→64 to find the overfitting boundary.

---

## Minimal peft LoRA Setup

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,              # alpha = 2*r is standard
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()  # verify < 5% trainable
```

---

## Enumerate Linear Layers (Any Model)

```python
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name.split(".")[-1])  # leaf name
# Don't include: lm_head, embed_tokens
```

---

## Save / Load Adapter

```python
# Save (tiny file ~30-100MB)
model.save_pretrained("./my-adapter")

# Load onto fresh base
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B", ...)
model = PeftModel.from_pretrained(base, "./my-adapter")
```

---

## Rank vs Dataset Size (Rule of Thumb)

| Dataset | Recommended Rank |
|---|---|
| < 1K examples | 4–8 |
| 1K–10K | 16 |
| 10K–100K | 32–64 |
| > 100K | 64–128 |

---

## Numbers to Remember

- Rank 16, all 7 layers, 7B model → ~40M trainable = 0.25% of total
- Rank 16, all 7 layers, 1.5B model → ~20M trainable = 1.3% of total
- Adapter file size: rank × model_size / base_model_size × ~10MB estimate
- `lora_alpha = 2 * r` → scaling = 2.0 (standard)

---

## Decision Rules

- Always enumerate target_modules for the specific model — don't copy from another model
- Do NOT include `lm_head` in target_modules
- If eval loss rises while training — reduce rank or add lora_dropout
- If val loss equals rank-16 performance at rank-64 — use rank 16 (same result, less memory)

---

## Red Flags

- `trainable% > 10%` → you forgot `get_peft_model` or included too many modules
- Loading adapter gives base model outputs → didn't use `PeftModel.from_pretrained`
- Rank 64 eval loss worse than rank 16 → overfitting, reduce rank
- Adapter file is 14GB → you accidentally saved the full merged model
