# Week 43 Assignment Solutions

## Task 1 — DPO Derivation Key Steps

```
Step 1: KL-constrained RL objective
  max_π E_{x,y~π}[r(x,y)] - β · KL(π(y|x) || π_ref(y|x))

Step 2: Expand KL and write as single expectation
  = max_π E_{x,y~π}[r(x,y) - β·log(π(y|x)/π_ref(y|x))]

Step 3: This is maximized pointwise for each (x,y). Setting functional
  derivative to zero (or using the known form of this variational problem):
  π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)
  where Z(x) = Σ_y π_ref(y|x)·exp(r(x,y)/β)   [partition function]

Step 4: Invert to express r
  log π*(y|x) = log π_ref(y|x) + r(x,y)/β - log Z(x)
  r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)

Step 5: Substitute into Bradley-Terry preference loss
  Since Z(x) cancels in r(y_w) - r(y_l):
  L_BT = -log σ(r(y_w) - r(y_l))
       = -log σ(β·log(π*(y_w|x)/π_ref(y_w|x)) - β·log(π*(y_l|x)/π_ref(y_l|x)))

Step 6: Replace π* with learned π_θ (this is the key step — we train π_θ to 
  satisfy this preference loss, which implies π_θ ≈ π*)
  L_DPO(π_θ) = -log σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x)))
```

---

## Task 2 — DPO Training Key Snippet

```python
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                              torch_dtype="auto")
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                  torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear")
model = get_peft_model(model, lora_config)

dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized",
                       split="train_prefs")

training_args = DPOConfig(
    beta=0.1,
    learning_rate=5e-7,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_prompt_length=512,
    max_completion_length=256,
    num_train_epochs=1,
    report_to="wandb",
    run_name="week-43-dpo",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

**Expected output:**
- `rewards/chosen` should increase from ~0 to positive values
- `rewards/rejected` should decrease and become negative
- `reward_margin` should grow from ~0 to 0.5–2.0 over training
- Loss should decrease from ~0.69 (random log(0.5)) toward 0.4–0.5

**Common gotchas:**
- If `reward_margin` stays at 0: the reference model and training model are the same object (wrong — they must be separate instances)
- Very low β (0.01) causes the model to change rapidly and may produce degenerate outputs — increase β to 0.1 or 0.2
- If loss decreases but `rewards/chosen` does not increase: check that the dataset is correctly formatted with `chosen` and `rejected` keys
- If memory OOM: reduce `max_completion_length` to 128 or use `bf16=True`
- `DPOTrainer` expects the reference model to be a separate `AutoModelForCausalLM`, not a PEFT model — do not apply LoRA to `ref_model`

---

## How to Verify You Did It Right

1. `rewards/chosen` > `rewards/rejected` at step 500. If not, β is too high or LR is too low.
2. `reward_margin` > 0.5 at step 500. This is the key health metric for DPO.
3. Generate a completion from the DPO model and the base model for the same prompt. The DPO model should generally be more helpful and less likely to refuse.
4. Can you write the DPO loss from memory: `-log σ(β·(log π_θ(y_w|x) - log π_ref(y_w|x)) - β·(log π_θ(y_l|x) - log π_ref(y_l|x)))`?
