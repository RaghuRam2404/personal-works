# Week 31 — LoRA via the `peft` Library, Target Modules, and Rank Sweeps

## Learning Objectives

By the end of this week, you will be able to:

- Use `peft` to apply LoRA to any HuggingFace model in under 20 lines of code
- Enumerate the target_modules for Qwen2.5-Coder-1.5B and justify which ones to include
- Run a W&B sweep over rank r ∈ {8, 16, 32, 64} and interpret the resulting loss curves
- Explain the practical trade-off between rank, parameter count, training speed, and generalization on small datasets
- Identify when a LoRA rank is too high for a given dataset size

---

## Concepts

### 1. The `peft` Library: LoRA in Production

The `peft` library (Parameter-Efficient Fine-Tuning, by HuggingFace) wraps the math you implemented in Week 30 into a clean API. It handles:
- Replacing target linear layers with LoRA-augmented versions automatically
- Freezing non-adapter parameters
- Saving and loading only the adapter weights (small files: 10–100MB vs. the full 14GB model)
- Merging adapters for inference

The key configuration object is `LoraConfig`:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 1,543,168,000 || trainable%: 2.72
```

`get_peft_model` wraps the model in-place. The original weights become frozen. Only the adapter matrices are trainable.

### 2. Choosing `target_modules`

This is the most consequential LoRA hyperparameter after rank. Your choices:

**Option A: Attention only** — `["q_proj", "v_proj"]` (original LoRA paper default)
- Fewer parameters, less memory
- Misses MLP's role in knowledge retrieval

**Option B: All attention** — `["q_proj", "k_proj", "v_proj", "o_proj"]`
- Standard for chat models; covers the full attention block

**Option C: All linear layers** — add `["gate_proj", "up_proj", "down_proj"]`
- Empirically best for supervised tasks (Sebastian Raschka's finding)
- ~2–3x more parameters than Option A at the same rank

For your SQL task: use Option C. The MLP layers store the model's "factual" and "syntactic" knowledge about SQL — including them gives the optimizer more levers to shift SQL generation behavior.

To find the exact module names for any model, run:

```python
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        print(name)
```

For Qwen2.5 models, the typical names are:
- Attention: `model.layers.{i}.self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `model.layers.{i}.mlp.gate_proj`, `up_proj`, `down_proj`

In `peft`, you specify just the leaf name: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.

### 3. Rank Sweep: What to Expect

A rank sweep over r ∈ {8, 16, 32, 64} with fixed alpha = 2r (so scaling = 2.0), fixed LR, same data and epochs:

| Rank | Trainable params (7B model) | Memory overhead | Expected val loss on 5K examples |
|---|---|---|---|
| 8 | ~20M | Low | 1.3–1.5 |
| 16 | ~40M | Medium | 1.1–1.3 |
| 32 | ~80M | Medium-high | 1.0–1.2 (but watch for overfitting) |
| 64 | ~160M | High | Can be worse than rank 16 if dataset is small |

The key insight: on a 5K example dataset, rank 64 has enough capacity to memorize the training set but not enough regularization to generalize. Rank 16 often performs as well as rank 64 on held-out SQL while training faster and using less memory.

This is the classic bias-variance trade-off in a new guise: higher rank = more capacity = lower bias but higher variance.

### 4. Saving and Loading PEFT Adapters

One of peft's biggest practical advantages: adapter files are tiny.

```python
# Save only adapter weights
model.save_pretrained("./qwen-1.5b-sql-lora-r16")  # saves adapter_model.safetensors (~30MB)

# Load adapters onto a fresh base model
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B")
model = PeftModel.from_pretrained(base_model, "./qwen-1.5b-sql-lora-r16")
```

Push to HuggingFace Hub the same way: `model.push_to_hub("your-handle/model-name")`. Viewers download only the adapter file and apply it locally to the base model.

### 5. Interaction with SFTTrainer

`SFTTrainer` accepts a PEFT model directly. The training loop is identical to Week 29's full SFT, except now >99% of parameters are frozen:

```python
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(r=16, lora_alpha=32, ...)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

trainer = SFTTrainer(model=model, args=..., train_dataset=..., eval_dataset=...)
trainer.train()
```

### 6. Bias Parameter in LoraConfig

`bias="none"` (default) means bias terms are not adapted — only the weight matrices. Options:
- `"none"`: Only A and B are trained, biases frozen (recommended)
- `"all"`: Biases of all layers are also trained
- `"lora_only"`: Only the biases of LoRA-targeted layers are trained

For SQL fine-tuning, `bias="none"` is standard. Adapting biases adds minimal benefit and complicates adapter reuse.

---

## Connections

**Builds on:** Week 30's from-scratch LoRA implementation — peft does the same thing more robustly. Week 29's SFTTrainer setup — same training loop, just with a PEFT model.

**Needed for:** Week 33 (QLoRA = LoRA via peft on a 4-bit quantized base model). Week 34 (Unsloth replaces peft's LoRA with optimized kernels but the API is similar). Week 36 (DoRA, RSLoRA are configured via LoraConfig flags).

---

## Common Misconceptions / Pitfalls

- **"All target_modules values work for all models."** False — module names are model-specific. Always enumerate them for the exact model you are using.
- **"Higher rank always gives lower validation loss."** False — on small datasets, rank 32–64 overfits. Sweep and verify empirically.
- **"peft saves the full fine-tuned model."** No — `model.save_pretrained()` on a PEFT model saves only the adapter. To merge and save the full weights, call `model.merge_and_unload()` first.
- **"Changing alpha is equivalent to changing LR."** Not exactly — alpha scales the adapter output magnitude; LR scales the gradient step. They interact but are not interchangeable. Fix alpha = 2r and tune LR instead.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read PEFT docs + Raschka's article on target_modules | 1h |
| Set up Colab Pro, install peft, format 5K dataset | 1h |
| Write LoRA fine-tuning script with peft + SFTTrainer | 1.5h |
| Run rank sweep (4 runs) — can run in parallel if 2 GPUs available | 2h |
| Analyze W&B sweep results, write comparison report | 1h |
| Commit to GitHub | 30m |
