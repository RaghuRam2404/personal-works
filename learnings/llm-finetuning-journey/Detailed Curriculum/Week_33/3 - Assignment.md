# Week 33 Assignment — QLoRA Fine-Tune Qwen2.5-Coder-7B

## Setup Checklist

- [ ] Colab Pro with A100 runtime selected (required for 7B training)
- [ ] `pip install trl peft transformers bitsandbytes datasets accelerate wandb`
- [ ] 5K training examples formatted (reuse Week 29/31 dataset)
- [ ] 100-example held-out test set from Week 32 (`held_out_test.json`)
- [ ] W&B project `week-33-qlora-7b` created
- [ ] HuggingFace write token set

---

## Task 1 — Write the QLoRA Training Script

**Goal:** Combine bitsandbytes 4-bit loading with peft LoRA into a working 7B training pipeline.

**Requirements:**
- Model: `Qwen/Qwen2.5-Coder-7B`
- BitsAndBytesConfig: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`, `bnb_4bit_use_double_quant=True`
- Set `model.config.use_cache = False` immediately after loading
- LoraConfig: `r=16`, `lora_alpha=32`, all 7 linear layer types, `lora_dropout=0.05`
- SFTConfig: `num_train_epochs=2`, `per_device_train_batch_size=4`, `gradient_accumulation_steps=4`, `optim="paged_adamw_8bit"`, `learning_rate=2e-4`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.05`, `gradient_checkpointing=True`, `max_seq_length=512`, `packing=True`
- W&B project: `week-33-qlora-7b`, run name: `qwen-coder-7b-sql-5k-r16`
- After `model = get_peft_model(model, lora_config)`, call `model.print_trainable_parameters()` and record the output

**Deliverable:** `train_qlora_7b.py` committed to GitHub. W&B run URL in `week33_results.md`.

---

## Task 2 — Run Training and Monitor

**Goal:** Execute the full 7B training run and confirm it completes without errors.

**Requirements:**
- Training must complete (2 epochs on 5K examples should take 20–40 minutes on A100)
- Record in `week33_results.md`:
  - GPU type (from `!nvidia-smi`)
  - VRAM usage during training (peak, from W&B system metrics or `torch.cuda.max_memory_allocated()`)
  - Final train loss
  - Final eval loss (run on 200 eval examples from your domain dataset)
  - Training time (minutes)
- Save checkpoint at end: `model.save_pretrained("./qwen-coder-7b-sql-qlora-r16")`

**Deliverable:** Training complete, checkpoint saved, metrics in `week33_results.md`.

---

## Task 3 — Evaluate on Held-Out Test Set

**Goal:** Measure your model's SQL generation quality on data it has never seen.

**Requirements:**
- Load your fine-tuned adapter onto the base model:
  ```python
  base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B", quantization_config=bnb_config, device_map="auto")
  model = PeftModel.from_pretrained(base, "./qwen-coder-7b-sql-qlora-r16")
  ```
- Run inference on all 100 examples in `held_out_test.json` using greedy decoding (`do_sample=False`)
- Compute exact-match accuracy: what % of generated SQL exactly matches the expected SQL?
- Also compute: what % of generated SQL is syntactically valid SQL (contains SELECT and FROM keywords at minimum)?
- Record: exact match %, valid SQL %, 5 example (question, expected, generated) triples in `week33_results.md`

**Acceptance criteria:** At least 30% exact match. At least 70% syntactically valid SQL.

**Deliverable:** Evaluation results section in `week33_results.md`.

---

## Task 4 — Push to HuggingFace Hub

**Goal:** Share your first 7B fine-tune with the world.

**Requirements:**
- Push the adapter to HuggingFace: `<your-handle>/qwen-coder-7b-postgres-v1`
- Include a model card documenting: base model, training data size, LoRA config (r, alpha, target_modules), eval results
- Commit to GitHub with message: `week-33-qlora-7b`

**Deliverable:** HuggingFace model link in `week33_results.md`.

---

## Task 5 — Compare to Base Model

**Goal:** Quantify the improvement from fine-tuning.

**Requirements:**
- Load the base `Qwen/Qwen2.5-Coder-7B` (no adapter) and run it on the same 100 held-out examples
- Record: base model exact match %, base model valid SQL %
- Create a comparison table in `week33_results.md`:

| Model | Exact Match | Valid SQL % |
|---|---|---|
| Qwen2.5-Coder-7B base | % | % |
| QLoRA fine-tuned (yours) | % | % |

**Deliverable:** Comparison table in `week33_results.md`.

---

## Stretch Goals

- Run training with `gradient_checkpointing=False` on a 5-example batch to measure peak VRAM — confirm gradient checkpointing reduces it
- Try `optim="paged_adamw_32bit"` vs `optim="paged_adamw_8bit"` — does final loss differ?
- Run inference on 3 complex PostgreSQL questions (involving JOINs and window functions) and qualitatively assess the model's ability to generate PostgreSQL-specific SQL
