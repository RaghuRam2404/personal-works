# Week 31 Assignment — LoRA via peft, Rank Sweep

## Setup Checklist

- [ ] Colab Pro active (A100 or T4)
- [ ] `pip install peft trl transformers datasets accelerate wandb`
- [ ] 5K SQL training examples formatted from Week 29 (reuse your dataset)
- [ ] W&B project `week-31-lora-sweep` created
- [ ] HuggingFace write token set

---

## Task 1 — Enumerate Target Modules

**Goal:** Know exactly which linear layers exist in Qwen2.5-Coder-1.5B before configuring LoRA.

**Requirements:**
- Load `Qwen/Qwen2.5-Coder-1.5B` (CPU is fine for this task)
- Print all module names that are `nn.Linear` instances
- In `week31_target_modules.md`, list: (a) all distinct layer name patterns, (b) which ones you will target with LoRA and why, (c) which ones you will skip and why
- Identify whether Qwen2.5-Coder-1.5B uses full-rank attention (no GQA) or grouped-query attention (GQA) — this affects k_proj and v_proj dimensions

**Deliverable:** `week31_target_modules.md` committed.

---

## Task 2 — LoRA Fine-Tune at Rank 16 (Baseline)

**Goal:** Establish your rank-16 baseline before the sweep.

**Requirements:**
- Model: `Qwen/Qwen2.5-Coder-1.5B`
- Dataset: 5,000 training examples, 200 held-out eval examples (SQL pairs from Week 29 dataset or similar)
- LoraConfig: `r=16`, `lora_alpha=32`, all 7 linear layer types as target_modules, `lora_dropout=0.05`, `bias="none"`, `task_type="CAUSAL_LM"`
- SFTConfig: 2 epochs, `per_device_train_batch_size=4`, `gradient_accumulation_steps=4` (effective batch 16), `learning_rate=2e-4`, `lr_scheduler_type="cosine"`, `warmup_ratio=0.05`, `max_seq_length=512`, `packing=True`
- Log to W&B run name: `qwen-1.5b-sql-r16`
- Record: trainable parameter count, % of total, final train loss, final eval loss, training time

**Deliverable:** `week31_results.md` with rank-16 baseline metrics. GitHub commit: `week-31-lora-r16`.

---

## Task 3 — Rank Sweep r ∈ {8, 32, 64}

**Goal:** Empirically compare 3 additional ranks against your rank-16 baseline.

**Requirements:**
- Run 3 more training runs with identical settings except rank: r=8, r=32, r=64
- Keep `lora_alpha = 2 * r` for all runs (so scaling = 2.0 is constant)
- Log each run to W&B with run names: `qwen-1.5b-sql-r8`, `qwen-1.5b-sql-r32`, `qwen-1.5b-sql-r64`
- For each run record: trainable params, final train loss, final eval loss, training time (steps/sec)

**Deliverable:** All 4 runs visible in W&B. W&B sweep URL in `week31_results.md`.

**Hints:**
- On Colab Pro T4, each run at r=16 should take 20–40 minutes on 5K examples
- If time is tight, run r=8 and r=64 (the extremes) and skip r=32

---

## Task 4 — Analysis and Recommendation

**Goal:** Draw conclusions from the sweep.

**Requirements:**
- Create a comparison table in `week31_results.md`:

| Rank | alpha | Trainable Params | Train Loss | Eval Loss | Steps/sec |
|---|---|---|---|---|---|
| 8 | 16 | ... | ... | ... | ... |
| 16 | 32 | ... | ... | ... | ... |
| 32 | 64 | ... | ... | ... | ... |
| 64 | 128 | ... | ... | ... | ... |

- Write a 2–3 paragraph analysis answering: Which rank minimizes eval loss? Is there a rank where eval loss gets worse? What does this suggest about the optimal rank for your 5K dataset? What rank would you use for a 15K dataset (justify)?

**Deliverable:** Analysis section in `week31_results.md`.

---

## Task 5 — Save and Load Adapter

**Goal:** Verify the peft save/load workflow.

**Requirements:**
- After training your rank-16 model, save the adapter: `model.save_pretrained("./qwen-1.5b-sql-r16")`
- Verify the saved files: list the directory and confirm `adapter_model.safetensors` and `adapter_config.json` exist
- Load the adapter onto a fresh base model using `PeftModel.from_pretrained`
- Run inference on 3 held-out examples and confirm output matches the original fine-tuned model
- Note the adapter file size in `week31_results.md`

**Deliverable:** Section in `week31_results.md` with adapter file size and inference confirmation.

---

## Stretch Goals

- Sweep `target_modules`: run rank 16 with just `["q_proj", "v_proj"]` vs. all 7 layers. Does covering all layers improve eval loss on your dataset?
- Push your best adapter to HuggingFace Hub
- Try `lora_dropout=0.0` vs `lora_dropout=0.1` — does dropout help or hurt on 5K examples?
