# Week 29 Assignment — Full SFT on Qwen2.5-0.5B

## Setup Checklist

- [ ] Colab Pro active (needed for A100/T4 GPU)
- [ ] HuggingFace account with write token (for model push)
- [ ] Weights & Biases account (free tier is fine)
- [ ] Packages: `pip install trl transformers datasets peft accelerate wandb`
- [ ] Dataset: [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) or Spider — load 1K rows for training, 100 rows held out for evaluation

---

## Task 1 — Format the Dataset

**Goal:** Convert a raw SQL dataset into SFTTrainer-compatible conversational format using Qwen2.5's chat template.

**Requirements:**
- Load 1,000 training examples and 100 held-out test examples
- Format each example as a `messages` list: system prompt + user (schema + question) + assistant (SQL answer)
- Call `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)` on one example and print the raw string — confirm `<|im_start|>` and `<|im_end|>` tokens appear correctly
- Ensure no training example exceeds 512 tokens (filter or truncate)
- Save the formatted dataset to disk: `dataset.save_to_disk("./sql_sft_1k")`

**Deliverable:** `prepare_dataset.py` committed to GitHub. Print output showing one formatted example.

**Hints:**
- `sql-create-context` has columns: `answer` (SQL), `question`, `context` (CREATE TABLE statements). Use `context` as the schema in your user message.
- Set `tokenizer.pad_token = tokenizer.eos_token` if `tokenizer.pad_token` is `None`.

---

## Task 2 — Write the SFT Training Script

**Goal:** Train `Qwen2.5-0.5B` with `SFTTrainer` for 2 epochs on your 1K dataset.

**Requirements:**
- Load `Qwen/Qwen2.5-0.5B` with `torch_dtype=torch.bfloat16`
- Use `SFTConfig` with:
  - `num_train_epochs=2`
  - `per_device_train_batch_size=4`
  - `gradient_accumulation_steps=2` (effective batch size 8)
  - `learning_rate=2e-5`
  - `lr_scheduler_type="cosine"`
  - `warmup_ratio=0.1`
  - `logging_steps=10`
  - `save_steps=100`
  - `evaluation_strategy="steps"` with `eval_steps=100`
  - `report_to="wandb"`
- Set W&B project name to `week-29-sft-tiny`
- W&B run name to `qwen-0.5b-sql-1k-epoch2`
- Use `dataset_text_field="text"` or pass `messages` format — check SFTTrainer docs for your trl version

**Deliverable:** `train_sft.py` committed. Training run linked in your commit message (W&B URL).

---

## Task 3 — Run Training and Log Metrics

**Goal:** Execute the full training run and record results.

**Requirements:**
- Training must complete without crashing (2 epochs on 1K examples should take <15 minutes on T4)
- Record the final train loss and final eval loss in `week29_results.md`
- Log at least one generated example: after training, run inference on 5 held-out examples and paste the model's SQL output into `week29_results.md`
- Include the W&B run URL in `week29_results.md`

**Deliverable:** `week29_results.md` with: final train loss, final eval loss, 5 sample generations, W&B URL.

---

## Task 4 — Push Model to HuggingFace

**Goal:** Share your first fine-tuned model.

**Requirements:**
- Push the trained model to HuggingFace Hub: `<your-handle>/qwen-0.5b-postgres-sft-v1`
- Include a model card (README on HuggingFace) with: what data was used, what the model is for, how to run inference
- Commit final code with message: `week-29-sft-tiny`

**Deliverable:** Public HuggingFace model link in `week29_results.md`.

**Hints:**
- Use `trainer.push_to_hub()` after training completes
- Alternatively: `model.push_to_hub(repo_name)` and `tokenizer.push_to_hub(repo_name)`

---

## Task 5 — Qualitative Evaluation

**Goal:** Verify the fine-tuned model produces more SQL-like output than the base model.

**Requirements:**
- Run the same 5 held-out questions through both: the base `Qwen/Qwen2.5-0.5B` (without fine-tuning) and your fine-tuned model
- For each question, note whether: (a) base model produces SQL, (b) fine-tuned model produces SQL, (c) fine-tuned model produces syntactically valid SQL
- Write a 2-paragraph qualitative assessment in `week29_results.md`

**Deliverable:** Added section in `week29_results.md`.

---

## Stretch Goals

- Add a `compute_metrics` function to `SFTTrainer` that computes exact-match SQL accuracy on the eval set
- Run training with and without `packing=True` in `SFTConfig` — compare throughput (steps/second)
- Try a slightly larger model: `Qwen/Qwen2.5-1.5B`. Does loss converge faster or slower than 0.5B on the same data?
