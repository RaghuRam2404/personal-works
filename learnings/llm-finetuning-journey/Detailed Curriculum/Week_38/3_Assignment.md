# Week 38 Assignment — Production QLoRA Fine-Tune on 15K

## Setup Checklist

- [ ] Colab Pro A100 runtime selected
- [ ] `train_15k.jsonl` and `val_500.jsonl` from Week 37 uploaded to Colab or accessible via HuggingFace dataset
- [ ] `held_out_test.json` (100 examples from Week 32) accessible
- [ ] Unsloth, peft, trl, bitsandbytes installed (reuse Week 34 environment)
- [ ] W&B project `week-38-qlora-15k` created
- [ ] HuggingFace write token set

---

## Task 1 — Prepare Training Script

**Goal:** Assemble the final production training script using all best practices from Phase 4.

**Requirements:**
- Model: `Qwen/Qwen2.5-Coder-7B`
- Quantization: NF4, double quant, BF16 compute dtype
- Adapter: DoRA (or LoRA from your Week 36 decision), rank 16, alpha 32, all 7 linear layer types
- Training config (from Week 35 sweep): LR=2e-4, cosine scheduler, warmup_ratio=0.05, 2 epochs, packing=True, max_seq_length=512, effective batch 16 (4 × 4), optim=paged_adamw_8bit
- Add `max_grad_norm=1.0` for safety
- Add `load_best_model_at_end=True` with eval every 100 steps
- W&B project `week-38-qlora-15k`, run name `postgres-sqlcoder-7b-v1`
- Load dataset from `train_15k.jsonl` and `val_500.jsonl`

**Deliverable:** `train_15k_qlora.py` committed before running.

---

## Task 2 — Run Training

**Goal:** Execute the full training run.

**Requirements:**
- Run the training script and allow it to complete
- Record in `week38_results.md`:
  - Training start time, end time, total duration
  - Final train loss and eval loss
  - Peak VRAM usage
  - Steps per second (from W&B)
  - Any issues encountered and how you resolved them
- Save checkpoint: `model.save_pretrained("./postgres-sqlcoder-7b-v1")`

**Deliverable:** Training complete, checkpoint saved, W&B run URL in `week38_results.md`.

---

## Task 3 — Evaluate Against Baselines

**Goal:** Quantify the improvement of your fine-tuned model.

**Requirements:**
- Run inference on all 100 examples in `held_out_test.json` using three models:
  1. `Qwen/Qwen2.5-Coder-7B` base model (in NF4)
  2. Your Week 33 model (5K training, from HuggingFace Hub)
  3. Your Week 38 model (15K training, this run)
- For each model, record: exact match %, valid SQL %
- Create comparison table in `week38_results.md`:

| Model | Training Data | Exact Match | Valid SQL |
|---|---|---|---|
| Qwen2.5-Coder-7B base | None | % | % |
| Week 33 model | 5K examples | % | % |
| Week 38 model (v1) | 15K examples | % | % |

- Qualitative analysis: for 5 examples where Week 38 model is correct but Week 33 is wrong, describe what the Week 38 model learned that Week 33 did not

**Deliverable:** Complete comparison table and qualitative analysis in `week38_results.md`.

---

## Task 4 — Push to HuggingFace Hub

**Goal:** Publish your first production SQL model.

**Requirements:**
- Push adapter to: `<your-handle>/postgres-sqlcoder-7b-v1`
- Create a comprehensive model card covering all fields from the Curriculum.md template
- Tag with: `text2sql`, `postgresql`, `sql`, `qwen2.5`, `qlora`, `lora`
- Verify: model is publicly accessible and model card shows correctly on HuggingFace
- Commit to GitHub with message: `week-38-qlora-15k`

**Deliverable:** HuggingFace model URL in `week38_results.md`.

---

## Task 5 — Error Analysis

**Goal:** Understand where your model fails to guide future improvements.

**Requirements:**
- For the 100 held-out examples, categorize each incorrect prediction into one of:
  - Wrong table referenced
  - Wrong column referenced
  - Wrong JOIN type/condition
  - Wrong aggregation or GROUP BY
  - Missing WHERE clause or wrong filter
  - Other structural error
  - Model refused / generated garbage
- Create a frequency table of error types in `week38_error_analysis.md`
- Write 2–3 paragraphs: what are the most common failure modes, and what data additions in a future v2 dataset would target these failures?

**Deliverable:** `week38_error_analysis.md` committed.

---

## Stretch Goals

- Run the same 100 evaluation examples through GPT-4o-mini or Claude 3.5 Sonnet (via API) — compare their exact match % to your 7B model
- Try a 3rd epoch on the same 15K examples — does eval loss improve or rise?
- Generate a GGUF version of the merged model for local inference on your Mac: `pip install llama-cpp-python`
