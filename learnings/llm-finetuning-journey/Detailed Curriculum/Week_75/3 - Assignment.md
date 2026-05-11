# Week 75 Assignment — Base Model Comparison

## Setup Checklist

- [ ] RunPod A100-40GB (or 2× A100) provisioned — budget ~$15 for 10 GPU-hours
- [ ] All three candidate models downloaded or accessible via Hub
- [ ] Your v3 SFT dataset at `data/sqlcoder_v3_train` (from Week 58)
- [ ] Custom-200 benchmark at `data/custom_200.json`
- [ ] W&B project `week75-base-model-comparison` created

## Task 1: Chat Template Verification

**Goal:** Confirm that each model's training data is formatted correctly before wasting GPU time.

**Requirements:**
- [ ] For each model, run: `python prepare_data.py --model <model_name> --output data/sft_<shortname>.jsonl`
- [ ] Print and manually inspect the first 3 examples from each prepared dataset
- [ ] Verify for each: (a) the system prompt appears in the correct position, (b) the user turn contains the schema and question, (c) the assistant turn contains only SQL (no explanation), (d) the end-of-turn token matches the model's expected stop token
- [ ] For Llama 3.1: confirm `<|eot_id|>` appears as the stop token
- [ ] For Gemma 2: confirm `<end_of_turn>` appears as the stop token
- [ ] For Qwen (R1-Distill): confirm `<|im_end|>` appears as the stop token
- [ ] Save: `results/template_verification.md` with one formatted example per model

**Deliverable:** `results/template_verification.md`

## Task 2: Run SFT on Each Candidate

**Goal:** Three fine-tuned checkpoints using identical hyperparameters.

**Requirements:**
- [ ] Run SFT for R1-Distill-Qwen-7B:
  - LR: 2e-4, LoRA r=64, alpha=128, steps=2400, max_seq_len=2048
  - Log to W&B run `w75-r1distill-sft`
  - Save checkpoint: `checkpoints/r1distill-sqlcoder/`
- [ ] Run SFT for Llama 3.1 8B Instruct (same hyperparameters)
  - Log to W&B run `w75-llama31-sft`
  - Save checkpoint: `checkpoints/llama31-sqlcoder/`
- [ ] (Optional if time permits) Run SFT for Gemma 2 9B
- [ ] For each run: record final training loss, time elapsed, GPU memory peak
- [ ] Verify each checkpoint loads without error before evaluating

**Deliverable:** Three checkpoint directories + W&B runs linked in `results/base_model_comparison.md`

**Hints:** If SFT for Gemma 2 9B runs OOM: reduce max_seq_len to 1024 or use gradient checkpointing. Gemma's sliding window attention has different memory scaling than standard attention.

## Task 3: Evaluate All Four Models

**Goal:** Comparable accuracy numbers for the full comparison table.

**Requirements:**
- [ ] Run `eval.py` on all four models (existing Qwen + 3 candidates) with identical settings:
  - temperature=0.1, max_tokens=512, same prompt template as training
  - benchmark: `data/custom_200.json`
- [ ] For each model, record: exact-match accuracy, mean generation time per query, peak VRAM
- [ ] Run the two best models (by Custom-200 accuracy) on BIRD-SQL dev for a cross-benchmark comparison
- [ ] Spot-check 5 examples where the new best model is right but your existing model is wrong — what type of queries improved?

**Deliverable:** `results/base_model_comparison.md` with full comparison table

## Task 4: Decision and Next Steps

**Goal:** A documented decision on whether to switch base models.

**Requirements:**
- [ ] Write a 200-word "model selection memo" in `results/base_model_comparison.md`:
  - Name the winner and the margin over existing Qwen2.5 model
  - State whether the improvement justifies: (a) quantizing and pushing the new model to Hub as v2, (b) using it as the base for Weeks 76–77
  - Identify the one biggest risk of switching models for Weeks 76–77 (compounding changes, new bugs)
- [ ] If the new model wins by ≥ 2 pp: quantize to Q4_K_M GGUF and push to `<handle>/postgres-sqlcoder-v2-Q4_K_M-GGUF`
- [ ] If the new model wins by < 2 pp: document "not a significant improvement" and continue with existing model

**Deliverable:** Model selection memo in `results/base_model_comparison.md` + optional Hub push

## Stretch Goals

- Run a brief DPO fine-tuning (400 steps) on the winning new model to see if the full pipeline gap narrows or widens
- Test the 50-example subset method for fast base model screening: run 50 examples before full 200-example eval to save GPU time
- Write a 300-word "base model ablation" section for your technical report
