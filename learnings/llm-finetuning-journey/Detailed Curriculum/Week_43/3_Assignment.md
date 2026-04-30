# Week 43 Assignment — Derive DPO and Run DPO Training

## Setup Checklist

- [ ] Colab Pro (DPO on ultrafeedback_binarized with a small model fits in 16GB VRAM)
- [ ] Packages: `trl>=0.8.0`, `transformers>=4.38`, `datasets`, `peft`, `unsloth` (optional for speed)
- [ ] HuggingFace account — you will push the trained model
- [ ] Read the DPO paper Appendix A.1 before coding

---

## Task 1 — Derive the DPO Loss on Paper

**Goal:** Reproduce the DPO derivation from scratch without looking at the paper.

**Requirements:**
- Start with the KL-constrained RL objective: max_π E[r] − β·KL(π||π_ref)
- Show that the optimal policy is π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)
- Invert to express r(x,y) in terms of π* and π_ref
- Substitute into the Bradley-Terry preference loss
- Arrive at the DPO loss: -log σ(β · (log_ratio_w − log_ratio_l))
- Write or type the derivation. Minimum 5 numbered steps.

**Deliverable:** `week-43-dpo/dpo_derivation.md` (or scanned handwritten PDF)

**Hints:**
- The variational calculus step (finding the optimal π that maximizes the objective) is shown in Appendix A.1. If you get stuck, read that section but try each step yourself first.
- The partition function Z(x) cancels when you subtract r(y_w) - r(y_l) because Z depends only on x.

---

## Task 2 — Run DPO on ultrafeedback_binarized

**Goal:** Train a DPO model on a general preference dataset end-to-end.

**Requirements:**
- Use TRL's `DPOTrainer` with a model of your choice: `Qwen/Qwen2.5-0.5B-Instruct` or `microsoft/Phi-3-mini-4k-instruct` (small enough for Colab Pro)
- Dataset: `HuggingFaceH4/ultrafeedback_binarized` (already formatted as chosen/rejected pairs)
- Training config:
  - β = 0.1 (start here)
  - Learning rate: 5e-7 (DPO uses much lower LR than SFT)
  - Batch size: 2 (per device), gradient accumulation: 8
  - Max prompt length: 512, max completion length: 256
  - Use LoRA (r=16, α=32) for memory efficiency
  - Log to W&B project `week-43-dpo`
- Monitor these metrics during training:
  - `rewards/chosen` and `rewards/rejected` — should diverge (chosen should increase, rejected decrease)
  - `logps/chosen` and `logps/rejected` — log probabilities under the training model
  - `reward_margin` = chosen_reward - rejected_reward — should increase over training
- Run for at least 500 steps

**Deliverable:** 
- Pushed model to HF Hub: `<your-handle>/dpo-test-v1`
- W&B training curves screenshot: `week-43-dpo/training_curves.png`
- GitHub commit: `week-43-dpo`

---

## Task 3 — β Sensitivity Analysis

**Goal:** Understand how β controls the trade-off between preference learning and staying close to the reference model.

**Requirements:**
- Run 3 short DPO experiments (100 steps each) with β = 0.01, 0.1, 1.0
- Same model, same dataset, same learning rate
- Record `reward_margin` at step 100 for each β
- Write a 3-sentence explanation of what you observe and why

**Deliverable:** `week-43-dpo/beta_analysis.md`

---

## Stretch Goals

- After training, evaluate the DPO model vs the base model by generating 10 completions for prompts from the ultrafeedback dataset. Does the DPO model produce noticeably better responses?
- Examine "refusal rate" — count how many of your 50 generated completions start with "I cannot" or "I'm sorry". Does DPO increase or decrease refusals relative to the base model?
- Read the TRL `DPOTrainer.compute_loss()` method and trace exactly how it computes log_ratio_w and log_ratio_l.
