# Week 30 Assignment — LoRA from Scratch

## Setup Checklist

- [ ] PyTorch installed (Mac MPS or Colab Free T4)
- [ ] Your nanoGPT from Phase 2 available, or clone a reference implementation
- [ ] LoRA paper saved: https://arxiv.org/abs/2106.09685
- [ ] No new packages needed — pure PyTorch only this week

---

## Task 1 — Implement `LoraLinear`

**Goal:** Build a correct, tested LoRA linear layer from scratch.

**Requirements:**
- Create `lora.py` with a `LoraLinear(nn.Module)` class
- Constructor arguments: `in_features`, `out_features`, `rank`, `alpha`, `dropout=0.0`
- The pretrained weight must be stored as `nn.Parameter` with `requires_grad=False`
- `lora_A` initialized with Kaiming uniform (or standard Gaussian)
- `lora_B` initialized to exactly zero
- Forward pass: `output = x @ W.T + (x @ A.T @ B.T) * (alpha / rank)`
- Write a correctness test in `test_lora.py`:
  - At initialization, verify that `lora_B` is all zeros
  - Verify that `forward(x)` with a zero `lora_B` equals `x @ W.T` (pretrained output only)
  - Verify that the module has `requires_grad=False` on the weight and `requires_grad=True` on `lora_A` and `lora_B`
  - Run one gradient step and verify `lora_B` is no longer zero
  - Verify `weight` did not change after the gradient step

**Deliverable:** `lora.py` and `test_lora.py` committed, with `python test_lora.py` printing "All tests passed."

---

## Task 2 — Derive the Parameter Count

**Goal:** Prove the parameter efficiency of LoRA on paper (or in a markdown file).

**Requirements:**
- Write `week30_derivation.md` containing:
  - The formula for trainable parameters in LoRA: `r × (d_in + d_out)`
  - Calculated for three Qwen2.5-7B attention projections: q_proj, k_proj, v_proj with d_in = d_out = 4096 and r = 16
  - Calculated for the MLP layers: gate_proj with d_in = 4096, d_out = 11008, r = 16
  - The total LoRA parameter count across all 28 transformer layers for those 4 layer types at rank 16
  - The percentage of total 7B parameters that LoRA at rank 16 represents
- Show the formula: `ratio = (num_lora_params / num_total_params) * 100`

**Deliverable:** `week30_derivation.md` with all calculations shown.

**Hints:**
- Qwen2.5-7B has 28 transformer layers
- Use `model.num_parameters()` or count manually
- Expected answer: rank-16 LoRA on all linear layers ≈ 0.5–2% of total parameters

---

## Task 3 — Apply LoRA to nanoGPT

**Goal:** Replace linear layers in your Phase 2 nanoGPT with `LoraLinear` and verify training works.

**Requirements:**
- Load your nanoGPT (or use a minimal GPT-2 reference from Phase 2)
- Write a function `apply_lora(model, rank=8, alpha=16, target_modules=["c_attn", "c_proj"])` that replaces specified linear layers with `LoraLinear`
- After `apply_lora`, freeze all non-LoRA parameters: `for p in model.parameters(): p.requires_grad = False` then un-freeze LoRA params: `for name, p in model.named_parameters(): if "lora" in name: p.requires_grad = True`
- Print: total parameters, trainable parameters, trainable percentage
- Run 100 training steps on any text (Shakespeare from Phase 2 is fine)
- Log loss at step 0 and step 100 — verify loss decreases

**Deliverable:** `apply_lora.py` and `week30_results.md` with parameter counts and step-0 vs step-100 loss. GitHub commit: `week-30-lora-from-scratch`.

**Hints:**
- In nanoGPT, `c_attn` is a single linear that computes Q, K, V simultaneously with `in_features=n_embd, out_features=3*n_embd` — your LoraLinear needs to handle this correctly
- You can verify LoRA is working by checking: `sum(p.numel() for p in model.parameters() if p.requires_grad)`

---

## Task 4 — Merge LoRA Weights

**Goal:** Demonstrate that LoRA can be merged into the base weights for zero-overhead inference.

**Requirements:**
- Add a `merge()` method to `LoraLinear`:
  - Computes `W_merged = W + (B @ A) * (alpha / rank)` in-place on `self.weight`
  - Sets `lora_A` and `lora_B` to zero after merging (so the LoRA path contributes nothing)
- After 100 training steps, run inference on a test input with the LoRA path active
- Then call `merge()` and run inference on the same test input
- Verify the outputs are identical (within floating-point tolerance)

**Deliverable:** Section in `week30_results.md` showing merge outputs match.

---

## Task 5 — Blog-Post Writeup

**Goal:** Force yourself to articulate the math in plain English.

**Requirements:**
- Write `week30_writeup.md` (500–800 words) explaining:
  - What problem LoRA solves
  - The math of the forward pass with intuition
  - Why B=0 initialization is important
  - What "rank" means intuitively (think: how many independent directions the adapter can express)
  - When you would choose rank 8 vs. rank 64

**Deliverable:** `week30_writeup.md` committed.

---

## Stretch Goals

- Implement a `LoraEmbedding` class that adapts embedding layers using the same principle
- Verify that training a rank-1 LoRA for 500 steps on a tiny dataset produces a lower loss than a rank-64 LoRA for 50 steps (same total parameter updates)
- Read Appendix D of the LoRA paper and understand why the authors chose Wq and Wv (not Wk) for their initial experiments
