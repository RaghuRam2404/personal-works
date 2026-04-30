# Week 30 — LoRA: The Math and the Intuition

## Learning Objectives

By the end of this week, you will be able to:

- Derive the LoRA weight update formula and count the trainable parameters for a given rank r
- Implement a `LoraLinear` module in PyTorch from scratch that passes a correctness test
- Explain why fine-tuning updates are intrinsically low-rank and what experimental evidence supports this
- Apply LoRA to a simple GPT-2 or nanoGPT model and verify it trains correctly
- Articulate the trade-off between rank r, parameter count, and model expressiveness

---

## Concepts

### 1. The Problem LoRA Solves

Full SFT on a 7B model requires storing and updating 7 billion parameters — plus gradients and optimizer states, which pushes memory requirements above 80GB. Even for a 0.5B model, full SFT is inefficient: most of those parameter updates are redundant.

LoRA (Low-Rank Adaptation) solves this by making a key observation: **you do not need to update the full weight matrix W ∈ R^(d_out × d_in). You only need to update a low-rank correction matrix ∆W.**

Instead of storing ∆W (which has d_out × d_in parameters), LoRA stores two small matrices:
- A ∈ R^(r × d_in) (initialized with random Gaussian)
- B ∈ R^(d_out × r) (initialized to zero)

Such that ∆W = BA.

The forward pass becomes:
```
y = Wx + (BA)x * (alpha / r)
```

Where W is frozen (no gradient), and only A and B are trained.

### 2. The Math

For a linear layer with weight W ∈ R^(d_out × d_in):

**Standard forward pass:** y = Wx

**LoRA forward pass:** y = Wx + (B A x) * (alpha / r)

**Parameter count:**
- Full fine-tuning: d_out × d_in parameters updated
- LoRA rank r: r × d_in (matrix A) + d_out × r (matrix B) = r × (d_in + d_out) parameters updated

For a typical attention projection with d_in = d_out = 4096 and r = 16:
- Full: 4096 × 4096 = 16,777,216 parameters
- LoRA: 16 × (4096 + 4096) = 131,072 parameters — about 128x fewer

**Why initialize B = 0?** At the start of training, ∆W = B A = 0 × A = 0. So the LoRA model starts as the identical to the pretrained model. Gradients flow through B first (since A is non-zero), and B learns to shape the correction direction.

**The alpha/r scaling factor:** The `alpha` hyperparameter controls the magnitude of the LoRA update relative to the pretrained weights. With alpha = 2r, the LoRA update is scaled by 2 at full rank — a common default. The effective learning rate for LoRA adapters is `lr * alpha / r`, so if you change r but keep alpha constant, you implicitly change the effective scale. Many practitioners set alpha = r (scale = 1) or alpha = 2r.

### 3. Why Does Low-Rank Work? The Intrinsic Dimensionality Hypothesis

Aghajanyan et al. (2020) showed that the loss landscape of fine-tuned NLP models has very low intrinsic dimensionality — meaning the gradient updates that matter live in a much smaller subspace than the full parameter space. Hu et al. extended this to show that weight change matrices ∆W from full fine-tuning experiments have low effective rank.

Intuition: a pretrained model already knows language, reasoning, and many task patterns. Fine-tuning for a specific task (like text-to-SQL) is more like "rotating a projection" than "learning from scratch." The rotation requires adjusting only a few principal directions of the weight matrix, not the entire matrix.

For your SQL task: Qwen2.5-Coder already knows SQL syntax and PostgreSQL semantics. The SFT signal mostly needs to shift the attention patterns that map natural language question tokens to SQL clause tokens. This is a low-rank operation.

### 4. Which Layers to Apply LoRA To

The original LoRA paper applied adapters to the attention query and value projections (Wq and Wv). Later work (including Sebastian Raschka's empirical analysis) showed that applying LoRA to all linear layers — q, k, v, o (output projection), and the MLP layers — consistently outperforms applying it to only q and v.

For Qwen2.5, the relevant module names are typically:
- `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention)
- `gate_proj`, `up_proj`, `down_proj` (MLP, SwiGLU variant)

You will configure this in Week 31 via `target_modules`. This week, focus on implementing the math correctly.

### 5. Implementing LoRA from Scratch

The implementation is remarkably simple:

```python
import torch
import torch.nn as nn
import math

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Frozen pretrained weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        pretrained_out = x @ self.weight.T
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return pretrained_out + lora_out * self.scaling
```

Key correctness checks: (1) `self.weight` has `requires_grad=False` — it never updates. (2) `lora_B` is initialized to zero — at step 0, `lora_out = 0` for all inputs. (3) The forward pass is W x + (B A x) * (alpha/r) — same as the paper.

### 6. Applying LoRA to nanoGPT

To apply your `LoraLinear` to nanoGPT, you replace the attention `c_attn` and `c_proj` linear layers with `LoraLinear` modules. Then freeze all original parameters and verify only LoRA parameters have gradients.

```python
# Count trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
```

---

## Connections

**Builds on:** Week 29's SFT training loop; Phase 2's nanoGPT implementation; linear algebra background (matrix rank, SVD).

**Needed for:** Week 31 (peft library wraps this math into a clean API); Week 33 (QLoRA adds 4-bit quantization on top of LoRA); Week 36 (DoRA, RSLoRA are variants of this same formula).

---

## Common Misconceptions / Pitfalls

- **"LoRA is a new architecture."** No — LoRA keeps the original architecture intact. It adds adapter matrices on the side. At inference time, you can merge B A into W for zero-overhead inference: W_merged = W + B A * (alpha/r).
- **"Higher rank is always better."** Not necessarily. Higher rank trains more parameters but may overfit on small datasets. You will sweep this in Week 31.
- **"alpha is like learning rate."** Conceptually similar but not identical. alpha scales the output of the LoRA path; learning rate scales the gradient step. They interact: with high alpha and high LR, the LoRA update can overpower the pretrained weights.
- **"B initialized to zero means nothing is learned at step 1."** Correct — but gradients still flow. The gradient of the loss w.r.t. B is nonzero at initialization because A is nonzero, so B starts learning immediately from step 1.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read LoRA paper fully (arxiv 2106.09685) | 2h |
| Watch Yannic Kilcher LoRA video (25m) + Raschka video (1h) | 1.5h |
| Implement `LoraLinear` from scratch | 1.5h |
| Apply LoRA to nanoGPT, verify trainable param count | 1h |
| Write blog-post-style writeup | 30m |
| Commit to GitHub | 15m |
