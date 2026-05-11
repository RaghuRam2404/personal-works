# Week 5 — Optimization, LR Schedules, and Reading Loss Curves

## Learning Objectives

By the end of this week, you will be able to:

- Explain SGD, Momentum, Adam, and AdamW mathematically and state when to prefer each.
- Implement a linear warmup + cosine decay LR schedule from scratch.
- Explain why weight decay in Adam is not equivalent to L2 regularization, and what AdamW fixes.
- Apply gradient clipping correctly and explain why it is needed independently of the optimizer.
- Enable PyTorch AMP (automatic mixed precision) training and verify it speeds up GPU-side compute.
- Read a W&B loss curve and diagnose: LR too high, LR too low, overfitting, gradient explosion.

---

## Concepts

### SGD, Momentum, and the Optimization Landscape

**Vanilla SGD:** `θ ← θ - lr * g` where `g = ∇L(θ)`. Simple, but sensitive to the LR and slow in ravines (directions with high curvature).

**SGD with momentum:** Maintains a velocity vector `v` that accumulates a fraction of past gradients:
```
v_t = β * v_{t-1} + g_t
θ_t = θ_{t-1} - lr * v_t
```
Momentum (typical `β = 0.9`) dampens oscillations across steep directions and accelerates movement along consistent gradient directions. SGD+momentum is still the optimizer of choice for training CNNs from scratch (it often generalizes better than Adam on vision tasks).

### Adam

Adam (Adaptive Moment Estimation) maintains per-parameter estimates of the first moment (mean gradient) and second moment (uncentered variance):

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t           # first moment
v_t = β2 * v_{t-1} + (1 - β2) * g_t²          # second moment
m̂_t = m_t / (1 - β1^t)                        # bias correction
v̂_t = v_t / (1 - β2^t)                        # bias correction
θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
```

**Default hyperparameters:** `β1=0.9`, `β2=0.999`, `ε=1e-8`, `lr=1e-3`.

**Why Adam wins for NLP:** The adaptive per-parameter learning rates handle the sparse gradient updates in embedding layers well. When most embedding rows are zero-gradient (because those tokens were not in the batch), Adam's second moment estimate is small, which allows a large effective update when they finally appear. SGD treats all parameters equally, making embedding updates too large or too small depending on frequency.

**Adam's weakness on CNNs:** Adam with weight decay conflates the weight decay term with the adaptive scaling, which corrupts the regularization effect. This is what AdamW fixes.

### AdamW — Decoupled Weight Decay

The bug in Adam: Adam's weight decay implementation in most frameworks adds L2 regularization to the gradient before the adaptive scaling:

```
θ_t = θ_{t-1} - lr * (m̂_t / sqrt(v̂_t) + ε) + lr * λ * θ_{t-1}
```

The `λ * θ_{t-1}` term gets divided by the adaptive factor `sqrt(v̂_t)`, so parameters with larger gradient history receive less effective weight decay. High-frequency parameters (common embeddings, early conv layers) are under-regularized relative to low-frequency ones.

**AdamW** decouples weight decay from the adaptive scaling:
```
θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε) - lr * λ * θ_{t-1}
```

The weight decay is applied directly to the parameter, independently of the gradient history. This is correct L2 regularization. **Use AdamW by default for all LLM fine-tuning.** Typical `λ = 0.01`–`0.1`.

The reference: [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, 2019.

### Learning Rate Schedules

**Why schedules matter:** A constant LR is suboptimal. Too high at the start → early instability. Too high at the end → the optimizer overshoots the minimum and oscillates rather than converging.

**Linear warmup:** Start at LR=0 (or very small), linearly increase to the target LR over a warmup period. Prevents early divergence when gradients are noisy and model is far from any useful region.

**Cosine decay:** After warmup, decay the LR following a cosine curve:
```
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T_max))
```
where `t` is the current step and `T_max` is the total training steps. This produces a smooth, gradual decay that tends to find better minima than linear decay.

**Warmup length rule of thumb:** 5–10% of total training steps for fine-tuning. For pre-training from scratch: 1–2% (LLMs use ~2000 steps of warmup for 100K+ total steps).

**Implementation in PyTorch:**

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

warmup = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
```

### Gradient Clipping

Gradient clipping limits the norm of the gradient vector before the optimizer step:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This does not clip per-parameter gradients — it clips the global gradient norm. If the norm exceeds `max_norm`, all gradients are scaled by `max_norm / norm`.

Clipping is not an optimizer feature — it should be applied regardless of optimizer choice, especially for: transformers, RNNs, and early training when the model is far from convergence.

**When to use:** Almost always in NLP. The standard for transformer pre-training is `max_norm=1.0`. For fine-tuning LoRA adapters: `max_norm=1.0` is safe.

### Mixed Precision Training (AMP)

Modern GPUs (A100, T4, V100) have hardware support for float16 (FP16) and bfloat16 (BF16) matrix multiplications that are 2–8× faster than FP32. However, FP16 has lower dynamic range (max value ≈ 65504), which can cause overflow or underflow during training.

PyTorch's Automatic Mixed Precision (AMP) handles this automatically:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for x, y in loader:
    optimizer.zero_grad()
    with autocast():               # forward pass in FP16/BF16
        loss = model(x)
    scaler.scale(loss).backward()  # scale loss to avoid underflow
    scaler.step(optimizer)         # unscale before optimizer step
    scaler.update()                # adjust scale factor
```

The `GradScaler` maintains a dynamic scale factor. If gradients overflow (NaN/Inf), it skips the optimizer step and reduces the scale. If no overflow occurs for several consecutive steps, it increases the scale to recover precision.

**For Apple Silicon (MPS):** Use `torch.autocast(device_type='mps', dtype=torch.float16)`. The `GradScaler` is not supported on MPS as of PyTorch 2.x — you can use half precision for the forward pass only.

### Reading Loss Curves

| Pattern | Diagnosis |
|---|---|
| Loss starts high, decreases then flat | Good training; may need more epochs |
| Loss oscillates or spikes then recovers | LR too high; reduce by 5–10× |
| Loss diverges (exponential increase) | LR far too high or gradient explosion; clip gradients |
| Loss decreases but val loss increases | Overfitting; add regularization |
| Loss decreases very slowly, nearly flat | LR too low; increase by 3–5× |
| Loss goes to NaN in first few steps | Numerical instability; check for log(0), incorrect normalization |
| Train loss good but val loss never decreases | Data leakage, or wrong val set |

---

## Connections

**Builds on:** Week 1's training loop. Week 3's CNN (you re-train it this week with better optimization). Week 4's gradient clipping (introduced for RNNs, now universalized).

**Unlocks:** Every future training run. AdamW + cosine schedule + gradient clipping is the canonical optimizer setup for all LLM fine-tuning in Phases 4–6. Mixed precision (FP16/BF16) is mandatory for training large models on limited compute.

---

## Common Misconceptions and Pitfalls

- **"Adam is always better than SGD."** On vision tasks with CNNs, SGD+momentum with a good LR schedule often generalizes better than Adam. Use Adam/AdamW for NLP; consider SGD+momentum for pure vision.
- **"Weight decay and L2 regularization are the same in Adam."** They are the same in SGD. In Adam, they are different (see AdamW above).
- **"Gradient clipping prevents large gradients from helping."** Clipping scales all gradients proportionally — the direction is preserved, only the magnitude is capped. The signal is still useful.
- **"Mixed precision always speeds things up."** On CPU, FP16 is often slower (CPUs optimize for FP32). On MPS, gains are mixed. The speedup is reliable on CUDA (V100, A100, T4).

---

## Time Allocation (6–8 hours this week)

| Activity | Time |
|---|---|
| Read Ruder's optimizer overview (all sections) | 1 h |
| Read AdamW paper intro + algorithm box | 30 min |
| Watch Yannic Kilcher's AdamW explanation (~25m) | 30 min |
| Implement warmup + cosine schedule from scratch | 1 h |
| Add AdamW + clipping + AMP to Week 3 CNN | 1.5 h |
| Generate W&B comparison report | 1 h |
| Journal + commit | 30 min |
