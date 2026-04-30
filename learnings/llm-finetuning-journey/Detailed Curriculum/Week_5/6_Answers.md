# Week 5 — Answers

---

**Q1. Answer: B**

A gradient norm of 47.3 (versus a typical 0.8) is a textbook gradient explosion — likely triggered by a batch with extreme input values, a very high loss value, or a numerical instability in the model. Gradient clipping with `max_norm=1.0` would have rescaled the gradients from 47.3 down to 1.0 before the optimizer step, and the spike would not have occurred. The LR is not the cause here — the problem is the gradient magnitude, not the step size applied to a well-scaled gradient.

**Why others are wrong:**
- A: Reducing LR would help with sustained instability but would not prevent this isolated spike caused by a specific bad batch.
- C: Weight decay does not cause gradient spikes — it adds a small term to the gradient each step.
- D: Warmup affects the first N steps; the spike at step 101 is unrelated to warmup length.

---

**Q2. Answer: B**

In standard Adam, the L2 penalty is added to the gradient: `g_t' = g_t + λθ_{t-1}`. This modified gradient then passes through the adaptive scaling: `update = lr * m̂_t' / (sqrt(v̂_t') + ε)`. The weight decay term `λθ` is divided by `sqrt(v̂)`, which means parameters with historically large gradients (common tokens in embeddings, early conv filters) receive weaker effective regularization. AdamW applies the decay directly to the parameter, bypassing the adaptive scaling entirely, giving consistent regularization across all parameter types.

---

**Q3. Answer: A**

`CosineAnnealingLR` with `T_max=2000` reaches `eta_min` (default 0) at step 2000. Beyond that, the PyTorch implementation continues the cosine formula, which in the standard `CosineAnnealingLR` wraps to a new cycle (it is a half-cosine, so at T_max, cos(π) = -1, giving LR = eta_min; past T_max it would go back up in the next cycle). However, if you manually set LR to 0 via a custom schedule, the optimizer will not update — which is typically the desired behavior when you want to stop training.

The correct answer for most production code: use `CosineAnnealingWithRestarts` if you want cycling, or simply stop training at T_max. Continuing with LR=0 is safe but wasteful.

---

**Q4. Answer: B**

The `GradScaler` starts with a large scale factor (default 65536) and halves it whenever it detects NaN/Inf in the unscaled gradients during a step. A decreasing scale means the scaler is frequently detecting overflow. Possible causes: very large learning rate, activations that are too large (check for activation norms >> 10), or a numerical issue in the model (log of small value, division by near-zero). Investigate by logging activation norms and gradient norms separately, and check for NaN in the forward pass.

---

**Q5. Answer: B**

Both mechanisms modify `optimizer.param_groups[0]['lr']`. Your manual assignment sets it to `get_lr(step)`. Then `scheduler.step()` overwrites it with the scheduler's own computation. The final LR applied is whatever ran last. This is a silent bug — you see no error, but the LR trace in W&B will be unexpected. Fix: use only one LR control mechanism. Either use `scheduler` and remove the manual assignment, or remove the scheduler and keep the manual function.

---

**Q6. Answer: A**

When train accuracy and test accuracy both drop and they are now similar (71% vs 70%), the model is underfitting — regularization is suppressing the model's ability to fit the training data. Weight decay 0.1 is very strong for AdamW (it subtracts 10% of the parameter value every step). This forces weights toward zero too aggressively, reducing model capacity. Reduce weight decay back to 0.01 or try a value in between (0.01–0.05 is the typical range for CIFAR-10).

---

**Q7 (short answer — model answer):**

`run_A` (constant LR=1e-3): The loss decreases quickly from the start but shows oscillation or noise around the minimum, especially in the last 500–800 steps. The optimizer keeps taking large steps even when close to the minimum, bouncing around it rather than converging into it. The final loss is moderate.

`run_B` (warmup + cosine): The loss starts slowly (near-zero LR for first 200 steps), then increases in step size as warmup completes, decreasing loss steadily. In the final 500–800 steps, the LR has decayed to near zero, and the loss converges very smoothly into a sharper minimum. The final loss is lower than `run_A`.

`run_B` converges to a lower final loss because cosine decay allows the optimizer to refine its solution at the end of training. With constant LR, the optimizer's step size is always "too big" for the precise adjustments needed near a minimum. LR schedules are essentially free compute — always use them.

---

**Q8 (short answer — model answer):**

During AMP training, the forward pass uses FP16 and the backward pass produces FP16 gradients. The `GradScaler` multiplies the loss by a large scale factor (e.g., 65536) before `backward()` to prevent FP16 underflow. The actual gradients stored in `.grad` are therefore the true gradients multiplied by 65536.

If you call `clip_grad_norm_(model.parameters(), max_norm=1.0)` before `scaler.unscale_(optimizer)`, you are computing the norm of the scaled gradients (65536× the true magnitude). The threshold comparison is completely wrong: a gradient that should be clipped (true norm = 50) now has apparent norm = 50 * 65536 = 3.2M, and one that should not be clipped (true norm = 0.5) has apparent norm = 32768. The clip threshold of 1.0 will clip nearly everything. The fix: call `scaler.unscale_(optimizer)` first (which divides all gradients by the scale factor), then clip with the correct true gradient norms, then call `scaler.step(optimizer)`.
