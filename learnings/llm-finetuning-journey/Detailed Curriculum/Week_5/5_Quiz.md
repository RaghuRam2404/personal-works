# Week 5 — Quiz

---

**Q1.** You are training a transformer on Spider SQL data with AdamW (`lr=3e-4`, `weight_decay=0.01`). After 100 steps, the loss is decreasing smoothly. At step 101, the loss spikes to 4× its previous value and then slowly recovers. The gradient norm before clipping at step 101 is 47.3 (normal is ~0.8). What happened, and what is the correct fix?

A) The LR is too high. Reduce by 10× and restart.
B) A batch with extreme values caused a gradient explosion. The gradient clipping threshold should have been set (e.g., `max_norm=1.0`). With proper clipping, this spike would not occur.
C) AdamW's weight decay caused the spike. Set `weight_decay=0.0`.
D) The warmup was too short. Restart with 500 warmup steps.

---

**Q2.** Why does AdamW apply weight decay directly to the parameter, while standard Adam applies it as part of the gradient (L2 regularization)?

A) AdamW's approach is faster to compute.
B) In Adam, the adaptive scaling `1/sqrt(v̂_t)` divides the weight decay term, so high-gradient parameters receive less effective regularization. AdamW bypasses the adaptive scaling for weight decay, making it uniform across all parameters.
C) L2 regularization is equivalent to weight decay in all optimizers; AdamW is just a naming convention.
D) AdamW's weight decay is applied before the gradient step to avoid numerical instability.

---

**Q3.** The cosine LR schedule reaches `min_lr=0` at step `T_max`. What happens if you continue training beyond `T_max` without resetting the schedule?

A) The LR stays at 0, and the optimizer effectively stops learning.
B) The cosine schedule wraps around and LR begins increasing again.
C) PyTorch raises an error.
D) The LR reverts to the initial `max_lr`.

---

**Q4.** You enable AMP with `autocast()` and `GradScaler`. You notice `scaler.get_scale()` decreasing over the first 100 steps. What does this indicate?

A) The model is converging faster due to mixed precision.
B) Gradient overflow (NaN or Inf) is being detected frequently, causing the scaler to reduce the scale factor to prevent future overflow. This suggests numerical instability — check for log(0) or very large pre-activation values.
C) The GradScaler is working correctly — scale always decreases during warmup.
D) FP16 has insufficient range for your model. Switch to BF16.

---

**Q5.** You implement the LR warmup manually using `optimizer.param_groups[0]['lr'] = get_lr(step)`. You also have a `CosineAnnealingLR` scheduler and call `scheduler.step()` each step. What is the bug?

A) No bug — the manual assignment and scheduler work independently.
B) Both the manual assignment and `scheduler.step()` update the LR at each step, so the LR is doubly modified. The scheduler overrides your manual assignment (or vice versa). Remove one or the other.
C) `param_groups` does not support manual LR assignment.
D) The bug only manifests at validation time.

---

**Q6.** Your CIFAR-10 CNN with AdamW reaches 77% test accuracy. You try increasing weight decay from 0.01 to 0.1. Train accuracy drops to 71% and test accuracy drops to 70%. What does this tell you?

A) Your model is now underfitting — too much regularization. Reduce `weight_decay` back toward 0.01.
B) The model needed even more regularization. Increase weight decay to 1.0.
C) Weight decay is not helping — use dropout instead.
D) The learning rate needs to decrease proportionally to the weight decay increase.

---

**Q7 (short answer).** You train two runs: `run_A` uses constant LR=1e-3, `run_B` uses warmup (200 steps) + cosine decay from 1e-3 to 0. Both run for 2000 steps. Sketch (in words) what the loss curve of each looks like, and explain which run you expect to converge to a lower final loss and why.

---

**Q8 (short answer).** Explain why the `GradScaler.unscale_()` call must be made before `clip_grad_norm_()` in an AMP training loop. What happens if you clip before unscaling?
